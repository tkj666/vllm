# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.expert_cache import (
    ExpertWeightBundle,
    allocate_streamed_expert_host_storage,
    get_streamed_expert_cache_load_token,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)
from vllm.model_executor.models.utils import extract_layer_index

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


logger = init_logger(__name__)

MXFP4_MARLIN_FORMAT_CLASS = "marlin-mxfp4-w13-w2-scales-v1"
_PINNED_STORAGE_ALIGNMENT_BYTES = 256
_TENSOR_NAMES = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
)


@dataclass(frozen=True)
class Mxfp4RuntimeTensorSpec:
    """Shape and dtype of one expert's tensor in the GPU arena."""

    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class ConvertedMxfp4Experts:
    """Marlin-formatted host experts ready for generic arena binding."""

    format_class: str
    host_bundles: Mapping[int, ExpertWeightBundle]
    runtime_specs: Mapping[str, Mxfp4RuntimeTensorSpec]
    hidden_size: int
    activation_dtype: torch.dtype
    _runtime_tensors: Mapping[str, torch.Tensor]

    def runtime_tensor(self, name: str) -> torch.Tensor:
        """Return the full pinned runtime view for one tensor kind."""
        try:
            return self._runtime_tensors[name]
        except KeyError as exc:
            raise KeyError(f"unknown streamed MXFP4 tensor {name!r}") from exc


@dataclass(frozen=True)
class _Mxfp4TensorLayout:
    load_shape: tuple[int, ...]
    load_dtype: torch.dtype
    runtime_shape: tuple[int, ...]
    runtime_dtype: torch.dtype

    @property
    def load_nbytes(self) -> int:
        return torch.Size(self.load_shape).numel() * self.load_dtype.itemsize

    @property
    def runtime_nbytes(self) -> int:
        return torch.Size(self.runtime_shape).numel() * self.runtime_dtype.itemsize


class _LayerState(Enum):
    CLAIMED = auto()
    CONVERTING = auto()
    CONVERTED = auto()
    FAILED = auto()


def streamed_mxfp4_cache_enabled() -> bool:
    """Return whether the current model load enables streamed experts."""
    try:
        vllm_config = get_current_vllm_config()
    except Exception:
        return False
    return vllm_config.offload_config.expert_cache_enabled


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _allocate_pinned_storage(num_bytes: int) -> torch.UntypedStorage:
    return allocate_streamed_expert_host_storage(num_bytes)


def _make_layouts(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    w13_num_shards: int,
) -> Mapping[str, _Mxfp4TensorLayout]:
    if w13_num_shards != 2:
        raise ValueError("streamed MXFP4 Marlin requires a two-shard gated expert MLP")
    if hidden_size % 32 or intermediate_size % 32:
        raise ValueError(
            "streamed MXFP4 Marlin requires hidden and intermediate sizes "
            "divisible by 32"
        )

    layouts = {
        "w13_weight": _Mxfp4TensorLayout(
            load_shape=(num_experts, 2 * intermediate_size, hidden_size // 2),
            load_dtype=torch.uint8,
            runtime_shape=(
                num_experts,
                hidden_size // 16,
                4 * intermediate_size,
            ),
            runtime_dtype=torch.int32,
        ),
        "w2_weight": _Mxfp4TensorLayout(
            load_shape=(num_experts, hidden_size, intermediate_size // 2),
            load_dtype=torch.uint8,
            runtime_shape=(
                num_experts,
                intermediate_size // 16,
                2 * hidden_size,
            ),
            runtime_dtype=torch.int32,
        ),
        "w13_weight_scale": _Mxfp4TensorLayout(
            load_shape=(num_experts, 2 * intermediate_size, hidden_size // 32),
            load_dtype=torch.uint8,
            runtime_shape=(
                num_experts,
                hidden_size // 32,
                2 * intermediate_size,
            ),
            runtime_dtype=torch.float8_e8m0fnu,
        ),
        "w2_weight_scale": _Mxfp4TensorLayout(
            load_shape=(num_experts, hidden_size, intermediate_size // 32),
            load_dtype=torch.uint8,
            runtime_shape=(
                num_experts,
                intermediate_size // 32,
                hidden_size,
            ),
            runtime_dtype=torch.float8_e8m0fnu,
        ),
    }
    for name, layout in layouts.items():
        if layout.load_nbytes != layout.runtime_nbytes:
            raise ValueError(
                f"streamed MXFP4 {name} load/runtime layouts differ in size"
            )
    return MappingProxyType(layouts)


def _configured_layer_ids() -> tuple[int, ...]:
    hf_config = get_current_vllm_config().model_config.hf_text_config
    if getattr(hf_config, "model_type", None) == "deepseek_v4":
        return tuple(range(hf_config.num_hidden_layers))
    raise ValueError("streamed MXFP4 host loading currently supports DeepSeek V4 only")


def _validate_load_strategy() -> None:
    load_config = get_current_vllm_config().load_config
    if load_config.load_format == "dummy":
        return
    if load_config.safetensors_load_strategy != "lazy":
        raise ValueError(
            "streamed MXFP4 expert loading requires "
            "--safetensors-load-strategy lazy to prevent checkpoint page-cache "
            "prefetch from competing with the pinned expert store"
        )


class _StreamedMxfp4HostStore:
    """One pinned storage backing raw and Marlin-formatted expert views."""

    def __init__(
        self,
        owner: object,
        layer_ids: tuple[int, ...],
        layouts: Mapping[str, _Mxfp4TensorLayout],
        hidden_size: int,
        activation_dtype: torch.dtype,
    ) -> None:
        if not layer_ids or len(set(layer_ids)) != len(layer_ids):
            raise ValueError("streamed MXFP4 layers must be non-empty and unique")
        if tuple(layouts) != _TENSOR_NAMES:
            raise ValueError("streamed MXFP4 tensor layouts are incomplete")
        if activation_dtype != torch.bfloat16:
            raise ValueError("streamed MXFP4 experts require bfloat16 activations")

        self._owner = owner
        self._layer_ordinals = {
            layer_id: ordinal for ordinal, layer_id in enumerate(layer_ids)
        }
        self._layouts = MappingProxyType(dict(layouts))
        self._hidden_size = hidden_size
        self._activation_dtype = activation_dtype
        self._states: dict[int, _LayerState] = {}
        self._lock = threading.Lock()
        self._conversion_stream: torch.cuda.Stream | None = None
        self._conversion_device: torch.device | None = None

        offsets: dict[str, int] = {}
        next_offset = 0
        for name in _TENSOR_NAMES:
            next_offset = _align_up(next_offset, _PINNED_STORAGE_ALIGNMENT_BYTES)
            offsets[name] = next_offset
            next_offset += self._layouts[name].load_nbytes
        self._field_offsets = MappingProxyType(offsets)
        self._layer_stride_bytes = _align_up(
            next_offset, _PINNED_STORAGE_ALIGNMENT_BYTES
        )
        storage_bytes = len(layer_ids) * self._layer_stride_bytes
        logger.info(
            "Allocating %.4f GiB of pinned CPU storage for %d streamed "
            "MXFP4 expert layers",
            storage_bytes / 1024**3,
            len(layer_ids),
        )
        self.storage = _allocate_pinned_storage(storage_bytes)

    @property
    def nbytes(self) -> int:
        return self.storage.nbytes()

    def belongs_to(self, owner: object) -> bool:
        return self._owner is owner

    def compatible_with(
        self,
        layouts: Mapping[str, _Mxfp4TensorLayout],
        hidden_size: int,
        activation_dtype: torch.dtype,
    ) -> bool:
        return (
            dict(self._layouts) == dict(layouts)
            and self._hidden_size == hidden_size
            and self._activation_dtype == activation_dtype
        )

    def validate_complete(self) -> None:
        missing = self._layer_ordinals.keys() - self._states.keys()
        if missing:
            raise RuntimeError(
                "streamed MXFP4 host storage is missing configured layers "
                f"{sorted(missing)}"
            )

    def claim_layer(self, layer_id: int) -> Mapping[str, torch.Tensor]:
        self._layer_ordinal(layer_id)
        with self._lock:
            if layer_id in self._states:
                raise RuntimeError(
                    f"streamed MXFP4 host layer {layer_id} was claimed twice"
                )
            self._states[layer_id] = _LayerState.CLAIMED
        return self._layer_views(layer_id, runtime=False)

    def convert_layer(
        self,
        layer_id: int,
        layer: RoutedExperts,
        device: torch.device,
    ) -> ConvertedMxfp4Experts:
        """Convert one layer in bounded GPU scratch and overwrite it in place."""
        self.validate_complete()
        ordinal = self._layer_ordinal(layer_id)
        del ordinal
        with self._lock:
            state = self._states.get(layer_id)
            if state != _LayerState.CLAIMED:
                raise RuntimeError(
                    f"streamed MXFP4 layer {layer_id} cannot convert from {state}"
                )
            self._states[layer_id] = _LayerState.CONVERTING

        stream: torch.cuda.Stream | None = None
        try:
            load_views = self._layer_views(layer_id, runtime=False)
            runtime_views = self._layer_views(layer_id, runtime=True)
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            stream = self._get_conversion_stream(device)
            with torch.cuda.stream(stream):
                num_experts = next(iter(load_views.values())).shape[0]
                for expert_id in range(num_experts):
                    gpu_inputs = {
                        name: torch.empty_like(
                            load_views[name][expert_id], device=device
                        )
                        for name in _TENSOR_NAMES
                    }
                    for name, gpu_input in gpu_inputs.items():
                        gpu_input.copy_(load_views[name][expert_id], non_blocking=True)

                    converted = prepare_moe_mxfp4_layer_for_marlin(
                        layer,
                        gpu_inputs["w13_weight"].unsqueeze(0),
                        gpu_inputs["w2_weight"].unsqueeze(0),
                        gpu_inputs["w13_weight_scale"].unsqueeze(0),
                        gpu_inputs["w2_weight_scale"].unsqueeze(0),
                        None,
                        None,
                    )
                    for name, converted_tensor in zip(_TENSOR_NAMES, converted[:4]):
                        expected = runtime_views[name][expert_id]
                        actual = converted_tensor[0]
                        if (
                            actual.shape != expected.shape
                            or actual.dtype != expected.dtype
                        ):
                            raise RuntimeError(
                                f"Marlin conversion produced incompatible {name}: "
                                f"{tuple(actual.shape)} {actual.dtype}, expected "
                                f"{tuple(expected.shape)} {expected.dtype}"
                            )
                        expected.copy_(actual, non_blocking=True)
            stream.synchronize()
        except Exception as error:
            if stream is not None:
                try:
                    stream.synchronize()
                except Exception as cleanup_error:
                    error.add_note(
                        f"MXFP4 conversion stream cleanup also failed: {cleanup_error}"
                    )
            with self._lock:
                self._states[layer_id] = _LayerState.FAILED
            raise

        with self._lock:
            self._states[layer_id] = _LayerState.CONVERTED

        runtime_specs = MappingProxyType(
            {
                name: Mxfp4RuntimeTensorSpec(
                    shape=layout.runtime_shape[1:],
                    dtype=layout.runtime_dtype,
                )
                for name, layout in self._layouts.items()
            }
        )
        host_bundles = MappingProxyType(
            {
                expert_id: ExpertWeightBundle(
                    MXFP4_MARLIN_FORMAT_CLASS,
                    {name: runtime_views[name][expert_id] for name in _TENSOR_NAMES},
                )
                for expert_id in range(next(iter(runtime_views.values())).shape[0])
            }
        )
        return ConvertedMxfp4Experts(
            format_class=MXFP4_MARLIN_FORMAT_CLASS,
            host_bundles=host_bundles,
            runtime_specs=runtime_specs,
            hidden_size=self._hidden_size,
            activation_dtype=self._activation_dtype,
            _runtime_tensors=MappingProxyType(runtime_views),
        )

    def _get_conversion_stream(self, device: torch.device) -> torch.cuda.Stream:
        if device.type != "cuda":
            raise ValueError("streamed MXFP4 conversion requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.accelerator.current_device_index())
        with self._lock:
            if self._conversion_stream is None:
                self._conversion_stream = torch.cuda.Stream(device=device)
                self._conversion_device = device
            elif self._conversion_device != device:
                raise ValueError(
                    "all streamed MXFP4 layers must convert on one CUDA device"
                )
            return self._conversion_stream

    def _layer_ordinal(self, layer_id: int) -> int:
        try:
            return self._layer_ordinals[layer_id]
        except KeyError as exc:
            raise ValueError(
                f"layer {layer_id} is not a configured streamed MXFP4 layer"
            ) from exc

    def _layer_views(self, layer_id: int, *, runtime: bool) -> dict[str, torch.Tensor]:
        layer_offset = self._layer_ordinal(layer_id) * self._layer_stride_bytes
        views: dict[str, torch.Tensor] = {}
        for name, layout in self._layouts.items():
            shape = layout.runtime_shape if runtime else layout.load_shape
            dtype = layout.runtime_dtype if runtime else layout.load_dtype
            views[name] = self._view(
                layer_offset + self._field_offsets[name], shape, dtype
            )
        return views

    def _view(
        self,
        byte_offset: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        storage_offset, remainder = divmod(byte_offset, dtype.itemsize)
        if remainder:
            raise RuntimeError("streamed MXFP4 host offset is not dtype-aligned")
        return torch.empty(0, dtype=dtype, device="cpu").set_(
            self.storage,
            storage_offset,
            shape,
        )


_STREAMED_MXFP4_HOST_STORES: weakref.WeakValueDictionary[
    int, _StreamedMxfp4HostStore
] = weakref.WeakValueDictionary()
_STREAMED_MXFP4_HOST_STORES_LOCK = threading.Lock()


def streamed_mxfp4_weight_views(
    layer: RoutedExperts,
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    w13_num_shards: int,
    activation_dtype: torch.dtype,
) -> Mapping[str, torch.Tensor]:
    """Claim raw checkpoint views from the model-wide pinned MXFP4 store."""
    _validate_load_strategy()
    layer_id = extract_layer_index(layer.layer_name)
    layer_ids = _configured_layer_ids()
    if layer_id not in layer_ids:
        raise ValueError(f"layer {layer_id} is not a configured streamed MXFP4 layer")
    layouts = _make_layouts(
        num_experts,
        hidden_size,
        intermediate_size,
        w13_num_shards,
    )
    load_token = get_streamed_expert_cache_load_token()
    runtime_key = id(load_token)
    with _STREAMED_MXFP4_HOST_STORES_LOCK:
        store = _STREAMED_MXFP4_HOST_STORES.get(runtime_key)
        if store is None:
            store = _StreamedMxfp4HostStore(
                load_token,
                layer_ids,
                layouts,
                hidden_size,
                activation_dtype,
            )
            _STREAMED_MXFP4_HOST_STORES[runtime_key] = store
        elif not store.belongs_to(load_token):
            raise RuntimeError("stale streamed MXFP4 host-store configuration")
        elif not store.compatible_with(layouts, hidden_size, activation_dtype):
            raise ValueError("all streamed MXFP4 layers must share one expert layout")

        views = store.claim_layer(layer_id)
        layer._streamed_mxfp4_host_store = store
        return views


def convert_streamed_mxfp4_layer(
    layer: RoutedExperts,
    device: torch.device,
) -> ConvertedMxfp4Experts:
    """Convert a loaded layer to pinned Marlin format without full GPU weights."""
    store = getattr(layer, "_streamed_mxfp4_host_store", None)
    if not isinstance(store, _StreamedMxfp4HostStore):
        raise RuntimeError("streamed MXFP4 layer is missing its pinned host store")
    return store.convert_layer(extract_layer_index(layer.layer_name), layer, device)
