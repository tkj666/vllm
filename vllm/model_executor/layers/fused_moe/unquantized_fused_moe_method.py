# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import weakref
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.config import get_current_vllm_config, get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    biased_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.expert_cache import (
    get_streamed_expert_cache_load_token,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    convert_to_unquantized_kernel_format,
    make_unquantized_moe_kernel,
    select_unquantized_moe_backend,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

logger = init_logger(__name__)
_PINNED_STORAGE_ALIGNMENT_BYTES = 256


def _streamed_expert_cache_enabled() -> bool:
    vllm_config = get_current_vllm_config_or_none()
    return bool(
        vllm_config is not None and vllm_config.offload_config.expert_cache_enabled
    )


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _allocate_pinned_storage(num_bytes: int) -> torch.UntypedStorage:
    storage = (
        torch.empty(
            num_bytes,
            dtype=torch.uint8,
            device="cpu",
        )
        .pin_memory()
        .untyped_storage()
    )
    if storage.nbytes() != num_bytes or not storage.is_pinned():
        raise RuntimeError("failed to allocate the streamed expert host store")
    return storage


class _StreamedExpertHostStore:
    """One preallocated pinned storage backing every routed expert parameter."""

    def __init__(
        self,
        owner: object,
        layer_ids: tuple[int, ...],
        w13_shape: tuple[int, ...],
        w2_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        if not layer_ids or len(set(layer_ids)) != len(layer_ids):
            raise ValueError("streamed expert host layers must be non-empty and unique")
        self._owner = owner
        self._layer_ordinals = {
            layer_id: ordinal for ordinal, layer_id in enumerate(layer_ids)
        }
        self._w13_shape = w13_shape
        self._w2_shape = w2_shape
        self._dtype = dtype
        self._claimed_layers: set[int] = set()
        self._lock = threading.Lock()

        w13_bytes = torch.Size(w13_shape).numel() * dtype.itemsize
        w2_bytes = torch.Size(w2_shape).numel() * dtype.itemsize
        self._w2_offset_bytes = _align_up(
            w13_bytes,
            _PINNED_STORAGE_ALIGNMENT_BYTES,
        )
        self._layer_stride_bytes = _align_up(
            self._w2_offset_bytes + w2_bytes,
            _PINNED_STORAGE_ALIGNMENT_BYTES,
        )
        storage_bytes = len(layer_ids) * self._layer_stride_bytes
        logger.info(
            "Allocating %.2f GiB of pinned CPU storage for %d routed-expert layers",
            storage_bytes / 1024**3,
            len(layer_ids),
        )
        self.storage = _allocate_pinned_storage(storage_bytes)

    @property
    def nbytes(self) -> int:
        return self.storage.nbytes()

    def belongs_to(self, owner: object) -> bool:
        return self._owner is owner

    def validate_complete(self) -> None:
        missing = self._layer_ordinals.keys() - self._claimed_layers
        if missing:
            raise RuntimeError(
                "streamed expert host storage is missing configured layers "
                f"{sorted(missing)}"
            )

    def validate_layer(
        self,
        layer_id: int,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
    ) -> None:
        try:
            ordinal = self._layer_ordinals[layer_id]
        except KeyError as exc:
            raise RuntimeError(
                f"layer {layer_id} is not backed by the streamed expert host store"
            ) from exc

        layer_offset = ordinal * self._layer_stride_bytes
        expected = (
            ("w13_weight", w13_weight, self._w13_shape, layer_offset),
            (
                "w2_weight",
                w2_weight,
                self._w2_shape,
                layer_offset + self._w2_offset_bytes,
            ),
        )
        storage_ptr = self.storage.data_ptr()
        for name, tensor, shape, byte_offset in expected:
            if (
                tensor.shape != shape
                or tensor.dtype != self._dtype
                or tensor.device.type != "cpu"
                or not tensor.is_pinned()
                or not tensor.is_contiguous()
                or tensor.untyped_storage().data_ptr() != storage_ptr
                or tensor.data_ptr() != storage_ptr + byte_offset
            ):
                raise RuntimeError(
                    f"streamed expert {name} for layer {layer_id} no longer "
                    "matches its pinned host-store slice"
                )

    def claim_layer(
        self,
        layer_id: int,
        w13_shape: tuple[int, ...],
        w2_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            w13_shape != self._w13_shape
            or w2_shape != self._w2_shape
            or dtype != self._dtype
        ):
            raise ValueError("all streamed expert layers must share one weight layout")
        try:
            ordinal = self._layer_ordinals[layer_id]
        except KeyError as exc:
            raise ValueError(f"layer {layer_id} is not a configured MoE layer") from exc
        with self._lock:
            if layer_id in self._claimed_layers:
                raise RuntimeError(
                    "streamed expert host storage for layer "
                    f"{layer_id} was claimed twice"
                )
            self._claimed_layers.add(layer_id)

        layer_offset = ordinal * self._layer_stride_bytes
        return (
            self._view(layer_offset, w13_shape),
            self._view(layer_offset + self._w2_offset_bytes, w2_shape),
        )

    def _view(
        self,
        byte_offset: int,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        storage_offset, remainder = divmod(byte_offset, self._dtype.itemsize)
        if remainder:
            raise RuntimeError("streamed expert host offset is not dtype-aligned")
        return torch.empty(0, dtype=self._dtype, device="cpu").set_(
            self.storage,
            storage_offset,
            shape,
        )


_STREAMED_HOST_STORES: weakref.WeakValueDictionary[int, _StreamedExpertHostStore] = (
    weakref.WeakValueDictionary()
)
_STREAMED_HOST_STORES_LOCK = threading.Lock()


def _configured_moe_layer_ids(vllm_config: object) -> tuple[int, ...]:
    config = vllm_config.model_config.hf_text_config  # type: ignore[attr-defined]
    mlp_only_layers = set(getattr(config, "mlp_only_layers", None) or ())
    sparse_step = getattr(config, "decoder_sparse_step", 1)
    return tuple(
        layer_id
        for layer_id in range(config.num_hidden_layers)
        if layer_id not in mlp_only_layers
        and config.num_experts > 0
        and (layer_id + 1) % sparse_step == 0
    )


def _streamed_expert_weight_views(
    layer: "RoutedExperts",
    w13_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    vllm_config = get_current_vllm_config()
    load_token = get_streamed_expert_cache_load_token()
    runtime_key = id(load_token)
    with _STREAMED_HOST_STORES_LOCK:
        store = _STREAMED_HOST_STORES.get(runtime_key)
        if store is None:
            store = _StreamedExpertHostStore(
                load_token,
                _configured_moe_layer_ids(vllm_config),
                w13_shape,
                w2_shape,
                dtype,
            )
            _STREAMED_HOST_STORES[runtime_key] = store
        elif not store.belongs_to(load_token):
            raise RuntimeError("stale streamed expert host-store configuration")

        weights = store.claim_layer(
            extract_layer_index(layer.layer_name),
            w13_shape,
            w2_shape,
            dtype,
        )
        layer._streamed_expert_host_store = store
        return weights


# --8<-- [start:unquantized_fused_moe]
@CustomOp.register("unquantized_fused_moe")
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    # --8<-- [end:unquantized_fused_moe]

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.unquantized_backend, self.experts_cls = select_unquantized_moe_backend(
            moe_config=self.moe,
        )

    @property
    def supports_eplb(self) -> bool:
        return True

    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        use_streamed_cache = _streamed_expert_cache_enabled()
        if self.moe.is_act_and_mul:
            w13_up_dim = 2 * intermediate_size_per_partition
        else:
            w13_up_dim = intermediate_size_per_partition
        w13_shape = (num_experts, w13_up_dim, hidden_size)
        w2_shape = (
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
        )
        if use_streamed_cache:
            w13_data, w2_data = _streamed_expert_weight_views(
                layer,
                w13_shape,
                w2_shape,
                params_dtype,
            )
        else:
            w13_data = torch.empty(*w13_shape, dtype=params_dtype)
            w2_data = torch.empty(*w2_shape, dtype=params_dtype)
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            w13_data,
            requires_grad=False,
        )
        if use_streamed_cache:
            w13_weight._vllm_streamed_expert_host = True
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(num_experts, w13_up_dim, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            w2_data,
            requires_grad=False,
        )
        if use_streamed_cache:
            w2_weight._vllm_streamed_expert_host = True
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        if self.moe.has_bias:
            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def _maybe_pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        # Pad the weight tensor. This is an optimization on ROCm platform, which
        # can benefit from tensors located far enough from one another in memory.
        # Skip padding when EPLB is enabled because EPLB requires contiguous
        # weights for the view/rearrangement operations.
        if (
            envs.VLLM_ROCM_MOE_PADDING
            and current_platform.is_rocm()
            and not self.moe.moe_parallel_config.enable_eplb
            and weight.stride(-1) == 1
            and (weight.stride(-2) * weight.element_size()) % 512 == 0
        ):
            num_pad = 256 // weight.element_size()
            weight = F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]
            torch.accelerator.empty_cache()

        return weight

    def _setup_kernel(
        self,
        layer: "RoutedExperts",
        w13: torch.Tensor,
        w2: torch.Tensor,
    ) -> None:
        # Shuffle weights to runtime format.
        w13_new, w2_new = convert_to_unquantized_kernel_format(
            self.unquantized_backend,
            moe_config=layer.moe_config,
            w13_weight=w13,
            w2_weight=w2,
        )
        # `moe_kernel` is initialized to None in FusedMoEMethodBase.__init__;
        # On the first call we replace the parameter normally. On subsequent
        # calls (e.g. RL weight updates that re-trigger
        # process_weights_after_loading) the moe kernel has already been set
        # up and CUDA graphs may have captured the parameter addresses, so
        # we copy the shuffled data into the existing storage instead of
        # re-registering a new Parameter.
        is_weight_update = self.moe_kernel is not None  # type: ignore[has-type]
        replace_parameter(layer, "w13_weight", w13_new, prefer_copy=is_weight_update)
        replace_parameter(layer, "w2_weight", w2_new, prefer_copy=is_weight_update)

        if not is_weight_update:
            # Setup moe kernel only on the first call. For the unquantized
            # method, moe_quant_config carries no quantized scales -- only
            # optional w{13,2}_bias references and SwiGLU gate params. Since
            # weight updates mutate those bias tensors in place, the kernel
            # does not need to be re-built.
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            assert self.moe_quant_config is not None
            assert self.experts_cls is not None
            self.moe_kernel = make_unquantized_moe_kernel(
                quant_config=self.moe_quant_config,
                moe_config=self.moe,
                backend=self.unquantized_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
            )

            if self.unquantized_backend == UnquantizedMoeBackend.CPU:
                # The CPU experts need the layer itself for the setup that
                # convert_to_unquantized_kernel_format cannot express, since
                # it only sees the two weight tensors: padding and prepacking
                # into the grouped-gemm layout (bias included), and capturing
                # the router config that monolithic apply() cannot carry.
                self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def process_weights_after_loading(self, layer: "RoutedExperts") -> None:
        super().process_weights_after_loading(layer)

        if _streamed_expert_cache_enabled():
            if self.moe_kernel is not None:
                raise RuntimeError(
                    "streamed expert caching does not support hot weight updates"
                )
            if self.unquantized_backend != UnquantizedMoeBackend.TRITON:
                raise ValueError(
                    "streamed expert caching requires the resolved Triton MoE "
                    f"backend, but selected {self.unquantized_backend.value!r}"
                )
            host_store = getattr(layer, "_streamed_expert_host_store", None)
            if not isinstance(host_store, _StreamedExpertHostStore):
                raise RuntimeError(
                    "streamed expert weights are missing their pinned host store"
                )
            host_store.validate_complete()
            host_store.validate_layer(
                extract_layer_index(layer.layer_name),
                layer.w13_weight,
                layer.w2_weight,
            )
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            assert self.experts_cls is not None
            self.moe_kernel = make_unquantized_moe_kernel(
                quant_config=self.moe_quant_config,
                moe_config=self.moe,
                backend=self.unquantized_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
            )
            from vllm.model_executor.layers.fused_moe.cached_expert_layer import (
                bind_streamed_expert_layer,
            )

            layer.cached_expert_layer = bind_streamed_expert_layer(
                layer, self.moe_kernel
            )
            return

        # Padding the weight for better performance on ROCm.
        # _maybe_pad_weight is idempotent: on the first call it allocates a
        # padded storage and returns a strided view; on subsequent calls
        # (weight updates) the stride condition no longer matches so it
        # returns the input unchanged. The reassignment to .data is therefore
        # a no-op on updates and preserves the storage address (data_ptr)
        # used by captured CUDA graphs.
        layer.w13_weight.data = self._maybe_pad_weight(layer.w13_weight.data)
        layer.w2_weight.data = self._maybe_pad_weight(layer.w2_weight.data)

        if self.unquantized_backend in [
            UnquantizedMoeBackend.TPU,
            UnquantizedMoeBackend.OOT,
        ]:
            # OOT handles internally.
            return

        elif self.unquantized_backend == UnquantizedMoeBackend.XPU:
            w13 = layer.w13_weight
            w2 = layer.w2_weight

            w13.data = w13.transpose(-1, -2).contiguous()
            w2.data = w2.transpose(-1, -2).contiguous()

            self._setup_kernel(
                layer=layer,
                w13=w13,
                w2=w2,
            )
        else:
            self._setup_kernel(
                layer=layer,
                w13=layer.w13_weight,
                w2=layer.w2_weight,
            )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        # SwiGLU/swigluoai gate params live on the layer; plumb them into the
        # quant config so the fused activation (e.g. swigluoai_uninterleave on
        # MiniMax-M3) receives gemm1_clamp_limit/alpha/beta.
        gemm1_alpha = getattr(layer, "swiglu_alpha", None)
        gemm1_beta = getattr(layer, "swiglu_beta", None)
        gemm1_clamp_limit = getattr(layer, "swiglu_limit", None)

        if self.moe.has_bias:
            return biased_moe_quant_config(
                layer.w13_bias,
                layer.w2_bias,
                gemm1_alpha=gemm1_alpha,
                gemm1_beta=gemm1_beta,
                gemm1_clamp_limit=gemm1_clamp_limit,
            )

        return FusedMoEQuantConfig.make(
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
            gemm1_clamp_limit=gemm1_clamp_limit,
        )

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        cached_layer = getattr(layer, "cached_expert_layer", None)
        if cached_layer is not None:
            assert self.moe_kernel is not None
            prepared = cached_layer.prepare(
                self.moe_kernel,
                x,
                topk_weights,
                topk_ids,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                shared_experts=shared_experts,
                shared_experts_input=shared_experts_input,
            )
            return cached_layer.execute(prepared)
        return self.forward(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def forward_native(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def forward_cuda(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.forward_native(
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts,
            shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )
