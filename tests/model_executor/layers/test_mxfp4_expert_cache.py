# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe import mxfp4_expert_cache
from vllm.model_executor.layers.fused_moe.mxfp4_expert_cache import (
    _allocate_pinned_storage,
    _make_layouts,
    _StreamedMxfp4HostStore,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)


def test_deepseek_v4_mxfp4_host_layout_size() -> None:
    layouts = _make_layouts(
        num_experts=256,
        hidden_size=4096,
        intermediate_size=2048,
        w13_num_shards=2,
    )

    layer_bytes = sum(layout.load_nbytes for layout in layouts.values())

    assert layer_bytes == 3_422_552_064
    assert 43 * layer_bytes == 147_169_738_752
    assert all(
        layout.load_nbytes == layout.runtime_nbytes for layout in layouts.values()
    )


def test_mxfp4_store_uses_direct_single_pinned_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_calls: list[dict] = []
    original_empty = torch.empty

    def record_empty(*args, **kwargs) -> torch.Tensor:
        empty_calls.append(dict(kwargs))
        kwargs.pop("pin_memory", None)
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", record_empty)
    monkeypatch.setattr(torch.UntypedStorage, "is_pinned", lambda storage: True)

    storage = _allocate_pinned_storage(4096)

    assert storage.nbytes() == 4096
    assert empty_calls == [
        {
            "dtype": torch.uint8,
            "device": "cpu",
            "pin_memory": True,
        }
    ]


@pytest.mark.parametrize(
    ("load_format", "strategy", "should_raise"),
    [
        ("auto", "lazy", False),
        ("dummy", None, False),
        ("auto", None, True),
        ("auto", "prefetch", True),
    ],
)
def test_streamed_mxfp4_requires_lazy_safetensors(
    monkeypatch: pytest.MonkeyPatch,
    load_format: str,
    strategy: str | None,
    should_raise: bool,
) -> None:
    config = SimpleNamespace(
        load_config=SimpleNamespace(
            load_format=load_format,
            safetensors_load_strategy=strategy,
        )
    )
    monkeypatch.setattr(
        mxfp4_expert_cache,
        "get_current_vllm_config",
        lambda: config,
    )

    if should_raise:
        with pytest.raises(ValueError, match="safetensors-load-strategy lazy"):
            mxfp4_expert_cache._validate_load_strategy()
    else:
        mxfp4_expert_cache._validate_load_strategy()


def test_streamed_mxfp4_rejects_unconfigured_layer_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        load_config=SimpleNamespace(load_format="dummy"),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type="deepseek_v4",
                num_hidden_layers=2,
            )
        ),
    )
    allocation_calls: list[int] = []

    monkeypatch.setattr(
        mxfp4_expert_cache,
        "get_current_vllm_config",
        lambda: config,
    )
    monkeypatch.setattr(
        mxfp4_expert_cache,
        "_allocate_pinned_storage",
        lambda num_bytes: allocation_calls.append(num_bytes),
    )

    layer = SimpleNamespace(layer_name="model.layers.2.mlp.experts")
    with pytest.raises(ValueError, match="layer 2 is not a configured"):
        mxfp4_expert_cache.streamed_mxfp4_weight_views(
            layer,
            num_experts=2,
            hidden_size=128,
            intermediate_size=64,
            w13_num_shards=2,
            activation_dtype=torch.bfloat16,
        )

    assert allocation_calls == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_streamed_mxfp4_in_place_conversion_matches_marlin() -> None:
    num_experts = 2
    hidden_size = 128
    intermediate_size = 64
    layouts = _make_layouts(
        num_experts,
        hidden_size,
        intermediate_size,
        w13_num_shards=2,
    )
    owner = object()
    store = _StreamedMxfp4HostStore(
        owner,
        (0,),
        layouts,
        hidden_size,
        torch.bfloat16,
    )
    raw = store.claim_layer(0)
    generator = torch.Generator().manual_seed(0)
    for tensor in raw.values():
        tensor.copy_(
            torch.randint(
                0,
                250,
                tensor.shape,
                dtype=tensor.dtype,
                generator=generator,
            )
        )

    layer = SimpleNamespace(params_dtype=torch.bfloat16)
    names = (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
    )
    reference = prepare_moe_mxfp4_layer_for_marlin(
        layer,
        *(raw[name].to("cuda") for name in names),
        None,
        None,
    )[:4]

    converted = store.convert_layer(0, layer, torch.device("cuda"))

    storage_ptr = store.storage.data_ptr()
    for name, expected in zip(names, reference):
        actual = converted.runtime_tensor(name)
        assert actual.untyped_storage().data_ptr() == storage_ptr
        assert actual.is_pinned()
        torch.testing.assert_close(actual, expected.cpu(), rtol=0, atol=0)
    assert all(bundle.is_pinned for bundle in converted.host_bundles.values())
