# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum


@pytest.mark.parametrize(
    ("capability", "supported"),
    [
        (DeviceCapability(8, 6), False),
        (DeviceCapability(8, 9), True),
        (DeviceCapability(9, 0), False),
    ],
)
def test_sm89_backend_capability(capability, supported) -> None:
    from vllm.models.deepseek_v4.nvidia.sm89_sparse import (
        DeepseekV4SM89SparseBackend,
    )

    assert (
        DeepseekV4SM89SparseBackend.supports_compute_capability(capability) is supported
    )


def test_sm89_swa_builder_skips_flashmla_scheduler() -> None:
    from vllm.models.deepseek_v4.nvidia.sm89_sparse import (
        DeepseekV4SM89SparseSWAMetadataBuilder,
    )

    builder = object.__new__(DeepseekV4SM89SparseSWAMetadataBuilder)

    assert builder.build_tile_scheduler(4) == {
        "swaonly": None,
        "c4a": None,
        "c128a": None,
    }


def test_sm89_attention_splits_mixed_batch(monkeypatch) -> None:
    from vllm.models.deepseek_v4.nvidia import sm89_sparse

    prefill = Mock()
    decode = Mock()
    layer = SimpleNamespace(
        prefix="layer",
        compress_ratio=4,
        kv_cache=torch.empty(1),
        swa_cache_layer=SimpleNamespace(prefix="swa", kv_cache=torch.empty(1)),
        _forward_prefill=prefill,
        _forward_decode=decode,
    )
    swa_metadata = SimpleNamespace(
        num_decodes=2,
        num_prefills=1,
        num_decode_tokens=2,
    )
    attn_metadata = object()
    monkeypatch.setattr(
        sm89_sparse,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={"layer": attn_metadata, "swa": swa_metadata}
        ),
    )
    q = torch.empty(5, 64, 512, dtype=torch.bfloat16)
    output = torch.empty_like(q)

    sm89_sparse.DeepseekV4SM89SparseAttention.forward_mqa(
        layer, q, torch.empty(5, 512), torch.arange(5), output
    )

    assert prefill.call_args.kwargs["q"].shape == (3, 64, 512)
    assert prefill.call_args.kwargs["output"].data_ptr() == output[2:].data_ptr()
    assert decode.call_args.kwargs["q"].shape == (2, 64, 512)
    assert decode.call_args.kwargs["output"].data_ptr() == output.data_ptr()


@pytest.mark.parametrize("compress_ratio", [1, 4, 128])
def test_sm89_decode_routes_swa_c4_and_c128(monkeypatch, compress_ratio) -> None:
    from vllm.models.deepseek_v4.nvidia import sm89_sparse

    c4_indices = torch.tensor([[4, 2]], dtype=torch.int32)
    c4_lens = torch.tensor([2], dtype=torch.int32)
    monkeypatch.setattr(
        sm89_sparse,
        "compute_global_topk_indices_and_lens",
        lambda *args: (c4_indices, c4_lens),
    )
    sparse_decode = Mock()
    monkeypatch.setattr(sm89_sparse, "triton_sparse_attn_decode", sparse_decode)

    c128_indices = torch.tensor([[[6, 3]]], dtype=torch.int32)
    c128_lens = torch.tensor([2], dtype=torch.int32)
    attn_metadata = SimpleNamespace(
        block_size=256,
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        c128a_global_decode_topk_indices=c128_indices,
        c128a_decode_topk_lens=c128_lens,
    )
    swa_metadata = SimpleNamespace(
        num_decodes=1,
        num_decode_tokens=1,
        is_valid_token=torch.ones(1, dtype=torch.bool),
        token_to_req_indices=torch.zeros(1, dtype=torch.int32),
        decode_swa_indices=torch.zeros(1, 2, dtype=torch.int32),
        decode_swa_lens=torch.ones(1, dtype=torch.int32),
    )
    layer = SimpleNamespace(
        compress_ratio=compress_ratio,
        topk_indices_buffer=torch.zeros(1, 2, dtype=torch.int32),
        swa_cache_layer=SimpleNamespace(kv_cache=torch.empty(1)),
        attn_sink=torch.empty(64),
        scale=512**-0.5,
        head_dim=512,
        nope_head_dim=448,
        rope_head_dim=64,
    )
    swa_only = compress_ratio == 1

    sm89_sparse.DeepseekV4SM89SparseAttention._forward_decode(
        layer,
        q=torch.empty(1, 64, 512, dtype=torch.bfloat16),
        kv_cache=None if swa_only else torch.empty(1),
        swa_metadata=swa_metadata,
        attn_metadata=None if swa_only else attn_metadata,
        swa_only=swa_only,
        output=torch.empty(1, 64, 512, dtype=torch.bfloat16),
    )

    kwargs = sparse_decode.call_args.kwargs
    if compress_ratio == 1:
        assert kwargs["topk_indices"] is None
        assert kwargs["topk_lens"] is None
    elif compress_ratio == 4:
        assert kwargs["topk_indices"] is c4_indices
        assert kwargs["topk_lens"] is c4_lens
    else:
        assert kwargs["topk_indices"] is c128_indices
        assert kwargs["topk_lens"] is c128_lens
    assert kwargs["output"].shape == (1, 64, 512)


@pytest.mark.parametrize(
    "backend",
    [
        None,
        AttentionBackendEnum.FLASHMLA_SPARSE,
        AttentionBackendEnum.FLASHMLA_SPARSE_DSV4,
    ],
)
def test_dsv4_selects_triton_sparse_fallback_on_sm89(monkeypatch, backend) -> None:
    from vllm.models.deepseek_v4.nvidia import model as model_module
    from vllm.models.deepseek_v4.nvidia.sm89_sparse import (
        DeepseekV4SM89SparseAttention,
    )

    monkeypatch.setattr(
        model_module.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(8, 9),
    )
    config = SimpleNamespace(attention_config=SimpleNamespace(backend=backend))

    assert model_module._select_dsv4_attn_cls(config) is DeepseekV4SM89SparseAttention


def test_dsv4_rejects_flashinfer_sparse_backend_on_sm89(monkeypatch) -> None:
    from vllm.models.deepseek_v4.nvidia import model as model_module

    monkeypatch.setattr(
        model_module.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(8, 9),
    )
    config = SimpleNamespace(
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4
        )
    )

    with pytest.raises(ValueError, match="does not support SM89"):
        model_module._select_dsv4_attn_cls(config)


@pytest.mark.parametrize(
    ("capability", "expected_name"),
    [
        (DeviceCapability(9, 0), "DeepseekV4FlashMLAAttention"),
        (DeviceCapability(12, 0), "DeepseekV4FlashInferSM120Attention"),
    ],
)
def test_dsv4_auto_selection_keeps_newer_cuda_backends(
    monkeypatch, capability, expected_name
) -> None:
    from vllm.models.deepseek_v4.nvidia import model as model_module

    monkeypatch.setattr(
        model_module.current_platform,
        "get_device_capability",
        lambda: capability,
    )
    config = SimpleNamespace(attention_config=SimpleNamespace(backend=None))

    assert model_module._select_dsv4_attn_cls(config).__name__ == expected_name


@pytest.mark.parametrize("use_v2_model_runner", [False, True])
def test_sm89_attention_break_is_wrapped_after_runtime_enable(
    monkeypatch, use_v2_model_runner
) -> None:
    import vllm.envs as envs
    from vllm.compilation import breakable_cudagraph

    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    envs.disable_envs_cache()

    from vllm.models.deepseek_v4 import attention as attention_module

    calls: list[str] = []

    def prepare_and_attn() -> None:
        calls.append("prepare")

    stale_wrapper = attention_module.eager_break_during_capture(prepare_and_attn)
    assert stale_wrapper is prepare_and_attn

    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    envs.disable_envs_cache()
    monkeypatch.setattr(
        attention_module.current_platform,
        "is_device_capability",
        lambda capability: capability == 89,
    )

    selected = attention_module._select_prepare_and_attn_fn(
        prepare_and_attn,
        stale_wrapper,
        use_v2_model_runner=use_v2_model_runner,
    )

    assert selected is not prepare_and_attn
    assert selected.__wrapped__ is prepare_and_attn

    def add_eager(fn):
        calls.append("break")
        return fn()

    capture = SimpleNamespace(
        _capturing=True,
        add_eager=add_eager,
    )
    monkeypatch.setattr(
        breakable_cudagraph.BreakableCUDAGraphCapture,
        "current",
        Mock(return_value=capture),
    )
    monkeypatch.setattr(
        breakable_cudagraph,
        "is_forward_context_available",
        lambda: False,
    )

    selected()

    assert calls == ["break", "prepare"]


def test_sm89_attention_without_breakable_graph_keeps_mrv1_eager_region(
    monkeypatch,
) -> None:
    import vllm.envs as envs
    from vllm.models.deepseek_v4 import attention as attention_module

    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    envs.disable_envs_cache()
    monkeypatch.setattr(
        attention_module.current_platform,
        "is_device_capability",
        lambda capability: capability == 89,
    )

    def prepare_and_attn() -> None:
        pass

    def prepare_and_attn_eager() -> None:
        pass

    selected = attention_module._select_prepare_and_attn_fn(
        prepare_and_attn,
        prepare_and_attn_eager,
        use_v2_model_runner=False,
    )

    assert selected is prepare_and_attn_eager
