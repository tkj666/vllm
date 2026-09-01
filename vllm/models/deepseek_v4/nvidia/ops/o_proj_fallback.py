# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
)
from vllm.model_executor.utils import replace_parameter
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _fused_inverse_rope_gptj,
)


def _dequantize_block_fp8_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, int],
) -> torch.Tensor:
    """Dequantize a serialized block-FP8 weight to BF16."""
    assert weight.ndim == 2 and scale.ndim == 2
    assert weight.dtype == torch.float8_e4m3fn
    block_m, block_k = block_size
    expected_scale_shape = (
        (weight.shape[0] + block_m - 1) // block_m,
        (weight.shape[1] + block_k - 1) // block_k,
    )
    assert scale.shape == expected_scale_shape, (
        f"Expected scale shape {expected_scale_shape} for weight shape "
        f"{tuple(weight.shape)}, got {tuple(scale.shape)}"
    )
    if scale.dtype in (torch.float8_e8m0fnu, torch.uint8):
        scale = _upcast_e8m0_to_fp32(scale)
    else:
        scale = scale.float()
    expanded_scale = torch.repeat_interleave(scale, block_m, dim=0)
    expanded_scale = torch.repeat_interleave(expanded_scale, block_k, dim=1)
    expanded_scale = expanded_scale[: weight.shape[0], : weight.shape[1]]
    return (weight.float() * expanded_scale).to(torch.bfloat16)


class DeepseekV4WoABf16LinearMethod(Fp8LinearMethod):
    """Load serialized FP8 ``wo_a`` and retain a grouped BF16 weight.

    The generic pre-Hopper FP8 linear path repacks weights for Marlin. DeepSeek
    V4 consumes ``wo_a`` as a grouped projection, so that private 2D layout is
    not usable by the correctness fallback.
    """

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if layer.weight.ndim == 3 and layer.weight.dtype == torch.bfloat16:
            return
        assert self.block_quant and self.weight_block_size is not None
        assert hasattr(layer, "weight_scale_inv")
        n_groups = getattr(layer, "bmm_batch_size", 0)
        assert n_groups > 0 and layer.weight.shape[0] % n_groups == 0

        weight = _dequantize_block_fp8_weight(
            layer.weight,
            layer.weight_scale_inv,
            (self.weight_block_size[0], self.weight_block_size[1]),
        )
        weight = weight.view(n_groups, weight.shape[0] // n_groups, weight.shape[1])
        replace_parameter(layer, "weight", weight)
        replace_parameter(layer, "weight_scale_inv", None)
        layer.input_scale = None

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        assert weight.ndim == 3 and weight.dtype == torch.bfloat16
        n_groups, output_size, input_size = weight.shape
        grouped_x = x.reshape(-1, n_groups, input_size)
        output = torch.bmm(grouped_x.transpose(0, 1), weight.transpose(1, 2)).transpose(
            0, 1
        )
        output = output.reshape(*x.shape[:-1], n_groups * output_size)
        if bias is not None:
            output = output + bias
        return output


def inverse_rope_bf16(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """Apply inverse GPT-J RoPE and write BF16 output."""
    return _fused_inverse_rope_gptj(o, positions, cos_sin_cache, rope_dim)


def sm89_bf16_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    rope_dim: int,
) -> torch.Tensor:
    """Correctness-first DeepSeek V4 output projection for pre-Hopper CUDA."""
    assert o.shape[1] == n_groups * heads_per_group
    rotated = inverse_rope_bf16(o, positions, cos_sin_cache, rope_dim)
    grouped = rotated.view(o.shape[0], n_groups, -1)
    weight = wo_a.weight
    assert weight.ndim == 3 and weight.shape[0] == n_groups
    projected = torch.bmm(grouped.transpose(0, 1), weight.transpose(1, 2)).transpose(
        0, 1
    )
    return wo_b(projected.flatten(1))
