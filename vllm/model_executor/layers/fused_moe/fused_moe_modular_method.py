# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEKernel,
    FusedMoEPrepareAndFinalizeModular,
)

logger = init_logger(__name__)


# --8<-- [start:modular_fused_moe]
@CustomOp.register("modular_fused_moe")
class FusedMoEModularMethod(FusedMoEMethodBase, CustomOp):
    # --8<-- [end:modular_fused_moe]

    def __init__(
        self, old_quant_method: FusedMoEMethodBase, moe_kernel: FusedMoEKernel
    ):
        super().__init__(old_quant_method.moe)
        self.moe_quant_config = old_quant_method.moe_quant_config
        self.moe_kernel = moe_kernel
        self.disable_expert_map = getattr(
            old_quant_method,
            "disable_expert_map",
            not self.moe_kernel.supports_expert_map(),
        )
        self.old_quant_method = old_quant_method
        logger.debug("Swapping out %s", self.old_quant_method.__class__.__name__)

    @staticmethod
    def make(
        moe_layer: torch.nn.Module,
        old_quant_method: FusedMoEMethodBase,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        shared_experts: torch.nn.Module | None,
        inplace: bool = False,
    ) -> "FusedMoEModularMethod":
        return FusedMoEModularMethod(
            old_quant_method,
            FusedMoEKernel(
                prepare_finalize,
                old_quant_method.select_gemm_impl(prepare_finalize, moe_layer),
                shared_experts,
                moe_parallel_config=moe_layer.moe_parallel_config,
                inplace=inplace,
            ),
        )

    @property
    def supports_eplb(self) -> bool:
        return self.old_quant_method.supports_eplb

    @property
    def method_name(self) -> str:
        return self.old_quant_method.method_name

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        raise NotImplementedError

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return self.moe_quant_config

    def apply(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.moe_kernel is not None

        provider = getattr(layer, "expert_weight_provider", None)
        if provider is not None:
            final_output = None
            is_first_chunk = True
            for result in provider.prepare(topk_ids):
                # expert_map: result.expert_map (per chunk) or layer fallback.
                expert_map = (
                    result.expert_map
                    if result.expert_map is not None
                    else (
                        None
                        if self.disable_expert_map
                        else layer.expert_map
                    )
                )

                if result.token_indices is not None:
                    # EP-style chunking: slice inputs, mask weights, accumulate.
                    x_chunk = x[result.token_indices]
                    ids_chunk = result.topk_ids
                    # Zero weights for out-of-chunk experts (sentinel == -1).
                    w_chunk = topk_weights[result.token_indices].clone()
                    sentinel_mask = ids_chunk == -1
                    w_chunk[sentinel_mask] = 0.0
                    ids_chunk = ids_chunk.clamp(min=0)
                    # shared_experts_input must only be added once.
                    se_chunk = (
                        shared_experts_input[result.token_indices]
                        if shared_experts_input is not None and is_first_chunk
                        else None
                    )
                else:
                    x_chunk = x
                    ids_chunk = result.topk_ids
                    w_chunk = topk_weights
                    se_chunk = shared_experts_input

                chunk_out = self.moe_kernel.apply(
                    hidden_states=x_chunk,
                    w1=result.w1,
                    w2=result.w2,
                    topk_weights=w_chunk,
                    topk_ids=ids_chunk,
                    activation=layer.activation,
                    global_num_experts=layer.global_num_experts,
                    apply_router_weight_on_input=(layer.apply_router_weight_on_input),
                    expert_map=expert_map,
                    shared_experts_input=se_chunk,
                )

                if result.token_indices is not None:
                    if final_output is None:
                        final_output = torch.zeros_like(x)
                    final_output[result.token_indices] += chunk_out
                else:
                    return chunk_out

                is_first_chunk = False

            return final_output

        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=None if self.disable_expert_map else layer.expert_map,
            shared_experts_input=shared_experts_input,
        )
