"""
Kunlun optimized FusedMoE - replaces UnquantizedFusedMoEMethod
Uses monolithic mode to receive router_logits directly and call KunlunOps.fused_moe
"""

import logging

import torch
import torch.nn.functional as F
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

from vllm_kunlun.ops.fp8 import dequantize_fp8_blocks


@CustomOp.register_oot(name="UnquantizedFusedMoEMethod")
class KunlunUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """
    Kunlun optimized UnquantizedFusedMoEMethod.

    Key design:
    - is_monolithic = True: FusedMoE calls apply_monolithic(layer, x, router_logits)
      instead of routing first and then calling apply(layer, x, topk_weights, topk_ids).
    - This passes router_logits directly to KunlunOps.fused_moe, which handles
      routing internally with device-optimized kernels.
    """

    @property
    def is_monolithic(self) -> bool:
        return True

    def _select_monolithic(self):
        """Override parent: parent's __init__ assigns
        ``self.apply_monolithic = self._select_monolithic()`` which would
        otherwise shadow the class-level ``apply_monolithic`` defined below
        with ``forward_monolithic_cuda``. Return the class method instead."""
        return KunlunUnquantizedFusedMoEMethod.apply_monolithic.__get__(self)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Skip _setup_kernel() since Kunlun does not need Triton kernels."""
        FusedMoEMethodBase.process_weights_after_loading(self, layer)

    def apply_monolithic(
        self,
        layer,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Monolithic mode entry point.
        When is_monolithic=True, FusedMoE.forward_impl calls this method
        directly with (layer, hidden_states, router_logits), bypassing
        the default routing logic.
        """
        from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops

        if self.moe.use_ep:
            return ops.fused_moe_ep(
                x,
                layer.w13_weight,
                layer.w2_weight,
                router_logits,
                self.moe.ep_rank,
                self.moe.experts_per_token,
                renormalize=layer.renormalize,
                inplace=True,
                use_grouped_topk=layer.use_grouped_topk,
                num_expert_group=layer.num_expert_group,
                topk_group=layer.topk_group,
            )
        else:
            return ops.fused_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                router_logits,
                self.moe.ep_rank,
                self.moe.experts_per_token,
                renormalize=layer.renormalize,
                inplace=True,
                use_grouped_topk=layer.use_grouped_topk,
                num_expert_group=layer.num_expert_group,
                topk_group=layer.topk_group,
                scoring_func=layer.scoring_func,
                e_score_correction_bias=layer.e_score_correction_bias,
                w1_bias=getattr(layer, "w13_bias", None),
                w2_bias=getattr(layer, "w2_bias", None),
            )


_logger = logging.getLogger("vllm_kunlun.ops.fused_moe")
_logged_routing_metadata = False


class KunlunFp8MoEMethod(Fp8MoEMethod):
    """Correctness-only FP8 MoE fallback for Kunlun.

    Selected expert weights are dequantized on CPU and executed with BF16
    linear layers. This path is intentionally not a production kernel.
    """

    def __init__(self, quant_config, layer):
        FusedMoEMethodBase.__init__(self, layer.moe_config)
        self.quant_config = quant_config
        self.weight_block_size = quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        if not self.block_quant:
            raise NotImplementedError(
                "Kunlun FP8 MoE correctness fallback requires block-quantized weights"
            )
        self.weight_scale_name = "weight_scale_inv"
        self.fp8_backend = None
        self.experts_cls = None

    @property
    def is_monolithic(self) -> bool:
        return False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        FusedMoEMethodBase.process_weights_after_loading(self, layer)
        # Keep weights as fp8 on device (saves memory).
        # Dequant happens lazily per-expert in _expert_weights via CPU path.

    def maybe_make_prepare_finalize(self, routing_tables=None):
        return None

    def _expert_weights(self, layer, expert_id):
        w13_scale = getattr(layer, f"w13_{self.weight_scale_name}")[expert_id]
        w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")[expert_id]
        return (
            dequantize_fp8_blocks(layer.w13_weight[expert_id], w13_scale).to(
                layer.w13_weight.device
            ),
            dequantize_fp8_blocks(layer.w2_weight[expert_id], w2_scale).to(
                layer.w2_weight.device
            ),
        )

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts,
        shared_experts_input,
    ):
        # FusedMoERunner executes shared experts separately for this non-modular
        # fallback. Accept both arguments to match the vLLM 0.25.1 contract.
        del shared_experts, shared_experts_input
        x_flat = x.reshape(-1, x.shape[-1])
        # Keep all routing metadata on CPU. Kunlun's XPU reshape/where/index
        # path is not reliable for the large vLLM routing buffers.
        weights_cpu = topk_weights.reshape(-1, topk_weights.shape[-1]).cpu()
        ids_cpu = topk_ids.reshape(-1, topk_ids.shape[-1]).cpu()
        global _logged_routing_metadata
        if not _logged_routing_metadata:
            _logger.warning(
                "FP8 MoE routing metadata: shape=%s dtype=%s min=%s max=%s values=%s",
                tuple(ids_cpu.shape),
                ids_cpu.dtype,
                ids_cpu.min().item(),
                ids_cpu.max().item(),
                ids_cpu.flatten()[:16].tolist(),
            )
            _logged_routing_metadata = True
        output = torch.zeros_like(x_flat)
        for expert_id in torch.unique(ids_cpu).tolist():
            token_rows_cpu, choices_cpu = torch.where(ids_cpu == expert_id)
            token_rows = token_rows_cpu.to(x_flat.device)
            expert_x = x_flat[token_rows].to(torch.bfloat16)
            w13, w2 = self._expert_weights(layer, expert_id)
            gate, up = F.linear(expert_x, w13).chunk(2, dim=-1)
            expert_y = F.linear(F.silu(gate) * up, w2)
            expert_weights = weights_cpu[token_rows_cpu, choices_cpu].to(
                expert_y.dtype
            ).to(expert_y.device)
            expert_y = expert_y * expert_weights.unsqueeze(-1)
            output.index_add_(0, token_rows, expert_y.to(output.dtype))
        return output.view_as(x)
