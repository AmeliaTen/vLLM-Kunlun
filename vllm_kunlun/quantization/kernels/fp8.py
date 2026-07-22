import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFP8ScaledMMLinearKernel,
    CutlassFp8BlockScaledMMKernel,
)


_FP8_BLOCK_SIZE = 128


class _KunlunFp8BlockDequantFallback:
    def apply_weights(self, layer, x, bias=None, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = x.data
        params = self._get_layer_params(layer)
        weight = params.weight
        weight_scale = params.weight_scale_inv
        if weight_scale is None:
            weight_scale = params.weight_scale
        assert weight_scale is not None

        # The fallback keeps checkpoint layout [N, K] instead of applying the
        # Cutlass [K, N] transpose. Model-sized FP8 casts are unavailable on
        # Kunlun, so dequantize one layer at a time on CPU.
        n, k = weight.shape
        assert n % _FP8_BLOCK_SIZE == 0
        assert k % _FP8_BLOCK_SIZE == 0
        n_blocks = n // _FP8_BLOCK_SIZE
        k_blocks = k // _FP8_BLOCK_SIZE
        assert weight_scale.shape[0] >= n_blocks
        assert weight_scale.shape[1] >= k_blocks
        weight_scale = weight_scale[:n_blocks, :k_blocks]
        if weight.is_contiguous():
            weight_bf16 = weight.cpu().to(torch.bfloat16)
        else:
            assert weight.ndim == 2 and weight.t().is_contiguous()
            weight_bf16 = (
                weight.t().cpu().to(torch.bfloat16).t().contiguous()
            )
        weight_bf16 = weight_bf16.view(
            n_blocks, _FP8_BLOCK_SIZE, k_blocks, _FP8_BLOCK_SIZE
        )
        weight_bf16 = weight_bf16 * weight_scale.cpu().to(torch.bfloat16).view(
            n_blocks, 1, k_blocks, 1
        )
        weight_bf16 = weight_bf16.reshape(n, k).to(x.device)
        if bias is not None:
            bias = bias.to(torch.bfloat16)
        return F.linear(x.to(torch.bfloat16), weight_bf16, bias)


class KunlunFP8ScaledMMLinearKernel(CutlassFP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(cls, compute_capability=None):
        return True, None


class KunlunFp8BlockScaledMMKernel(
    _KunlunFp8BlockDequantFallback, CutlassFp8BlockScaledMMKernel
):
    @classmethod
    def is_supported(cls, compute_capability=None):
        return True, None

    def process_weights_after_loading(self, layer):
        Fp8BlockScaledMMLinearKernel.process_weights_after_loading(self, layer)
