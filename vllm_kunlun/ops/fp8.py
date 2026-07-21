import torch


_FP8_BLOCK_SIZE = 128


def fused_indexer_q_rope_quant_kunlun(
    positions,
    index_q,
    index_q_cos_sin_cache,
    index_weights,
    index_weights_softmax_scale,
    index_weights_head_scale,
    use_fp4=False,
):
    """Correctness fallback for V4 Indexer Q RoPE and FP8 quantization."""
    assert not use_fp4, "Kunlun correctness fallback only supports FP8 Indexer Q"
    rope_dim = index_q_cos_sin_cache.shape[-1]
    half_rope_dim = rope_dim // 2
    nope_dim = index_q.shape[-1] - rope_dim
    assert rope_dim % 2 == 0 and nope_dim >= 0

    cos_sin = index_q_cos_sin_cache.index_select(0, positions.long()).float()
    cos = cos_sin[..., :half_rope_dim].unsqueeze(1)
    sin = cos_sin[..., half_rope_dim:].unsqueeze(1)

    q_float = index_q.float()
    q_rot = q_float[..., nope_dim:]
    q_even = q_rot[..., 0::2]
    q_odd = q_rot[..., 1::2]
    rotated = torch.empty_like(q_rot)
    rotated[..., 0::2] = q_even * cos - q_odd * sin
    rotated[..., 1::2] = q_odd * cos + q_even * sin
    rotated = rotated.to(torch.bfloat16).float()
    q_rope = torch.cat((q_float[..., :nope_dim], rotated), dim=-1)

    fp8_max = 448.0
    q_scale = q_rope.abs().amax(dim=-1).clamp_min(1e-4) / fp8_max
    q_scale = torch.pow(2.0, torch.ceil(torch.log2(q_scale)))
    q_normalized = q_rope / q_scale.unsqueeze(-1)
    q_fp8 = q_normalized.cpu().to(torch.float8_e4m3fn).to(index_q.device)

    weights_out = index_weights.float() * q_scale
    weights_out *= index_weights_softmax_scale * index_weights_head_scale
    return q_fp8, weights_out


def dequantize_fp8_blocks(weight, weight_scale):
    """Dequantize a block-scaled FP8 matrix to BF16.

    fp8->bf16 cast must go through CPU (Kunlun XPU copy_kernel doesn't support it),
    but scale multiply is done on-device to minimize PCIe traffic.
    """
    n, k = weight.shape
    assert n % _FP8_BLOCK_SIZE == 0
    assert k % _FP8_BLOCK_SIZE == 0
    n_blocks = n // _FP8_BLOCK_SIZE
    k_blocks = k // _FP8_BLOCK_SIZE
    assert weight_scale.shape[0] >= n_blocks
    assert weight_scale.shape[1] >= k_blocks
    weight_scale = weight_scale[:n_blocks, :k_blocks]
    # fp8->bf16 cast on CPU (XPU doesn't support this cast), then back to device
    device = weight.device
    if weight.is_contiguous():
        weight_bf16 = weight.cpu().to(torch.bfloat16).to(device)
    else:
        assert weight.ndim == 2 and weight.t().is_contiguous()
        weight_bf16 = weight.t().cpu().to(torch.bfloat16).t().contiguous().to(device)
    weight_bf16 = weight_bf16.view(
        n_blocks, _FP8_BLOCK_SIZE, k_blocks, _FP8_BLOCK_SIZE
    )
    # Scale multiply on-device (avoids sending the large result over PCIe again)
    weight_bf16 = weight_bf16 * weight_scale.to(torch.bfloat16).view(
        n_blocks, 1, k_blocks, 1
    )
    return weight_bf16.reshape(n, k)
