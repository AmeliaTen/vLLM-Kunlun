"""
Patch save_partial_states.py and fused_compress_quant_cache.py with
pure PyTorch fallback implementations for Kunlun XPU (no Triton).

save_partial_states: stores KV and (score+APE) into state_cache
compress_norm_rope_store_triton: compress -> RMSNorm -> RoPE -> FP8 quant -> KV cache write

For minimum viable token generation, we skip the FP8 UE8M0 quantization
and just store bf16 directly into the KV cache (since --kv-cache-dtype bfloat16).
"""

SITE = '/opt/vllm_kunlun/lib/python3.10/site-packages'

# ==========================================================================
# Patch 1: save_partial_states.py - replace Triton kernel with PyTorch
# ==========================================================================
path = f'{SITE}/vllm/models/deepseek_v4/common/ops/save_partial_states.py'

new_content = '''# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch fallback for save_partial_states (Kunlun - no Triton)."""

import torch


def save_partial_states(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    state_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    state_width: int,
    compress_ratio: int,
    pdl_kwargs: dict | None = None,
) -> None:
    """Write packed [kv, score+ape] partial states into the compressor cache.

    Pure PyTorch fallback replacing the Triton kernel.
    One operation per token; pads (slot_id == -1) are skipped.
    """
    num_actual = slot_mapping.shape[0]
    head_size = kv.shape[-1]

    # Filter out invalid tokens (slot_id == -1)
    valid_mask = slot_mapping >= 0
    if not valid_mask.any():
        return

    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
    valid_slots = slot_mapping[valid_indices]

    block_idx = valid_slots // block_size
    pos_in_block = valid_slots % block_size

    # Store KV into first half of state_cache
    # state_cache shape: [num_blocks, block_size, state_width*2]
    # kv_state at [:state_width], score_state at [state_width:]
    state_cache[block_idx, pos_in_block, :head_size] = kv[valid_indices]

    # Fused: score += ape[position % compress_ratio]
    valid_positions = positions[valid_indices]
    ape_rows = valid_positions % compress_ratio
    score_with_ape = score[valid_indices] + ape[ape_rows]

    state_cache[block_idx, pos_in_block, state_width:state_width + head_size] = score_with_ape
'''

with open(path, 'w') as f:
    f.write(new_content)
print("[1/2] save_partial_states.py: REPLACED with PyTorch fallback")

# ==========================================================================
# Patch 2: fused_compress_quant_cache.py - replace compress_norm_rope_store_triton
# ==========================================================================
path = f'{SITE}/vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py'
with open(path, 'r') as f:
    content = f.read()

# We replace the compress_norm_rope_store_triton function entirely.
# The triton kernels below it can stay (they won't be called).
# We just need to replace the Python launcher function.

old_fn_start = "def compress_norm_rope_store_triton("
old_fn_end = "    )\n"  # end of the kernel launch call

# Find the function boundaries
fn_start_idx = content.find(old_fn_start)
if fn_start_idx < 0:
    print("[2/2] ERROR: Could not find compress_norm_rope_store_triton function")
    exit(1)

# Find the next @triton.jit to know where the function ends
next_triton = content.find("@triton.jit", fn_start_idx)
if next_triton < 0:
    print("[2/2] ERROR: Could not find end of compress_norm_rope_store_triton")
    exit(1)

# Replace everything from function start to the @triton.jit
old_fn_text = content[fn_start_idx:next_triton]

new_fn_text = '''def compress_norm_rope_store_triton(
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata,
    pdl_kwargs: dict,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    use_fp4_cache: bool,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
) -> None:
    """Pure PyTorch fallback for the fused compress+norm+RoPE+store kernel.

    For each token at a compress boundary (position+1 % compress_ratio == 0):
    1. Gather compress_ratio (or 2*compress_ratio if overlap) state entries
    2. Softmax attention over scores to get weights
    3. Weighted sum of KV states -> compressed_kv
    4. RMSNorm
    5. GPT-J RoPE on rope portion
    6. Store directly as bfloat16 into kv_cache (skip FP8 quant for Kunlun)
    """
    import torch

    coff = 2 if overlap else 1
    window = coff * compress_ratio
    nope_head_dim = head_dim - rope_head_dim
    half_rope = rope_head_dim // 2

    # Find boundary tokens: (position + 1) % compress_ratio == 0
    # AND slot_id >= 0
    all_positions = positions[:num_actual]
    all_slots = slot_mapping[:num_actual]
    valid_mask = (all_slots >= 0) & ((all_positions + 1) % compress_ratio == 0)

    if not valid_mask.any():
        return

    valid_indices = valid_mask.nonzero(as_tuple=True)[0]

    # Get kv_cache slot mapping from k_cache_metadata
    kv_slot_mapping = k_cache_metadata.slot_mapping
    kv_cache_block_size = kv_cache.shape[1] if kv_cache.ndim >= 3 else 1

    for idx in valid_indices:
        token_idx = idx.item()
        position = all_positions[token_idx].item()
        req_idx = token_to_req_indices[token_idx].item()

        # 1. Gather state cache entries for this compress window
        start = position - window + 1
        gather_positions = torch.arange(start, position + 1, device=state_cache.device)
        gather_mask = gather_positions >= 0

        # Map positions to state_cache locations via block_table
        block_indices = gather_positions // block_size
        # Clamp for safety
        block_indices = block_indices.clamp(min=0)
        block_numbers = block_table[req_idx, block_indices]
        block_offsets = gather_positions % block_size

        # Handle head_offset for overlap (second half uses offset HEAD_SIZE in state)
        tokens_in_window = torch.arange(window, device=state_cache.device)
        head_offset = (tokens_in_window >= compress_ratio).long() * head_dim if overlap else torch.zeros(window, dtype=torch.long, device=state_cache.device)

        # Gather KV states: state_cache[block_num, block_offset, head_offset:head_offset+head_dim]
        # state_cache shape: [num_blocks, block_size, total_state_width]
        kv_states = torch.zeros(window, head_dim, dtype=torch.float32, device=state_cache.device)
        score_states = torch.zeros(window, head_dim, dtype=torch.float32, device=state_cache.device)

        for i in range(window):
            if not gather_mask[i]:
                score_states[i] = float('-inf')
                continue
            bn = block_numbers[i].item()
            bo = block_offsets[i].item()
            ho = head_offset[i].item()
            kv_states[i] = state_cache[bn, bo, ho:ho + head_dim]
            score_states[i] = state_cache[bn, bo, state_width + ho:state_width + ho + head_dim]

        # 2. Softmax over scores -> weights
        # score_states: [window, head_dim], softmax over dim=0 (across window)
        weights = torch.softmax(score_states, dim=0)  # [window, head_dim]

        # 3. Weighted sum -> compressed_kv
        compressed_kv = (kv_states * weights).sum(dim=0)  # [head_dim] fp32

        # 4. RMSNorm
        variance = (compressed_kv * compressed_kv).mean()
        rrms = torch.rsqrt(variance + rms_norm_eps)
        normed = compressed_kv * rrms * rms_norm_weight.float()  # [head_dim] fp32

        # 5. GPT-J RoPE on rope portion (last rope_head_dim elements)
        compressed_pos = (position // compress_ratio) * compress_ratio
        cs = cos_sin_cache[compressed_pos]  # [rope_head_dim]
        cos_vals = cs[:half_rope]
        sin_vals = cs[half_rope:]

        # Apply GPT-J style RoPE (interleaved pairs) to rope portion
        rope_part = normed[nope_head_dim:]  # [rope_head_dim]
        # Reshape to pairs: [half_rope, 2] -> even/odd
        rope_pairs = rope_part.view(half_rope, 2)
        even = rope_pairs[:, 0]
        odd = rope_pairs[:, 1]
        new_even = even * cos_vals - odd * sin_vals
        new_odd = odd * cos_vals + even * sin_vals
        # Interleave back
        rope_result = torch.stack([new_even, new_odd], dim=-1).view(rope_head_dim)
        normed[nope_head_dim:] = rope_result

        # 6. Store into kv_cache as bf16 (skip FP8 quant for Kunlun fallback)
        kv_slot_idx = kv_slot_mapping[token_idx].item()
        if kv_slot_idx < 0:
            continue

        kv_block_idx = kv_slot_idx // kv_cache_block_size
        kv_pos_in_block = kv_slot_idx % kv_cache_block_size

        # kv_cache shape: [num_blocks, block_size, head_dim] for bfloat16
        # Store the full normed+roped vector as bf16
        kv_cache[kv_block_idx, kv_pos_in_block, :head_dim] = normed.to(kv_cache.dtype)


'''

content = content[:fn_start_idx] + new_fn_text + content[next_triton:]

with open(path, 'w') as f:
    f.write(content)
print("[2/2] fused_compress_quant_cache.py: REPLACED compress_norm_rope_store_triton with PyTorch fallback")

print("\n=== COMPRESSOR PATCHES APPLIED ===")
print("Both save_partial_states and compress_norm_rope_store now use pure PyTorch.")
print("Note: KV cache stores bf16 directly (no FP8 UE8M0 quant) for simplicity.")
