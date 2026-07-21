"""Fix RoPE to use GPT-J interleaved style (not NeoX half-split).

DeepSeek V4 uses is_neox_style=False:
  - Pairs: (x[..., 0], x[..., 1]), (x[..., 2], x[..., 3]), ...
  - Even indices: x[..., 0::2]
  - Odd indices:  x[..., 1::2]
  - Forward: even_new = even*cos - odd*sin, odd_new = even*sin + odd*cos
  - Inverse: even_new = even*cos + odd*sin, odd_new = -even*sin + odd*cos

Our bf16 fallback was INCORRECTLY using NeoX style (first-half / second-half split).
"""

SITE = '/opt/vllm_kunlun/lib/python3.10/site-packages'

# ==========================================================================
# Fix 1: attention.py - _fused_qnorm_rope_kv_insert bf16 branch
# ==========================================================================
path = f'{SITE}/vllm/models/deepseek_v4/attention.py'
with open(path, 'r') as f:
    content = f.read()

old_bf16 = """        if cache_dtype == torch.bfloat16:
            # --- PyTorch fallback for bf16 qnorm+rope+kv_insert ---
            N, H, D = q.shape
            rope_dim = cos_sin_cache.shape[1]  # 64
            half_rope = rope_dim // 2  # 32
            nope_dim = D - rope_dim  # 448

            # 1. Per-head RMSNorm on Q (no weight, in-place)
            q_f32 = q.float()
            rms = q_f32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
            q.copy_((q_f32 * rms).to(q.dtype))

            # 2. GPT-J RoPE on Q and KV - applied to LAST rope_dim dims
            cs = cos_sin_cache[positions]  # [N, rope_dim]
            cos_vals = cs[:, :half_rope]   # [N, 32]
            sin_vals = cs[:, half_rope:]   # [N, 32]

            # RoPE on Q: LAST rope_dim dims of each head [nope_dim:]
            q_cos = cos_vals.unsqueeze(1).expand(-1, H, -1)  # [N, H, 32]
            q_sin = sin_vals.unsqueeze(1).expand(-1, H, -1)  # [N, H, 32]
            q0 = q[:, :, nope_dim:nope_dim+half_rope].clone()
            q1 = q[:, :, nope_dim+half_rope:].clone()
            q[:, :, nope_dim:nope_dim+half_rope] = q0 * q_cos - q1 * q_sin
            q[:, :, nope_dim+half_rope:] = q0 * q_sin + q1 * q_cos

            # RoPE on KV: LAST rope_dim dims [nope_dim:]
            kv0 = kv[:, nope_dim:nope_dim+half_rope].clone()
            kv1 = kv[:, nope_dim+half_rope:].clone()
            kv[:, nope_dim:nope_dim+half_rope] = kv0 * cos_vals - kv1 * sin_vals
            kv[:, nope_dim+half_rope:] = kv0 * sin_vals + kv1 * cos_vals

            # 3. KV cache insert
            block_indices = swa_metadata.slot_mapping // block_size
            offsets = swa_metadata.slot_mapping % block_size
            swa_kv_cache_3d[block_indices, offsets] = kv
            return q"""

new_bf16 = """        if cache_dtype == torch.bfloat16:
            # --- PyTorch fallback for bf16 qnorm+rope+kv_insert ---
            N, H, D = q.shape
            rope_dim = cos_sin_cache.shape[1]  # 64
            half_rope = rope_dim // 2  # 32
            nope_dim = D - rope_dim  # 448

            # 1. Per-head RMSNorm on Q (no weight, in-place)
            q_f32 = q.float()
            rms = q_f32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
            q.copy_((q_f32 * rms).to(q.dtype))

            # 2. GPT-J interleaved RoPE (is_neox_style=False)
            #    Pairs: (dim[2i], dim[2i+1]) within the rope portion
            cs = cos_sin_cache[positions]  # [N, rope_dim]
            cos_vals = cs[:, :half_rope]   # [N, 32]
            sin_vals = cs[:, half_rope:]   # [N, 32]

            # RoPE on Q: interleaved pairs in LAST rope_dim dims
            q_cos = cos_vals.unsqueeze(1).expand(-1, H, -1)  # [N, H, 32]
            q_sin = sin_vals.unsqueeze(1).expand(-1, H, -1)  # [N, H, 32]
            q_rope = q[:, :, nope_dim:]  # [N, H, 64]
            q_even = q_rope[:, :, 0::2].clone()  # [N, H, 32] - dims 448,450,...
            q_odd = q_rope[:, :, 1::2].clone()   # [N, H, 32] - dims 449,451,...
            q_rope[:, :, 0::2] = q_even * q_cos - q_odd * q_sin
            q_rope[:, :, 1::2] = q_even * q_sin + q_odd * q_cos

            # RoPE on KV: interleaved pairs in LAST rope_dim dims
            kv_rope = kv[:, nope_dim:]  # [N, 64]
            kv_even = kv_rope[:, 0::2].clone()  # [N, 32]
            kv_odd = kv_rope[:, 1::2].clone()   # [N, 32]
            kv_rope[:, 0::2] = kv_even * cos_vals - kv_odd * sin_vals
            kv_rope[:, 1::2] = kv_even * sin_vals + kv_odd * cos_vals

            # 3. KV cache insert
            block_indices = swa_metadata.slot_mapping // block_size
            offsets = swa_metadata.slot_mapping % block_size
            swa_kv_cache_3d[block_indices, offsets] = kv
            return q"""

if old_bf16 in content:
    content = content.replace(old_bf16, new_bf16)
    with open(path, 'w') as f:
        f.write(content)
    print("[1/3] attention.py: FIXED RoPE to GPT-J interleaved style")
else:
    print("[1/3] attention.py: SKIPPED (text not found)")
    idx = content.find('if cache_dtype == torch.bfloat16:')
    if idx >= 0:
        print(f"  Found bf16 branch at offset {idx}")
        print(repr(content[idx:idx+200]))

# ==========================================================================
# Fix 2: flashmla.py - _o_proj inverse RoPE
# ==========================================================================
path = f'{SITE}/vllm/models/deepseek_v4/nvidia/flashmla.py'
with open(path, 'r') as f:
    content = f.read()

old_oproj_rope = """        # 1. Inverse GPT-J RoPE on the rope portion of o
        cos_sin_cache = self.rotary_emb.cos_sin_cache  # [max_pos, rope_dim]
        cs = cos_sin_cache[positions]  # [T, rope_dim]
        cos_vals = cs[:, :half_rope].unsqueeze(1).expand(-1, H, -1)  # [T, H, 32]
        sin_vals = cs[:, half_rope:].unsqueeze(1).expand(-1, H, -1)  # [T, H, 32]

        # Inverse RoPE: conjugate rotation
        o_rope_0 = o[:, :, nope_dim:nope_dim+half_rope].clone()
        o_rope_1 = o[:, :, nope_dim+half_rope:].clone()
        o[:, :, nope_dim:nope_dim+half_rope] = o_rope_0 * cos_vals + o_rope_1 * sin_vals
        o[:, :, nope_dim+half_rope:] = -o_rope_0 * sin_vals + o_rope_1 * cos_vals"""

new_oproj_rope = """        # 1. Inverse GPT-J interleaved RoPE (is_neox_style=False)
        cos_sin_cache = self.rotary_emb.cos_sin_cache  # [max_pos, rope_dim]
        cs = cos_sin_cache[positions]  # [T, rope_dim]
        cos_vals = cs[:, :half_rope].unsqueeze(1).expand(-1, H, -1)  # [T, H, 32]
        sin_vals = cs[:, half_rope:].unsqueeze(1).expand(-1, H, -1)  # [T, H, 32]

        # Inverse RoPE: conjugate rotation on interleaved pairs
        o_rope = o[:, :, nope_dim:]  # [T, H, 64]
        o_even = o_rope[:, :, 0::2].clone()  # [T, H, 32]
        o_odd = o_rope[:, :, 1::2].clone()   # [T, H, 32]
        o_rope[:, :, 0::2] = o_even * cos_vals + o_odd * sin_vals
        o_rope[:, :, 1::2] = -o_even * sin_vals + o_odd * cos_vals"""

if old_oproj_rope in content:
    content = content.replace(old_oproj_rope, new_oproj_rope)
    with open(path, 'w') as f:
        f.write(content)
    print("[2/3] flashmla.py _o_proj: FIXED inverse RoPE to interleaved style")
else:
    print("[2/3] flashmla.py _o_proj: SKIPPED (text not found)")
    idx = content.find('Inverse GPT-J RoPE')
    if idx >= 0:
        print(f"  Found at offset {idx}")
        print(repr(content[idx:idx+300]))

# ==========================================================================
# Fix 3: fused_compress_quant_cache.py - verify compress fallback
# ==========================================================================
path = f'{SITE}/vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py'
with open(path, 'r') as f:
    content = f.read()

# The compressor fallback uses .view(half_rope, 2) which IS correct for
# interleaved style (pairs[:, 0] = even, pairs[:, 1] = odd).
if 'rope_pairs = rope_part.view(half_rope, 2)' in content:
    print("[3/3] compress fallback: ALREADY CORRECT (uses interleaved pairs via .view(N,2))")
else:
    print("[3/3] compress fallback: text not found - may need manual check")

print("\n=== ROPE INTERLEAVED FIX APPLIED ===")
print("Key change: x[..., :half] / x[..., half:] -> x[..., 0::2] / x[..., 1::2]")
