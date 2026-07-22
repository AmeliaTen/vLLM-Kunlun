"""
Patch SparseAttnIndexer to use a no-op fallback on Kunlun.

The indexer requires DeepGEMM (fp8_fp4_paged_mqa_logits) which is unavailable.
For minimum viable token generation, we make the indexer a no-op:
- topk_indices_buffer stays filled with -1 (no sparse tokens selected)
- The attention will only use the SWA window path

This is functionally degraded (no long-range sparse attention) but correct
enough for first token generation.
"""

SITE = '/opt/vllm_kunlun/lib/python3.10/site-packages'

# Patch the KunlunSparseAttnIndexer in vllm_kunlun/__init__.py
# The current code does: forward_oot -> forward_cuda -> DeepGEMM (fails)
# We need to make forward_oot return topk_indices_buffer directly (no-op)

path = f'{SITE}/vllm_kunlun/__init__.py'
with open(path, 'r') as f:
    content = f.read()

old_indexer = '''    @CustomOp.register_oot(name="SparseAttnIndexer")
    class KunlunSparseAttnIndexer(mod.SparseAttnIndexer):
        def forward_oot(self, hidden_states, q_quant, k, weights):
            return self.forward_cuda(hidden_states, q_quant, k, weights)'''

new_indexer = '''    @CustomOp.register_oot(name="SparseAttnIndexer")
    class KunlunSparseAttnIndexer(mod.SparseAttnIndexer):
        def forward_oot(self, hidden_states, q_quant, k, weights):
            # Kunlun no-op fallback: no DeepGEMM available.
            # Fill topk_indices with -1 (no sparse tokens selected).
            # Attention will rely solely on SWA window.
            import torch
            num_tokens = hidden_states.shape[0]
            self.topk_indices_buffer[:num_tokens] = -1
            return self.topk_indices_buffer'''

if old_indexer in content:
    content = content.replace(old_indexer, new_indexer)
    with open(path, 'w') as f:
        f.write(content)
    print("[OK] KunlunSparseAttnIndexer patched to no-op fallback")
else:
    print("[SKIP] Could not find exact KunlunSparseAttnIndexer text")
    # Try to find it
    idx = content.find("KunlunSparseAttnIndexer")
    if idx >= 0:
        print(f"  Found at offset {idx}")
        print(repr(content[idx:idx+300]))
