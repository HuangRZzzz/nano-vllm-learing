import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache = self.k_cache
        v_cache = self.v_cache
        
        # 1. 统一写入 KV Cache (Mixed 模式下 slot_mapping 已经拼好了)
        if k_cache.numel() > 0:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        # 2. 混合计算逻辑
        if context.is_mixed:
            # 切分点
            N = context.num_prefill_tokens
            
            # === Part A: Prefill (Varlen) ===
            q_p = q[:N]
            k_p = k[:N]
            v_p = v[:N]
            
            o_p = flash_attn_varlen_func(
                q_p, k_p, v_p,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.prefill_block_tables # 如果有 Prefix Cache
            )
            
            # === Part B: Decode (Paged) ===
            q_d = q[N:]
            # Decode 不需要 k_d, v_d，直接读 Cache
            o_d = flash_attn_with_kvcache(
                q_d.unsqueeze(1), # [Batch, 1, Head, Dim]
                k_cache, v_cache,
                block_table=context.decode_block_tables, # 独立的 decode 页表
                cache_seqlens=context.context_lens,
                softmax_scale=self.scale,
                causal=True
            )
            
            # 3. 拼接结果
            return torch.cat([o_p.view(N, -1, self.num_heads * self.head_dim), 
                              o_d.view(-1, 1, self.num_heads * self.head_dim).squeeze(1)])
        
        elif context.is_prefill:
            k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)


        return o
