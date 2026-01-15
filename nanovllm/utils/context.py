from dataclasses import dataclass
import torch

@dataclass
class Context:
    is_prefill: bool = False
    is_mixed: bool = False  # [新增] 混合批处理标志
    
    # 混合模式下的分界线
    num_prefill_tokens: int = 0  # [新增] Prefill 部分的 Token 数量
    
    # Attention 元数据
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    
    # 页表 (Block Tables)
    block_tables: torch.Tensor | None = None  # 兼容旧逻辑
    prefill_block_tables: torch.Tensor | None = None # [新增] Prefill 专用页表 (Prefix Caching)
    decode_block_tables: torch.Tensor | None = None  # [新增] Decode 专用页表

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, 
                cu_seqlens_q=None, cu_seqlens_k=None, 
                max_seqlen_q=0, max_seqlen_k=0, 
                slot_mapping=None, context_lens=None, 
                block_tables=None,
                # [新增参数]
                is_mixed=False,
                num_prefill_tokens=0,
                prefill_block_tables=None,
                decode_block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, is_mixed, num_prefill_tokens,
                       cu_seqlens_q, cu_seqlens_k, 
                       max_seqlen_q, max_seqlen_k, 
                       slot_mapping, context_lens, 
                       block_tables, prefill_block_tables, decode_block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()