from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.max_num_batched_tokens = config.max_num_batched_tokens

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)
    



    def schedule(self) -> tuple[list[Sequence], list[Sequence]]:
            # 1. Decode 调度
            decode_seqs = []
            # 使用 list() 创建副本，防止 preempt 修改队列导致遍历出错
            for seq in list(self.running):
                if not self.block_manager.can_append(seq):
                    self.preempt(seq) # 显存不够就踢人
                else:
                    self.block_manager.may_append(seq)
                    decode_seqs.append(seq)
            
            # 2. 计算剩余预算
            num_decode_tokens = len(decode_seqs) 
            remaining_budget = self.max_num_batched_tokens - num_decode_tokens
            
            # 3. Prefill 调度
            prefill_seqs = []
            while self.waiting and remaining_budget > 0:
                seq = self.waiting[0]
                
                # 检查能否分配（注意：这里检查的是完整序列的显存）
                # 如果想支持超长序列推理(超过显存)，需要改造 allocate 支持部分分配，目前 nano-vllm 不支持
                if not self.block_manager.can_allocate(seq):
                    break
                
                if not seq.block_table:
                    self.block_manager.allocate(seq)
                
                # --- 修正变量名 ---
                num_needed_tokens = len(seq) - seq.num_cached_tokens

                chunk_size = min(num_needed_tokens, remaining_budget)
                
                seq.prefill_chunk_size = chunk_size # 标记本次跑多少
                prefill_seqs.append(seq)
                remaining_budget -= chunk_size
                
                # 状态管理
                if num_needed_tokens <= 0:
                # 既然已经存完了，说明 Prefill 早就做完了
                    self.waiting.popleft()
                    self.running.append(seq)
                    seq.status = SequenceStatus.RUNNING
                
                if chunk_size == num_needed_tokens:
                    # 跑完了：移出 waiting，进入 running
                    self.waiting.popleft()
                    self.running.append(seq)
                    seq.status = SequenceStatus.RUNNING
                else:
                    # 没跑完：留在 waiting 队头，下次继续
                    # --- 修正逻辑 ---
                    # 必须 break！因为 seq 还在队头，且 num_cached_tokens 还没更新。
                    # 不能在同一个 step 里调度同一个 seq 两次。
                    break 

            return prefill_seqs, decode_seqs
            
        
        
# '''

#     nano-vllm原有的schedule调度逻辑

#     def schedule(self) -> tuple[list[Sequence], bool]:
#         # prefill
#         scheduled_seqs = []
#         num_seqs = 0
#         num_batched_tokens = 0
#         while self.waiting and num_seqs < self.max_num_seqs:
#             seq = self.waiting[0]
#             if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
#                 #break 如果序列无法分配或批处理已满
#                 break
#             num_seqs += 1
#             self.block_manager.allocate(seq)
#             num_batched_tokens += len(seq) - seq.num_cached_tokens
#             seq.status = SequenceStatus.RUNNING
#             self.waiting.popleft()
#             self.running.append(seq)
#             scheduled_seqs.append(seq)
        

#         # decode
#         while self.running and num_seqs < self.max_num_seqs:
#             seq = self.running.popleft()
#             # 
#             while not self.block_manager.can_append(seq):

#                 if self.running:
#                     self.preempt(self.running.pop())
#                 else:
#             # no more running seqs to preempt
#                     self.preempt(seq)
#                     break
#             else:
#                 num_seqs += 1
#                 self.block_manager.may_append(seq)
#                 scheduled_seqs.append(seq)
#         # assetr
#         assert scheduled_seqs
#         #重新入队，确保上文信息不丢失，
#         self.running.extendleft(reversed(scheduled_seqs))
#         return scheduled_seqs, False
# '''

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)


# 将新生成的token_ids添加到对应的seq中，并检查是否完成 或者超出最大token数
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                # 从running队列中移除，并释放对应的block
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
