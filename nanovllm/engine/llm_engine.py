import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        # 1. 调度：获取两组序列
        prefill_seqs, decode_seqs = self.scheduler.schedule()
        
        # 2. 执行：传入两组序列
        # 注意：ModelRunner.run 内部会将 input_ids 拼接为 [Prefill | Decode]
        # 因此返回的 token_ids 顺序也是 [Prefill_Result | Decode_Result]
        token_ids = self.model_runner.call("run", prefill_seqs, decode_seqs)
        
        # 3. [关键修改] 合并序列，且顺序必须与 input_ids 拼接顺序一致
        all_seqs = prefill_seqs + decode_seqs
        
        # 4. 后处理：现在 all_seqs 和 token_ids 是一一对应的了
        self.scheduler.postprocess(all_seqs, token_ids)
        
        # 5. [新增] Chunked Prefill 进度更新
        # 必须手动更新进度，否则下次调度会重复跑这部分
        for seq in prefill_seqs:
            if hasattr(seq, "prefill_chunk_size"):
                seq.num_cached_tokens += seq.prefill_chunk_size
                # 清理标记，保持对象干净
                del seq.prefill_chunk_size
        
        # 6. 收集输出 (只收集已完成的)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in all_seqs if seq.is_finished]
        
        # 7. 计算统计量 (用于 tqdm 进度条显示)
        # 现在的 step 可能同时包含 Prefill 和 Decode，原有的简单的正负号逻辑不够用了
        # 这里给出一个简单的兼容方案：优先显示 Prefill 吞吐
        if prefill_seqs:
            # 计算本次 Prefill 处理了多少 Token
            num_tokens = sum(len(seq) - seq.num_cached_tokens for seq in prefill_seqs)
        else:
            # 如果只有 Decode，用负数表示 (兼容原有的 generate 函数逻辑)
            num_tokens = -len(decode_seqs)
            
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
