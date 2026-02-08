# test_my_kernel.py
import torch
from transformers import AutoTokenizer

# === 关键：导入你自己写的模型，而不是 transformer 的 ===
# 假设你的模型定义在 nanovllm.model.qwen 下
from nanovllm.models.qwen import Qwen3ForCausalLM  # <--- 用你自己的类
from nanovllm.utils.loader import load_model      # <--- 用你自己的加载器

model_path = "/mnt/workspace/nano-vllm/AWQ/AWQ"

# 1. 初始化空模型 (你自己的结构)
# 这里你需要手动构建 config 或者从 json 加载
from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_path)
model = Qwen3ForCausalLM(config).cuda() # 初始化你的 Qwen3

# 2. 使用你辛苦写的 load_model 加载权重
load_model(model, model_path)

# 3. 推理测试
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
input_ids = tokenizer.encode("列出10以内的奇数", return_tensors="pt").cuda()

with torch.no_grad():
    # 调用你自己的 forward
    logits = model(input_ids)
    print("Output Logits Shape:", logits.shape)
    # 简单的贪婪解码测试
    next_token = torch.argmax(logits[:, -1, :], dim=-1)
    print("Next token:", tokenizer.decode(next_token))