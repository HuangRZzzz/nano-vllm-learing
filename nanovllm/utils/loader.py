import os
from glob import glob
import torch
from torch import nn
from safetensors.torch import safe_open

from nanovllm.layers.linear import LinearBase, RowParallelLinear, ColumnParallelLinear, QKVParallelLinear
# 导入上面定义的 AWQLinear
from nanovllm.layers.quantization import AWQLinear

def default_weight_loader(param, loaded_weight):
    """默认权重加载器：直接拷贝数据"""
    try:
        param.data.copy_(loaded_weight)
    except RuntimeError:
        # 简单的形状容错处理 (例如 view 之后拷贝)
        if param.numel() == loaded_weight.numel():
             param.data.copy_(loaded_weight.view_as(param))
        else:
             raise

def get_module_by_name(model, name):
    """辅助函数：通过 'layers.0.attn' 这样的名字获取模块对象"""
    parts = name.split('.')
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module

def recursive_get_param(model, name):
    """辅助函数：通过 'layers.0.weight' 获取参数，兼容 PyTorch 旧版本"""
    try:
        return model.get_parameter(name)
    except (AttributeError, ValueError):
        parts = name.split('.')
        module = model
        for part in parts[:-1]:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return getattr(module, parts[-1])

def load_model(model: nn.Module, path: str):
    """
    加载权重主函数。
    特性:
    1. 支持 Safetensors
    2. 支持 AWQ 自动层替换 (Auto-Quantization)
    3. 支持 Packed Modules (如 QKV 合并)
    """
    # 获取模型定义的打包映射 (如果有)
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    # 兼容目录路径或单个文件路径
    if os.path.isdir(path):
        files = glob(os.path.join(path, "*.safetensors"))
    else:
        files = [path]

    print(f"Loading weights from {path}...")

    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                device = "cuda"
                # ============================================
                # 1. AWQ 量化逻辑 (优先级最高)
                # ============================================
                is_awq_weight = any(s in weight_name for s in ["qweight", "qzeros", "scales"])
                
                if is_awq_weight:
                    # 推断模块名: model.layers.0.self_attn.q_proj.qweight -> ...q_proj
                    module_name = weight_name.rsplit('.', 1)[0]
                    suffix = weight_name.split('.')[-1] # qweight / scales / qzeros
                    
                    try:
                        module = get_module_by_name(model, module_name)
                    except AttributeError:
                        # 如果找不到模块，可能是路径不对，跳过
                        continue

                    # 动态替换: 如果遇到普通 Linear 层但权重是量化的，替换为 AWQLinear
# ... 前面的代码 ...
                    
                    # 动态替换逻辑
                    # 关键修改：检查它是不是 LinearBase，并且 绝不能 是已经替换过的 AWQLinear
                    # 注意：如果不加 and not isinstance(module, AWQLinear)，就会重复初始化导致丢数据
# ... (前文：解析 module_name 和 suffix) ...
                    
                    # 1. 动态替换逻辑 (造壳)
                    # 只有当它是普通 Linear 且还没被变成 AWQLinear 时，才进行替换
def load_model(model: nn.Module, path: str, device="cuda"):
    """
    加载权重主函数 (修复 merged layer 丢失问题)
    """
    # 1. 获取映射表: {"gate_proj": ["gate_up_proj", 0], "up_proj": ["gate_up_proj", 1]}
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    if os.path.isdir(path):
        files = glob(os.path.join(path, "*.safetensors"))
    else:
        files = [path]

    print(f"Loading weights from {path}...")

    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                
                # ============================================
                # 1. AWQ 量化逻辑
                # ============================================
                is_awq_weight = any(s in weight_name for s in ["qweight", "qzeros", "scales"])
                
def load_model(model: nn.Module, path: str, device="cuda"):
    """
    加载权重主函数 (修复 QKV 字符索引导致的 TypeError)
    """
    # 1. 获取映射表
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    if os.path.isdir(path):
        files = glob(os.path.join(path, "*.safetensors"))
    else:
        files = [path]

    print(f"Loading weights from {path}...")

    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                
                # ============================================
                # 1. AWQ 量化逻辑
                # ============================================
                is_awq_weight = any(s in weight_name for s in ["qweight", "qzeros", "scales"])
                
                if is_awq_weight:
                    module_name_in_file = weight_name.rsplit('.', 1)[0]
                    suffix = weight_name.split('.')[-1] # qweight / scales / qzeros
                    
                    target_module_name = module_name_in_file
                    shard_id = None
                    
                    # 检查映射表 (gate -> gate_up, q -> qkv)
                    for source_key, target_info in packed_modules_mapping.items():
                        if module_name_in_file.endswith(source_key):
                            target_key, sid = target_info
                            target_module_name = module_name_in_file.replace(source_key, target_key)
                            shard_id = sid
                            break
                    
                    try:
                        module = get_module_by_name(model, target_module_name)
                    except AttributeError:
                        continue

                    # === 动态替换 Logic (造壳) ===
                    if isinstance(module, (nn.Linear, LinearBase)) and not isinstance(module, AWQLinear):
                        out_features, in_features = module.weight.shape
                        print(f"Quantizing {target_module_name} ({in_features}x{out_features}) ...")
                        
                        new_module = AWQLinear(
                            in_features, 
                            out_features, 
                            group_size=128,

                        ).to(device)

                        # 【关键修复1】: 复制 QKV 的头数信息，用于后续计算偏移
                        if hasattr(module, "num_heads"): new_module.num_heads = module.num_heads
                        if hasattr(module, "num_kv_heads"): new_module.num_kv_heads = module.num_kv_heads
                        if hasattr(module, "head_size"): new_module.head_size = module.head_size
                        
                        # 挂载
                        if '.' in target_module_name:
                            parent_name, attr_name = target_module_name.rsplit('.', 1)
                            parent = get_module_by_name(model, parent_name)
                        else:
                            parent_name = ""
                            attr_name = target_module_name
                            parent = model
                        
                        if isinstance(parent, (nn.ModuleList, nn.Sequential)):
                            parent[int(attr_name)] = new_module
                        else:
                            setattr(parent, attr_name, new_module)
                        
                        module = new_module
                        
                        module.qweight = module.qweight.to(device)
                        module.qscales = module.qscales.to(device)
                        module.qzeros = module.qzeros.to(device)

                    # === 数据加载 Logic (填肉) ===
                    if isinstance(module, AWQLinear):
                        if hasattr(module, suffix):
                            target_tensor = getattr(module, suffix)
                            loaded_weight = f.get_tensor(weight_name)
                            
                            if shard_id is not None:
                                # 【关键修复2】: 处理字符串 ID 和 QKV 偏移
                                cat_dim = 1 # 都在 dim=1 拼接
                                
                                # 计算真实的 start_idx
                                if isinstance(shard_id, str):
                                    # QKV 逻辑: 必须用头数计算精确偏移
                                    if not hasattr(module, "head_size"):
                                        raise ValueError(f"Module {target_module_name} 缺少 head_size 信息，无法处理 QKV 切片")
                                        
                                    head_dim = module.head_size
                                    if shard_id == "q":
                                        real_offset = 0
                                        shard_size = module.num_heads * head_dim
                                    elif shard_id == "k":
                                        real_offset = module.num_heads * head_dim
                                        shard_size = module.num_kv_heads * head_dim
                                    elif shard_id == "v":
                                        real_offset = (module.num_heads + module.num_kv_heads) * head_dim
                                        shard_size = module.num_kv_heads * head_dim
                                    else:
                                        raise ValueError(f"Unknown shard_id: {shard_id}")
                                    
                                    # 考虑 AWQ 打包压缩 (除以8)
                                    # qweight 和 qzeros 是压缩的，scales 是未压缩的
                                    packing_factor = 1
                                    if suffix in ["qweight", "qzeros"]:
                                        packing_factor = 8
                                    
                                    start_idx = real_offset // packing_factor
                                    shard_size = shard_size // packing_factor
                                    
                                else:
                                    # MLP 逻辑 (gate=0, up=1): 
                                    # shard_id 是整数，且切片大小相等，且 shard_size 已经是压缩后的大小
                                    # 尝试转成 int 防止 config 里是字符串 "0"
                                    sid_int = int(shard_id)
                                    # shard_size = loaded_weight.shape[cat_dim]
                                    start_idx = sid_int * loaded_weight.shape[cat_dim]

                                end_idx = start_idx + loaded_weight.shape[cat_dim]
                                
                                # 拷贝数据
                                if shard_id == "k" or shard_id == "q" or shard_id == "v":
                                    target_tensor.narrow(cat_dim, start_idx, shard_size).copy_(loaded_weight.to(device))
                                else:
                                    target_tensor.narrow(cat_dim, start_idx, loaded_weight.shape[cat_dim]).copy_(loaded_weight.to(device))
                                
                            else:
                                if target_tensor.shape != loaded_weight.shape:
                                    target_tensor.copy_(loaded_weight.view_as(target_tensor))
                                else:
                                    target_tensor.copy_(loaded_weight.to(device))
                    
                    continue 

                # ... (后续常规加载代码保持不变) ...

                # ... (后续的常规权重加载逻辑保持不变) ...
                # ============================================
                # 2. Packed Modules 逻辑 (QKV 合并等)
                # ============================================
                param = None
                shard_id = None
                
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 映射文件名到参数名: q_proj -> qkv_proj
                        v, s_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        try:
                            param = recursive_get_param(model, param_name)
                            shard_id = s_id
                        except AttributeError:
                            pass
                        break
                
                # ============================================
                # 3. 常规权重加载
                # ============================================
                if param is None:
                    try:
                        param = recursive_get_param(model, weight_name)
                    except AttributeError:
                        continue

                # 使用参数自带的 loader 或者默认 loader
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                
                loaded_weight = f.get_tensor(weight_name)
                
                if shard_id is not None:
                    weight_loader(param, loaded_weight, shard_id)
                else:
                    weight_loader(param, loaded_weight)
    # 在 return model 之前
    print("Checking LayerNorm weights...")
    for name, param in model.named_parameters():
        if "norm" in name and "weight" in name:
            print(f"{name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
            if param.data.mean().item() == 0 and param.data.std().item() == 0:
                print(f"!!! DANGER: {name} is ALL ZEROS! Output will be garbage.")
    print("Weights loaded successfully.")