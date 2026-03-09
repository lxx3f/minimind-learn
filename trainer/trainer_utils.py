"""
训练工具函数集合
"""
import os
import sys
# 设置当前模块的包名，方便跨目录导入时的模块识别
__package__ = "trainer"
# 将当前文件所在目录的上一级目录添加到Python的模块搜索路径中
# os.path.dirname(__file__)：获取当前文件所在目录
# os.path.join(..., '..')：拼接上一级目录路径
# os.path.abspath()：转换为绝对路径
# sys.path.append()：添加到模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist # 导入PyTorch分布式训练模块，用于多卡/多机分布式训练
from torch.nn.parallel import DistributedDataParallel # 导入PyTorch分布式数据并行模块，用于模型多卡并行训练
from torch.utils.data import Sampler
from transformers import AutoTokenizer
# 导入自定义的MiniMind因果语言模型（Causal Language Model）
# 什么是因果语言模型？CLM是只基于上文（左侧）信息预测下一个 token 的语言模型，
# 特点是模型只能看到前面的内容，无法看到后面的内容，符合人类语言生成的逻辑（从左到右、上下文因果）。
# 主流的大语言模型都是CLM
from model.model_minimind import MiniMindForCausalLM 

def get_model_params(model, config):
    """
    计算并打印模型参数信息（总参数、激活参数等，适配MoE模型）
    Args:
        model: 待计算参数的PyTorch模型
        config: 模型配置对象（包含MoE相关参数）
    """
    # model.parameters()返回一个可迭代的参数生成器，包含模型中所有可训练的参数张量（torch.Tensor）
    # p.numel()：计算单个参数张量的元素个数
    # sum()：累加所有参数的元素个数
    # /1e6：转换为百万数量级
    total = sum(p.numel() for p in model.parameters()) / 1e6
    # 获取MoE（混合专家模型）相关配置参数，兼容不同命名方式
    # getattr(a, b, c)：获取a的b属性，若不存在则返回c
    # n_routed：路由专家数量（MoE模型的核心参数）
    # n_active：每个token激活的专家数量
    # n_shared：共享专家数量（所有token一定会用到的专家，保证模型的通用能力）
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    # 计算单个路由专家的参数数量（通过参数名匹配）
    # 只统计参数名包含'mlp.experts.0.'的参数，代表单个专家的参数
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    # 计算单个共享专家的参数数量
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    # 计算模型基础参数（非专家部分）
    # 总参数 - 所有路由专家参数 - 所有共享专家参数
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    # 计算模型实际激活的参数（基础参数 + 激活的专家参数 + 共享专家参数）
    active = base + (expert * n_active) + (shared_expert * n_shared)
    # 打印参数信息：如果激活参数小于总参数（MoE模型），则显示总参数和激活参数
    # 否则（普通模型）只显示总参数
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    """
    判断当前进程是否是分布式训练的主进程（rank=0）
    Returns:
        bool: True表示是主进程，False表示不是
    """
    # dist.is_initialized()：判断是否初始化了分布式训练
    # dist.get_rank() == 0：获取当前进程的rank，0为主进程
    # 非分布式训练时，默认是主进程
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """
    封装打印函数，只在主进程打印日志
    Args:
        content: 要打印的日志内容
    """
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    计算余弦退火学习率（带预热的余弦衰减）
    Args:
        current_step: 当前训练步数
        total_steps: 总训练步数
        lr: 初始学习率
    Returns:
        float: 当前步数的学习率
    """
    # 余弦退火公式：lr * (0.1 + 0.45*(1 + cos(pi*current_step/total_steps)))
    # 最终学习率会衰减到初始的10%（0.1倍），中间是平滑的余弦曲线
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    """
    初始化分布式训练模式
    Returns:
        int: 本地rank（非分布式训练返回0）
    """
    # os.environ.get("RANK", -1)：获取环境变量RANK，不存在则返回-1
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP（分布式数据并行）模式，返回本地rank=0
    
    # 初始化分布式训练进程组，使用NCCL后端（GPU分布式训练的主流后端）
    dist.init_process_group(backend="nccl")
    # 获取本地rank（单台机器内的进程编号）
    local_rank = int(os.environ["LOCAL_RANK"])
    # 设置当前进程使用的GPU设备
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    """
    设置全局随机种子，保证实验可复现
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置cuDNN为确定性算法（保证结果可复现，但可能降低速度）
    torch.backends.cudnn.deterministic = True
    # 关闭cuDNN的自动调优（benchmark），保证可复现性
    # cuDNN库为同一个卷积 / 矩阵操作提供了多种不同的算法实现（比如不同的并行策略、内存访问方式），
    # benchmark=True 时，程序会先跑一遍「基准测试」，把所有可用算法都试一遍，选出最快的那个固定使用；
    # benchmark=False 时，则直接使用默认算法，不做测试选择。
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    """
    模型 checkpoint 保存/加载函数
    Args:
        lm_config: 语言模型配置对象
        weight: 权重名称前缀
        model: 要保存的模型（None表示加载模式）
        optimizer: 优化器（用于保存训练状态）
        epoch: 当前训练轮数
        step: 当前训练步数
        wandb: wandb日志对象（用于保存实验ID）
        save_dir: 保存目录
        **kwargs: 其他要保存的状态
    Returns:
        dict: 加载模式下返回checkpoint数据，否则返回None
    """
    os.makedirs(save_dir, exist_ok=True)
    # 根据是否使用MoE模型添加后缀
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None: # 保存模式
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    """
    初始化模型和tokenizer
    Args:
        lm_config: 模型配置对象
        from_weight: 预训练权重类型（'none'表示不加载）
        tokenizer_path: tokenizer路径
        save_dir: 权重文件所在目录
        device: 模型加载的设备（cuda/cpu）
    Returns:
        tuple: (模型对象, tokenizer对象)
    """
    # 从指定路径加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    # 如果不是从头训练（from_weight != 'none'），加载预训练权重
    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    # 打印模型参数信息
    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """
    自定义采样器：跳过前N个batch（用于恢复训练时跳过已训练的batch）
    继承自PyTorch的Sampler基类
    """
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)