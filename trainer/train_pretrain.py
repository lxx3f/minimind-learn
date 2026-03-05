import os
import sys

__package__ = "trainer" # 定义当前模块的包名，方便模块间引用
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # 增加系统路径，方便查找引用其他模块

import argparse # 命令行参数解析
import time # 计时
import warnings # 警告处理
import torch # PyTorch核心
import torch.distributed as dist # 分布式训练
from contextlib import nullcontext 
from torch import optim, nn # 优化器和神经网络模块
from torch.nn.parallel import DistributedDataParallel # PyTorch 内置的分布式训练工具
from torch.utils.data import DataLoader, DistributedSampler # 数据加载、分布式采样器
from model.model_minimind import MiniMindConfig # 导入模型配置
from dataset.lm_dataset import PretrainDataset # 导入预训练数据集
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler # 工具函数

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    一个epoch的完整训练流程，包括数据迭代、前向传播、反向传播、参数更新、日志打印和模型保存

    epoch: int，当前训练的轮次（从0开始计数，与主循环的epoch保持一致）
    loader: DataLoader，当前epoch的数据加载器，用于迭代获取批量训练数据（input_ids, labels）
    iters: int，当前epoch的总训练步数（即loader的总batch数，用于计算学习率和剩余时间）
    start_step: int，可选参数，默认0，续训时的起始步数（跳过前start_step个batch，从该步数后继续训练）
    wandb: swanlab/wandb实例，可选参数，默认None，用于记录训练过程中的各项指标（损失、学习率等）

    """
    # 记录开始时间
    start_time = time.time()
    # 遍历数据加载器，step从start_step+1开始（支持断点续训）
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 数据移到指定设备（GPU/CPU）
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # 动态计算学习率（基于当前总步数和总训练步数）
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # PyTorch 的optimizer.param_groups是一个列表，包含了优化器管理的所有参数组（每组对应模型中一类参数），需遍历所有组才能让全局参数的学习率同步更新。
        # 这里实际只有一个参数组，但是遍历写法是通用规范，即使后续调整为多参数组（如区分权重 / 偏置、冻结部分参数），代码也无需修改，保证兼容性和可扩展性。
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            # 前向传播：输入input_ids，传入labels计算损失
            # （res为模型输出对象，包含loss、aux_loss等）
            res = model(input_ids, labels=labels)
            # 主损失：模型预测结果（logits）与真实标签（labels）的差距。
            # 辅助损失：不直接对应核心任务，而是为了解决训练中的特定问题（如模型稳定性、架构适配性），加速主损失收敛或提升模型泛化能力。
            loss = res.loss + res.aux_loss
            # 梯度累积：损失除以累积步数（避免单次batch_size过小导致loss波动过大）
            loss = loss / args.accumulation_steps

        # 反向传播（混合精度场景）：scaler缩放损失，防止梯度下溢
        # scaler 是 torch.cuda.amp.GradScaler 实例，专门用于混合精度训练的梯度处理；
        # 具体做的事：将当前计算出的损失（loss）乘以一个缩放因子（比如 2^16），把原本可能很小的损失值 “放大” 到半精度能正常表示的范围；
        # 放大后的损失再通过 .backward() 做反向传播，计算出的梯度也会同步被放大（避免梯度下溢，保证梯度有有效数值）。
        # 注：缩放只是 “临时操作”，后续会有 “反缩放” 抵消
        scaler.scale(loss).backward()

        # 梯度累积到指定步数后，执行参数更新
        if (step + 1) % args.accumulation_steps == 0:
            # 反缩放优化器梯度，抵消前面的放大
            scaler.unscale_(optimizer)
            # 梯度裁剪：防止梯度爆炸，阈值为args.grad_clip，限制梯度的最大范数
            # 当范数超过阈值时，对所有参数的梯度等比例缩小
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 优化器更新模型参数
            scaler.step(optimizer)
            # 更新scaler的缩放因子，适配下一轮梯度计算
            scaler.update()

            # 清空梯度（set_to_none=True比zero_()更高效，节省显存）
            optimizer.zero_grad(set_to_none=True)

        # 打印训练日志（按log_interval间隔，或当前为最后一步时打印）
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps # 恢复真实损失（抵消梯度累积的除法）
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            # 预估本轮训练剩余时间（单位：分钟）
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            # 若启用wandb，记录当前训练指标（用于后续可视化分析）
            if wandb: 
                wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})
        # 保存模型（按save_interval间隔，或当前为最后一步时保存，且仅主进程执行，避免多进程重复保存）
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            # 模型切到评估模式（禁用BN层、Dropout，避免影响保存的权重）
            model.eval()
            # 生成模型文件名（区分是否使用MoE架构，便于后续识别和加载）
            moe_suffix = '_moe' if lm_config.use_moe else ''
            # ckp是checkpoint的缩写，即模型检查点文件
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            # 获取模型权重字典
            state_dict = raw_model.state_dict()
            # 保存权重（转半精度+CPU，大幅节省存储空间）
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 保存完整检查点（包含模型、优化器、scaler、epoch、step等，支持后续续训）
            # lm前缀是language model的缩写
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            # 模型切回训练模式，继续后续训练
            model.train()
            # 释放内存，避免显存溢出
            del state_dict
        # 清理当前step的变量，释放显存（避免显存累积导致溢出）
        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    # wandb是weights and biases的缩写，是训练过程监测工具
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    # lm_config：MiniMindConfig配置实例，包含模型核心结构参数（hidden_size、num_hidden_layers、use_moe等），用于定义模型的网络结构
    # from_weight：命令行传入的权重路径，若为'none'（默认），则从头初始化模型权重；若传入具体权重路径（如../out/pretrain_512.pth），则加载该路径下的模型权重，实现"基于已有权重继续训练"
    # device：训练设备（GPU/CPU），函数会将初始化后的模型自动移动到该设备上，避免后续手动移动导致的设备不匹配报错
    # 返回两个实例（model和tokenizer），后续训练中，tokenizer用于处理数据，model用于执行前向/反向传播
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 启用torch.compile加速（PyTorch 2.0+特性，提升训练速度）
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 分布式采样器（多GPU训练时，将数据集均分至各个GPU，避免数据重复）
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 混合精度缩放器（仅float16类型时启用，防止梯度下溢）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # AdamW优化器（用于更新模型参数）
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    # 分布式训练
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()