# LoRA原理
LoRA（Low-Rank Adaptation，低秩适配）

核心思想：
低秩假设：权重更新在模型微调过程中具有“低本征维度（Low Intrinsic Dimension）”。

简单来说，虽然大模型的参数矩阵（如d×d维）非常大，但在针对特定任务进行微调时，权重的变化（ΔW）并不需要改变所有的方向，实际上只需要在一个极小的低维子空间内变动就能达到很好的效果。

数学原理：
在普通的微调中，参数矩阵W可以分解为：
$$
W_{nem} = W_0 + \delta W
$$

其中$W_0$是预训练模型的原始权重，$\delta W$是训练过程的梯度更新。

LoRA 的做法是：
1. 冻结（Freeze）原始权重$W_0$，在训练过程中不改变它。
2. 分解（Decomposition）将$\delta W$分解为两个极小的矩阵A和B的乘积：$\delta W = B \times A$

假设：
1. $w_0$的维度是dxk
2. A的维度是rxk
3. B的维度是dxr

这里的r称为秩(rank)，通常设置为很小（相对于d和k）的值，如4,8,16；

初始化：
- A通常采用高斯分布随机初始化
- B初始化为全0
- 初始时$B\times A=0$


# 训练
## identity
```
python train_lora.py --lora_name lora_identity --data_path ../../autodl-tmp/dataset/lora_identity.jsonl --from_weight full_sft --use_wandb --ckp_save_dir ../../autodl-tmp/checkpoints
```

测试:
```

```


```
python train_lora.py --lora_name lora_medical --data_path ../../autodl-tmp/dataset/lora_medical.jsonl --from_weight full_sft --use_wandb 
--ckp_save_dir ../../autodl-tmp/checkpoints
```