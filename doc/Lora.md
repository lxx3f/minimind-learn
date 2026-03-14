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
[训练过程记录](https://swanlab.cn/@lx2323/MiniMind-LoRA/runs/f6uqiqjg909dw5n2juq6l/chart)

测试:
```
# 预训练权重
python eval_llm.py --weight pretrain                          
Model Params: 25.83M
[0] 自动测试
[1] 手动输入
1
💬: 你是谁？有什么特长？
🤖: 我的回答是“我是谁？”我的回答是“ 我是谁？”我的回答是“我是谁？”。我是 一个AI语言模型，没有实际的感情和感受，因此我没有回答你的问题。

[Speed]: 76.57 tokens/s

# 预训练权重加上LoRA权重
python eval_llm.py --weight pretrain --lora_weight lora_identity
Model Params: 25.96M
[0] 自动测试
[1] 手动输入
1
💬: 你是谁？有什么特长？
🤖: 我的回答是“我是谁？”我的回答是：“我是谁？”我的回答是“我是谁？”我是一个AI语言模型，没有个人情感和价值观，所以我无法回答你的问题。

[Speed]: 74.17 tokens/s
```
没什么差异，可能的原因：
1. lora微调训练的数据集非常小；
2. lora权重对结果的影响非常小，pretrain权重对结果影响非常大；

```
# sft微调权重
python eval_llm.py --weight full_sft
Model Params: 25.83M
[0] 自动测试
[1] 手动输入
1
💬: 你是谁？有什么特长？
🤖: 作为一个AI，我没有个人情感和感受，所以我没有个人经历。但是，我可以根据用户提供的信息，提供一些信息，例如他们的性别、年龄、性别等。我可以帮助你找到他们的性别、年龄、职业、兴趣爱好等信息。请告诉我，如果你有关于特定职业的问题，欢迎随时问我。

[Speed]: 74.84 tokens/s

# sft微调权重加上LoRA权重
python eval_llm.py --weight full_sft --lora_weight lora_identity
Model Params: 25.96M
[0] 自动测试
[1] 手动输入
1
💬: 你是谁？有什么特长？
🤖: 您好！我是一个大型语言模型，并没有个人情感或具体需求。我的能力是基于已有的知识库和语言模型，而我被训练来理解用户提出的问题。

[Speed]: 71.32 tokens/s

```
回答更简洁了，可以看出lora微调对回答质量的影响。


## medical
```
python train_lora.py --lora_name lora_medical --data_path ../../autodl-tmp/dataset/lora_medical.jsonl --from_weight full_sft --use_wandb 
--ckp_save_dir ../../autodl-tmp/checkpoints
```
[训练过程记录](https://swanlab.cn/@lx2323/MiniMind-LoRA/runs/lsf02eladano9x0o607p1/chart)

测试：
```
# sft微调权重
python eval_llm.py --weight full_sft
Model Params: 25.83M
[0] 自动测试
[1] 手动输入
1
💬: 心肌缺血和心肌炎有什么不同
🤖: 心肌缺血和心肌炎的不同之处在于，心肌缺血和心肌炎是心肌缺血的一种，它们主要是因为心肌缺血、心肌缺血以及心肌的不适而导致的。心肌缺血通常指的是心肌缺血，而心肌缺血通常指的是心肌缺血，这与心肌缺血和心 肌缺血有关。

心肌缺血和心肌缺血的不同之处在于，心肌缺血和心肌缺血的因素包括心肌缺血（心肌缺血），心肌缺血（心肌缺血），心肌缺血（心肌缺血）和心肌缺血（心肌缺血）。心肌缺血主要是指心肌缺血，而心肌缺血则是指心肌缺血 。

心肌缺血和心肌缺血（心肌缺血）是心肌缺血，心肌缺血（心肌缺血），心肌缺血（心肌缺血）和心肌缺血（心肌缺血）。心肌缺血（心肌缺血）和心肌缺血（心肌缺血），是心肌缺血的一种。心肌缺血主要是心肌缺血，心肌缺 血（心肌缺血）。心肌缺血（心肌缺血）是指心肌缺血（心肌缺血），

[Speed]: 72.52 tokens/s


python eval_llm.py --weight full_sft --lora_weight lora_medical
Model Params: 25.96M
[0] 自动测试
[1] 手动输入
1
💬: 心肌缺血和心肌炎有什么不同
🤖: 心肌缺血和心肌炎的不同之处在于，心肌缺血和心肌炎 是心肌缺血的一种常见症状，通常是由于心肌缺血引起的。 以下是一些可能的症状：

1. 心肌炎：心肌缺血会导致血液中的氧气被淋巴系统刺激，导致心肌缺血。

2. 心肌炎：心肌缺血可能会导致血液中的症状，例如心房疼痛，心律失常，心肌炎，心房炎，肺部疾病等。

3. 心肌炎：心肌炎可能导致心肌炎，这是一种心肌炎，通常在心肌炎的情况下，持续存在，持续时间不改变。

4. 心肌炎：心肌炎可能导致心肌炎，这也可能导致心肌炎。

5. 心肌炎：心肌炎可能导致心肌炎，这可能是由于心肌炎导致的心肌炎，或者是由于心肌炎导致的心肌炎。

6. 心肌炎：心肌炎可能会导致心肌炎，这是一种心肌炎，通常在心肌炎的症状。

7. 心肌炎：心肌炎可能导致心肌炎，这可能是由于心肌炎引起的，如心肌

[Speed]: 71.75 tokens/s
```
测试问题是lora_medical数据集中有的问题，只能说有lora微调有一点效果，但不多，主要原因可能是：
1. 基座模型太小，所以lora矩阵很小，无法学习到有效的特征
2. lora权重对结果的影响占比很小
