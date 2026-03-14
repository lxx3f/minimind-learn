```
python train_dpo.py --data_path ../../autodl-tmp/dataset/dpo.jsonl --use_wandb --ckp_save_dir ../../autodl-tmp/checkpoints
```


# DPO原理
DPO（Direct Preference Optimization，直接偏好优化）是一种无需强化学习（RL）就能对齐大语言模型与人类偏好的训练方法，核心是直接利用人类的 “偏好数据”（比如更喜欢回答 A 而不是回答 B）来优化模型，避免了传统 RLHF（强化学习人类反馈）中复杂的奖励模型训练和策略优化步骤。

传统RLHF对齐分为三步：
- 训练 SFT 模型（有监督微调）；
- 训练 RM 模型（奖励模型，给回答打分）；
- 用 RL（强化学习）优化模型，让输出符合 RM 的高分偏好。
但RLHF有明显问题：
- 训练复杂（需要先训 RM 再训 RL）；
- 容易出现 “奖励崩塌”（模型只刷高分，内容质量下降）；
- 训练不稳定（RL 的超参数敏感）。

核心假设
人类的偏好可以用 “概率比” 来表示：如果人类更喜欢回答 y1​（chosen）而非 y2​（rejected），那么策略模型（待训练模型）对y1​的概率应该远大于对y2​的概率，且这个概率比要超过一个 “参考模型”（通常是 SFT 模型，未对齐的基线模型）的概率比。
