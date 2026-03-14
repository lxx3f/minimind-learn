# MiniMind 模型架构

## 一、整体架构框架（自底向上）
MiniMind 整体为 纯解码器（Decoder-only） 架构（与 LLaMA/GPT 一致），核心由「输入层 → N 个 Transformer Decoder Block → 输出层」组成，且支持可选的 MoE（混合专家）扩展，整体流程：
```plaintext
输入Token ID → 词嵌入层 → Dropout → N个Decoder Block → 归一化 → LM Head → 输出Token概率
```

## 二、核心组件
### 1. 输入层（Embedding + 位置编码）

词嵌入层（Embed Tokens）：
- 将输入的 Token ID（维度：[批次大小，序列长度]）映射为稠密向量，维度为 hidden_size（默认 512）；
- 与输出层 LM Head 共享权重（权重绑定），减少参数量。


位置编码（RoPE）：采用旋转位置编码（Rotary Position Embedding）：
- 预计算所有位置的 cos/sin 矩阵（注册为模型 buffer，不参与训练）；
- 支持 YaRN 长度外推：通过调整频率参数，让模型能处理超过训练长度（2048）的序列（最大支持 32768）；
- 仅对注意力层的 Q/K 向量做旋转编码，保留位置信息的同时不引入额外参数量。



### 2. Transformer Decoder Block（核心层）
模型包含 num_hidden_layers 个（默认 8）相同的 Decoder Block，每个 Block 采用 Pre-Norm 架构（区别于原始 Transformer 的 Post-Norm），结构为：
```plaintext
归一化 → 多头注意力 → 残差连接 → 归一化 → FFN/MoE → 残差连接
```

关键子组件：

RMSNorm 归一化：
替代传统 LayerNorm，移除偏置项，仅保留缩放参数，计算更高效；
公式：权重。


多头注意力层（GQA）：
分组查询注意力（Grouped Query Attention）：Q 头数（8）是 K/V 头数（2）的整数倍，每个 K/V 头服务多个 Q 头，平衡计算效率和性能；
支持 FlashAttention：PyTorch 2.0+ 下自动启用，大幅降低显存占用和耗时；
因果掩码：保证每个 Token 仅关注自身及前面的 Token，符合语言模型「预测下一个 Token」的逻辑；
KV Cache：推理时缓存历史 K/V 向量，避免重复计算，提升生成速度。


前馈网络（FFN）：
采用 GLU（门控线性单元）结构：act(gate_proj(x)) * up_proj(x) → down_proj，相比标准 FFN 减少 25% 参数量；
激活函数为 SiLU（Swish）：平滑非线性，缓解梯度消失；
中间维度自动计算：hidden_size * 8/3 并对齐 64 的倍数，平衡表达能力和计算成本。


MoE 扩展（可选）：
当 use_moe=True 时，FFN 替换为 MoEFeedForward：多个 FFN「专家」+ 门控网络；
门控网络为每个 Token 选择 top-k 个（默认 2）路由专家处理，同时所有 Token 共享 1 个共享专家；
辅助损失：通过平衡专家负载，避免部分专家被过度使用 / 闲置。



3. 输出层（LM Head）

线性层（无偏置）：将最后一个 Decoder Block 的输出（维度 hidden_size）映射到词汇表大小（6400），输出每个 Token 的 logits；
损失计算：采用交叉熵损失，标签移位（预测第 i 个 Token 的下一个 Token），忽略 -100 标签（padding 部分）。

# 提问
## layerNorm细节
## RoPE细节
## RoPE长度外推
## attention计算细节
## FlashAttention
## kv cache是什么
## FFN的GLU细节
## MoE架构如何适配，需要改动什么