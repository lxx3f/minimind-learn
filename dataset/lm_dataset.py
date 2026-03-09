from torch.utils.data import Dataset # pytorch数据集基类
import torch
import os
import random
from datasets import load_dataset
# 关闭tokenizaiton并行
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 因为外层的dataloader已经有多进程加速了，这里再开多线程反而降低效率

def pre_processing_chat(conversations, add_system_ratio=0.2):
    """
    数据预处理
    在对话前随机添加 system prompt
    用于增强训练多样性
    """
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    if conversations and conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    """
    数据后处理，删除大部分空的\<think\>标签
    """
    # 为什么会有空think标签？很多训练数据是从不同来源拼接的：有推理数据（带思维链）有普通对话数据（没有思维链）
    # 为了统一格式，可能强制包一层：<think>\n\n{reasoning}</think>\n\n

    # 为什么保留部分空think标签？保留部分结构一致性，让模型理解：think是可选的。
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        # data_path: json数据路径
        # tokenizer: 分词器
        # max_length: 最大序列长度
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        # 样本数
        return len(self.samples)

    def __getitem__(self, index):
        # 读取一个样本
        sample = self.samples[index]
        # 为什么max_length=self.max_length - 2?
        # 因为要为后面增加的[begin]和[end]预留空间
        # truncation:当 token 数超过 max_length 时自动截断
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        # bos:beginning of sentence;  eos:end of sentence
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # 填充padding
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        # 数据类型转换为tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # 自回归语言模型：labels和输入序列完全一致，模型根据输入序列预测下一个token，目标是让输出和输入序列本身对齐
        labels = input_ids.clone()
        # 为什么要把 padding 位置的 labels 赋值为 -100？
        # pytorch特性：CrossEntropyLoss会自动忽略标签值为 -100 的位置，不计算这些位置的损失
        # padding的token是为了让批次内的序列长度统一而添加的 “无效内容”，不应该参与损失计算，
        # 否则会误导模型学习无意义的 padding token，导致训练效果变差。
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        """
        初始化数据集
        :param jsonl_path: 训练数据的jsonl文件路径（每行一个样本）
        :param tokenizer: 使用的tokenizer
        :param max_length: 序列最大长度，超出则截断、不足则padding
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # split用于指定要加载的数据集拆分（子集），比如训练集、验证集、测试集等
        # 这里的split='train'表示：只加载数据集的 train（训练集）拆分。
        # 注：当用 data_files 指定本地文件时，datasets 库会默认把整个文件标记为 train 拆分（无论文件内容是什么）
        # 所以split='train'的实际效果：加载这个 JSONL 文件的全部内容，并标记为训练集；
        # 如果省略 split 参数：load_dataset 会返回一个 DatasetDict 对象（字典结构），键是拆分名称（如 train），值是对应数据集。
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')

        # 预计算「assistant回复开头标识」的token id
        # add_special_tokens=False：不自动添加bos/eos等特殊token，仅编码字面内容
        # 注意这里的bos_id其实是一个列表，即这一段开头对应的若干token_id （所以实际上命名加一个s会好理解一点）
        # 为什么要计算这个？因为这些部分token是不需要学习的，所以对应位置的label值要置-100
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        # 预计算「回复结束标识」的token id，目的同上
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        """
        必须实现的方法：返回数据集总样本数
        Dataset类要求实现__len__，用于DataLoader计算批次数量
        """
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """
        构造标准化的对话提示词（prompt）
        :param conversations: 单条样本的对话列表（如[{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]）
        :return: 格式化后的文本prompt（未分词）
        """
        # 复制对话列表，避免修改原数据（浅拷贝，防止后续操作污染原始样本）
        messages = conversations.copy()
        # 处理工具调用场景：如果第一条是system角色且包含functions字段，提取工具定义
        # 逻辑拆解：
        # 1. conversations非空 + 第一条是system角色 + 包含functions字段 → 提取tools
        # 2. 否则tools为None（无工具调用）
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False, # 只返回文本，不进行分词（后续手动分词）
            add_generation_prompt=False, # 不添加"生成提示"（如assistant开头的占位符）
            tools=tools  # 传入工具定义（如有），适配工具调用场景的模板
        )

    def generate_labels(self, input_ids):
        """
        生成训练用的labels：仅让模型学习assistant的回复部分（其他部分标为-100，不计算损失）
        :param input_ids: 整段prompt的token id序列
        :return: labels序列（非assistant回复部分为-100，回复部分与input_ids一致）
        """
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        """
        必须实现的方法：根据索引获取单条样本的input_ids和labels
        :param index: 样本索引
        :return: (input_ids_tensor, labels_tensor) 均为long型tensor
        """
        sample = self.samples[index]
        # 预处理对话数据,在对话前随机添加 system prompt 用于增强训练数据多样性
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        # 后处理prompt,删除大部分空的<think>标签
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        # 生成labels
        labels = self.generate_labels(input_ids)
        # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        # 将input_ids和labels转换为PyTorch tensor（long型，符合模型输入要求）
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    """
    DPO (Direct Preference Optimization) 数据集类
    用于处理偏好对齐训练数据，加载并格式化chosen/rejected样本对
    """
    def __init__(self, file_path, tokenizer, max_length=4096):
        """
        初始化数据集
        Args:
            file_path: 数据集文件路径（JSON格式）
            tokenizer: 分词器（如HuggingFace的Tokenizer）
            max_length: 文本最大长度限制
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 设置padding的token id：优先使用分词器的pad_token_id，无则默认0
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # 同上SFT例
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # 提取优选回复（chosen）
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        # 将chosen对话结构转换为标准化文本
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        # 对生成的文本进行后处理
        chosen_prompt = post_processing_chat(chosen_prompt)

        # 对rejected执行相同的文本模板转换和后处理
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)

        # 对chosen文本进行分词编码：截断过长文本、填充到max_length
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        # 对rejected文本执行相同的分词编码
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        # 生成chosen的损失掩码（标记哪些token需要计算损失）
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        # 生成rejected的损失掩码
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content']
        prompt = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True  # 这里需要True
        )
        prompt = post_processing_chat(prompt)
        return prompt, answer

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt, answer = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': answer
        }

if __name__ == "__main__":
    pass