from torch.utils.data import Dataset
import torch
import json
import numpy as np
from torch.utils.data import DataLoader

from load_jsonl import Tokenizer

class QADataset(Dataset):
    """
    QA数据集类，继承自torch.utils.data.Dataset。

    该类用于加载、预处理和提供问答任务所需的训练数据。
    """

    def __init__(self, data_path, tokenizer, max_length) -> None:
        """
        初始化QA数据集。

        参数:
        - data_path (str): 数据文件路径，用于加载数据。如果为None或空字符串，则不加载数据。
        - tokenizer (object): 用于文本编码的tokenizer对象。
        - max_length (int): 输入序列的最大长度。
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        if data_path:
            with open(data_path, "r", encoding='utf-8') as f:
                for line in f:
                    if not line or line == "":
                        continue
                    json_line = json.loads(line)
                    question = json_line["question"]
                    answer = json_line["answer"]
                    self.data.append({
                        "question": question,
                        "answer": answer
                    })
        print("model_train_data load ， size：", len(self.data))

    def preprocess(self, question, answer):
        """
        预处理单个问题和答案对。

        参数:
        - question (str): 问题文本。
        - answer (str): 答案文本。

        返回:
        - input_ids (list): 编码后的输入序列ID列表。
        - att_mask (list): 注意力掩码列表。
        - labels (list): 用于语言模型的标签序列。
        """
        encode, att_mask = self.tokenizer.encode(question, answer, max_length=self.max_length, pad_to_max_length=True)
        input_ids = encode[:-1]
        att_mask = att_mask[:-1]
        labels = encode[1:]
        return input_ids, att_mask, labels

    def __getitem__(self, index):
        """
        根据索引获取数据项。

        参数:
        - index (int): 数据项索引。

        返回:
        - dict: 包含输入ID、注意力掩码和标签的字典。
        """
        item_data = self.data[index]
        input_ids, att_mask, labels = self.preprocess(**item_data)
        return {
            "input_ids": torch.LongTensor(np.array(input_ids)),
            "attention_mask": torch.LongTensor(np.array(att_mask)),
            "labels": torch.LongTensor(np.array(labels))
        }

    def __len__(self):
        """
        获取数据集大小。

        返回:
        - int: 数据集大小。
        """
        return len(self.data)


def load_data(batch_size, max_length=140, num_workers=4):
    train_json_path = "model_train_data/train.json"
    val_json_path = "model_train_data/val.json"
    tokenizer = Tokenizer("model_train_data/vocab.json")
    print("Start Load Train Data...")
    train_params = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
    }
    training_set = QADataset(train_json_path, tokenizer, max_length)
    training_loader = DataLoader(training_set,  **train_params)
    print("Start Load Validation Data...")
    val_params = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
    }
    val_set = QADataset(val_json_path, tokenizer, max_length)
    val_loader = DataLoader(val_set,  **val_params)
    return training_loader, val_loader


if __name__ == '__main__':
    train_loader, val_loader = load_data(2)
    print("train_loader:", next(iter(train_loader)))
