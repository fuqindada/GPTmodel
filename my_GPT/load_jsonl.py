import json
import os.path


def build_vocab(file_path, save_path):
    """
    构建词汇表。

    从指定的文件中读取问题和答案文本，提取所有唯一的字符，
    并为这些字符生成一个映射表，然后将该映射表保存到指定的路径。

    参数:
    file_path (str): 输入文件路径，文件应包含"question"和"answer"字段。
    save_path (str): 生成的词汇表文件保存路径。

    返回:
    无
    """
    # 读取所有文本
    texts = []
    with open(file_path, 'r', encoding='utf-8') as r:
        for line in r:
            if not line.strip():  # 检查行是否为空（去除首尾空白字符）
                continue
            line = json.loads(line)  # 解析 JSONL 行为字典
            question = line["question"]  # 获取问题文本
            answer = line["answer"]  # 获取答案文本
            texts.append(question)  # 将问题添加到文本列表
            texts.append(answer)  # 将答案添加到文本列表
    # 拆分 Token
    words = set()
    for t in texts:
        if not t:  # 检查文本是否为空
            continue
        for word in t.strip():  # 去除首尾空白字符并逐字符拆分
            words.add(word)  # 将每个字符添加到集合 `words`
    words = list(words)
    words.sort()  # 对字符进行排序
    # 特殊Token
    # pad 占位、unk 未知、eos 结束
    word2id = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    # 构建词表
    word2id.update({word: i + len(word2id) for i, word in enumerate(words)})
    id2word = list(word2id.keys())  # 创建 id2word 列表
    vocab = {"word2id": word2id, "id2word": id2word}  # 构建词汇表字典
    vocab = json.dumps(vocab, ensure_ascii=False)  # 将词汇表转换为 JSON 格式的字符串
    with open(save_path, 'w', encoding='utf-8') as w:
        w.write(vocab)  # 将词汇表写入文件
    print(f"finish. words: {len(id2word)}")  # 打印完成信息及词汇量大小


class Tokenizer():
    """
    Tokenizer类，用于处理文本数据，将其编码为模型可理解的格式，并能将模型的输出解码为人类可读的形式。
    """

    def __init__(self, vocab_path):
        """
        初始化Tokenizer类。

        从指定路径加载词汇表，词汇表是将单词映射到其相应ID的字典，以及将ID映射回单词的字典。

        参数:
        - vocab_path: 词汇表文件的路径。

        异常:
        - 如果词汇表为空，则抛出异常。
        """
        with open(vocab_path, "r", encoding="utf-8") as r:
            vocab=r.read()
            if not vocab:
                raise Exception("词表读取为空！")
        vocab = json.loads(vocab)
        self.word2id = vocab["word2id"]
        self.id2word = vocab["id2word"]
        self.pad_token = self.word2id["<pad>"]
        self.unk_token = self.word2id["<unk>"]
        self.eos_token = self.word2id["<eos>"]

    def encode(self, text, text1=None, max_length=140, pad_to_max_length=True):
        """
        编码文本为ID序列。

        参数:
        - text: 第一个文本序列。
        - text1: 可选的第二个文本序列。
        - max_length: 编码后序列的最大长度。
        - pad_to_max_length: 是否将序列填充到最大长度。

        返回:
        - tokens: 编码后的ID序列。
        - valid_mask: 有效标记掩码，用于指示哪些标记是有效的，哪些是填充。
        """
        tokens = [self.word2id[word] if word in self.word2id else self.unk_token for word in text]
        tokens.append(self.eos_token)
        if text1:
            tokens.extend([self.word2id[word] if word in self.word2id else self.unk_token for word in text1])
            tokens.append(self.eos_token)
        valid_mask = [1] * len(tokens)
        if pad_to_max_length:
            if len(tokens) > max_length:
                tokens = tokens[0:max_length]
                valid_mask = valid_mask[0:max_length]
            else:
                tokens.extend([self.pad_token] * (max_length - len(tokens)))
                valid_mask.extend([0] * (max_length - len(valid_mask)))
        return tokens, valid_mask

    def decode(self, token):
        """
        解码ID序列回文本。

        参数:
        - token: ID序列，可以是列表或单个ID。

        返回:
        - 解码后的文本字符串，如果输入是列表，则返回列表的解码结果。
        """
        if isinstance(token, (tuple, list)):
            return[self.decode(l) for l in token]
        else:
            return self.id2word[token]

    def __len__(self):
        """
        获取词汇表大小。

        返回:
        - 词汇表中单词的数量。
        """
        return len(self.id2word)


def split_dataset(file_path, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    datas = []
    with open("../data/extend.jsonl", "r", encoding='utf-8') as f:
        for line in f:
            if not line or line == "":
                continue
            datas.append(line)
    datas.append("\n")
    with open(file_path, "r", encoding='utf-8') as f:
        for line in f:
            if not line or line == "":
                continue
            datas.append(line)
    train = datas[:-1000]
    val = datas[-1000:]
    with open(os.path.join(output_path, "train.json"), "w", encoding="utf-8") as w:
        for line in train:
            w.write(line)
            w.flush()
    with open(os.path.join(output_path, "val.json"), "w", encoding="utf-8") as w:
        for line in val:
            w.write(line)
            w.flush()
    print("train count: ", len(train))
    print("val count: ", len(val))


if __name__ == '__main__':
    file_path = '../data/train.jsonl'
    save_path = 'model_train_data/vocab.json'
    # build_vocab(file_path, save_path)
    split_dataset(file_path=file_path, output_path = "model_train_data")
