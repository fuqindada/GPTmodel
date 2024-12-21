
class GPTConfig:
    """
    GPT模型的配置类

    该类用于初始化和存储GPT模型的各种配置参数，包括隐藏层单元数、头数、前馈网络隐藏单元数、
    键大小、值大小、词汇表大小、最大位置数、层数以及设备信息。

    参数:
    - tokenizer: 用于获取词汇表大小的tokenizer对象
    - device: 模型运行的设备，如CPU或GPU
    """
    def __init__(self, tokenizer, device):
        # 隐藏层单元数
        self.num_hiddens = 768
        # 头数
        self.num_heads = 8
        # 前馈网络隐藏单元数
        self.ffn_hiddens = 2048
        # 键大小
        self.key_size = 64
        # 值大小
        self.value_size = 64
        # 词汇表大小
        self.vocab_size = len(tokenizer)
        # 最大位置数
        self.max_position = 1800
        # 层数
        self.num_layers = 6
        # 设备
        self.device = device

    def get_config(self):
        """
        获取模型配置信息

        返回一个包含模型所有配置参数的字典，包括隐藏层单元数、头数、前馈网络隐藏单元数、
        键大小、值大小、词汇表大小、最大位置数、层数以及设备信息。
        """
        return {
            "num_hiddens": self.num_hiddens,
            "num_heads": self.num_heads,
            "ffn_hiddens": self.ffn_hiddens,
            "key_size": self.key_size,
            "value_size": self.value_size,
            "vocab_size": self.vocab_size,
            "max_position": self.max_position,
            "num_layers": self.num_layers,
            "device": self.device
        }
