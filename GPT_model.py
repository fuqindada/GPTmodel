import torch
from torch import nn
import numpy as np

from my_GPT.get_params import GPTConfig
from my_GPT.load_jsonl import Tokenizer


class DotProductAttention(nn.Module):
    """
    实现点积注意力机制的类。

    点积注意力机制是一种用于计算注意力权重的方法，常用于自然语言处理中的Transformer模型。
    它通过计算查询向量和键向量之间的点积，然后应用Softmax函数来获取注意力权重，最后将这些权重应用于值向量。
    """

    def __init__(self, d_k):
        """
        初始化点积注意力机制。

        :param d_k: 键向量的维度，用于缩放注意力分数。
        """
        super(DotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, queries, keys, values, attention_mask):
        """
        执行点积注意力机制的前向传播。

        :param queries: 查询向量，形状为(batch_size, n_heads, seq_len, d_k)。
        :param keys: 键向量，形状为(batch_size, n_heads, seq_len, d_k)。
        :param values: 值向量，形状为(batch_size, n_heads, seq_len, d_v)。
        :param attention_mask: 注意力掩码，用于指定哪些注意力分数应被忽略，形状为(batch_size, n_heads, seq_len, seq_len)。
        :return: 应用了注意力权重的值向量和注意力权重，形状分别为(batch_size, n_heads, seq_len, d_v)和(batch_size, n_heads, seq_len, seq_len)。
        """
        # 计算注意力分数，通过点积操作获得，然后除以键向量维度的平方根进行缩放。
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2)) / np.sqrt(self.d_k)

        # 使用注意力掩码，将掩码位置的注意力分数设置为极小值，以避免这些位置在Softmax后获得较大的权重。
        attention_scores.masked_fill_(attention_mask, -1e9)

        # 应用Softmax函数，将注意力分数转换为注意力权重。
        attention_weights = nn.Softmax(dim=-1)(attention_scores)

        # 将注意力权重应用于值向量，得到加权后的值向量。
        return torch.matmul(attention_weights, values), attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制类。

    关键参数:
    - key_size (int): 关键向量的维度。
    - value_size (int): 价值向量的维度。
    - num_hiddens (int): 隐藏层的维度。
    - num_heads (int): 注意力头的数量。
    - bias (bool): 是否使用偏差项，默认为False。
    """
    def __init__(self, key_size, value_size, num_hiddens, num_heads, bias=False):
        super(MultiHeadAttention, self).__init__()
        # 初始化参数
        self.key_size = key_size
        self.value_size = value_size
        self.num_hiddens = num_hiddens
        self.num_heads = num_heads
        # 点积注意力机制
        self.attention = DotProductAttention(key_size)
        # 线性变换权重矩阵
        self.W_q = nn.Linear(num_hiddens, key_size * num_heads, bias=bias)
        self.W_k = nn.Linear(num_hiddens, key_size * num_heads, bias=bias)
        self.W_v = nn.Linear(num_hiddens, value_size * num_heads, bias=bias)
        self.W_o = nn.Linear(num_heads * value_size, num_hiddens, bias=bias)
        # 层归一化
        self.layernorm = nn.LayerNorm(num_hiddens)

    def forward(self, queries, keys, values, attention_mask):
        """
        前向传播函数。

        关键参数:
        - queries (Tensor): 查询张量，形状为(batch_size, 查询数量, num_hiddens)。
        - keys (Tensor): 键张量，形状为(batch_size, 键数量, num_hiddens)。
        - values (Tensor): 值张量，形状为(batch_size, 值数量, num_hiddens)。
        - attention_mask (Tensor): 注意力掩码张量，形状为(batch_size, 1, 键数量)。

        返回:
        - output (Tensor): 注意力机制输出张量，经过层归一化。
        - attention_weights (Tensor): 注意力权重张量。
        """
        # 残差连接和批量大小
        residual, batch_size = queries, queries.shape[0]
        # 对查询、键和值进行线性变换，并分拆成多个注意力头
        queries = self.W_q(queries).view(batch_size, -1, self.num_heads, self.key_size).transpose(1, 2)
        keys = self.W_k(keys).view(batch_size, -1, self.num_heads, self.key_size).transpose(1, 2)
        values = self.W_v(values).view(batch_size, -1, self.num_heads, self.value_size).transpose(1, 2)
        # 扩展注意力掩码以适应多头
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        # 计算注意力输出和权重
        output, attention_weights = self.attention(queries, keys, values, attention_mask)
        # 重新组合多个注意力头的输出
        output = output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.value_size)
        # 最终的线性变换
        output = self.W_o(output)
        # 残差连接和层归一化
        return self.layernorm(output + residual), attention_weights


class PositionWiseFFN(nn.Module):
    """
    实现位置感知前馈网络(Position-Wise Feed-Forward Network)。

    该网络结构包含两个全连接层，其中第一层将输入维度扩展到一个更大的维度，
    经过ReLU激活函数后，第二层再将维度缩小回原始维度。最后，使用层归一化(LayerNorm)
    来标准化输出与输入的残差和。

    参数:
    - num_hiddens (int): 输入和输出的特征维度。
    - ffn_hiddens (int): 前馈网络中间层的特征维度。
    """
    def __init__(self, num_hiddens, ffn_hiddens):
        super(PositionWiseFFN, self).__init__()
        # 第一个全连接层，将输入维度扩展到ffn_hiddens
        self.dense1 = nn.Linear(num_hiddens, ffn_hiddens)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # 第二个全连接层，将维度缩小回num_hiddens
        self.dense2 = nn.Linear(ffn_hiddens, num_hiddens)
        # 层归一化，用于标准化输出
        self.layernorm = nn.LayerNorm(num_hiddens)

    def forward(self, input):
        """
        前向传播函数。

        保存输入作为残差，通过两个全连接层和ReLU激活函数处理后，
        将输出与原始输入相加，并通过层归一化处理，最后返回结果。

        参数:
        - input (Tensor): 输入张量。

        返回:
        - Tensor: 经过位置感知前馈网络处理后的输出张量。
        """
        # 保存输入，用于残差连接
        residual = input
        # 第一个全连接层后接ReLU激活函数
        output = self.relu(self.dense1(input))
        # 第二个全连接层，将维度恢复至原始大小
        output = self.dense2(output)
        # 层归一化处理输出与输入的残差和，并返回
        return self.layernorm(output + residual)


class DecoderBlock(nn.Module):
    """
    解码器块类。

    该类用于构建解码器的一个基本块，包含多头注意力机制和位置前馈网络。

    参数:
    - key_size (int): 键的特征维度。
    - value_size (int): 值的特征维度。
    - num_hiddens (int): 隐藏层单元数。
    - num_heads (int): 注意力头的数量。
    - ffn_hiddens (int): 前馈神经网络的隐藏层大小。
    """
    def __init__(self, key_size, value_size, num_hiddens, num_heads, ffn_hiddens):
        super(DecoderBlock, self).__init__()
        # 初始化多头注意力机制
        self.attention = MultiHeadAttention(key_size, value_size, num_hiddens, num_heads)
        # 初始化位置前馈神经网络
        self.ffn = PositionWiseFFN(num_hiddens, ffn_hiddens)

    def forward(self, X, attention_mask):
        """
        前向传播方法。

        参数:
        - X (Tensor): 输入张量。
        - attention_mask (Tensor): 注意力掩码张量，用于避免在注意力计算中考虑某些元素。

        返回:
        - output (Tensor): 解码器块的输出张量。
        - attention_weights (Tensor): 注意力权重张量。
        """
        # 使用多头注意力机制处理输入张量
        output, attention_weights = self.attention(X, X, X, attention_mask)
        # 使用位置前馈神经网络进一步处理输出
        output = self.ffn(output)
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """
    实现位置编码功能。

    位置编码被添加到输入的嵌入向量中，以赋予模型关于序列中单词位置的信息。
    这对于Transformer模型尤其重要，因为自我注意力机制会丢失位置信息。

    参数:
    - num_hiddens: 嵌入向量的维度。
    - device: 设备信息，用于确定计算是在CPU还是GPU上进行。
    - max_len: 最大序列长度，默认为1000。
    """
    def __init__(self, num_hiddens, device, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.device = device
        # 初始化位置编码矩阵P，形状为(1, max_len, num_hiddens)
        self.P = torch.zeros((1, max_len, num_hiddens))
        # 计算位置编码，使用正弦和余弦函数来表示偶数和奇数维度的位置信息
        X = (torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) /
             torch.pow(10000,torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens))
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        """
        前向传播函数。

        将输入的嵌入向量X与位置编码P相加，然后返回结果。

        参数:
        - X: 输入的嵌入向量，形状为(batch_size, seq_len, num_hiddens)。

        返回:
        - X: 与位置编码相加后的嵌入向量。
        """
        # 将位置编码添加到输入X中，注意只取与X序列长度相等的部分
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return X


def get_attn_subsequence_mask(seq, device):
    """
    生成一个下三角矩阵，用于屏蔽掉未来的注意力。

    参数:
    - seq: 序列长度，用于确定矩阵的形状。

    返回:
    - mask: 下三角矩阵，形状为(batch_size, seq_len, seq_len)。
    """
    # 使用torch.triu生成上三角矩阵，参数diagonal=1表示对角线以上的元素全为1，其余为0
    # 这里生成的实际上是一个上三角矩阵，用于后续的注意力屏蔽
    mask = torch.triu(torch.ones(seq.shape[0], seq.shape[1], seq.shape[1]), diagonal=1)
    # 将mask转换为bool类型，方便在模型中使用
    mask = mask.byte()
    # 将mask移动到指定的设备上，如GPU
    return mask.to(device)


def get_attn_pad_mask(attention_mask, pad_label):
    """
    生成注意力填充掩码矩阵。

    该函数的目的是根据输入的注意力掩码，生成一个填充掩码矩阵，用于在后续的注意力计算中屏蔽掉填充部分。
    这对于保持注意力机制聚焦在非填充部分的输入非常重要。

    参数:
    attention_mask (Tensor): 输入的注意力掩码，形状为 [batch_size, seq_len]，
                             其中padding部分为0，非padding部分为1。
    pad_label (int): 用于标识padding部分的标签。

    返回:
    Tensor: 扩展后的注意力填充掩码矩阵，形状为 [batch_size, seq_len, seq_len]，
            用于屏蔽填充部分的注意力。
    """
    # 获取批次大小和序列长度
    batch_size, seq_len = attention_mask.shape

    # 创建一个布尔掩码，将padding部分标记为True，并在维度1上扩展，以匹配注意力矩阵的尺寸
    attention_mask = attention_mask.eq(pad_label).unsqueeze(1)

    # 扩展填充掩码，使其在序列维度上重复，以匹配注意力矩阵的尺寸
    return attention_mask.expand(batch_size, seq_len, seq_len)


class Decoder(nn.Module):
    """
    解码器类，用于Transformer模型中的解码部分。

    参数:
    - num_hiddens: 隐藏层维度
    - num_heads: 多头注意力机制中的头数
    - ffn_hiddens: 前馈神经网络的隐藏层维度
    - key_size: 关键向量的维度
    - value_size: 价值向量的维度
    - vocab_size: 词汇表大小
    - max_position: 最大位置编码
    - num_layers: 解码器块的层数
    - device: 设备信息，用于指示计算是在CPU还是GPU上执行

    返回:
    - output: 解码器的输出
    - self_attention_weights: 自注意力权重列表
    """
    def __init__(self, num_hiddens, num_heads, ffn_hiddens, key_size, value_size, vocab_size, max_position, num_layers, device):
        super(Decoder, self).__init__()
        self.pad_label = 0
        # 初始化设备信息
        self.device = device
        # 词汇嵌入层，将词汇表大小的索引映射到隐藏层维度的向量
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        # 位置编码层，为输入序列添加位置信息
        self.pos_encoding = PositionalEncoding(num_hiddens, device, max_position)
        # 解码器块的集合，包含多个解码器块
        self.blocks = nn.ModuleList([DecoderBlock(key_size, value_size, num_hiddens, num_heads, ffn_hiddens) for _ in range(num_layers)])

    def forward(self, X, attention_mask):
        """
        解码器的前向传播函数。

        参数:
        - X: 输入序列
        - attention_mask: 注意力掩码，用于防止注意力机制关注到不应该关注的部分

        返回:
        - output: 解码器的输出
        - self_attention_weights: 自注意力权重列表
        """
        # 通过词汇嵌入层
        output = self.embedding(X)
        # 添加位置编码
        output = self.pos_encoding(output)
        # 生成子序列掩码，用于解码过程中掩蔽未来时间步的信息
        subsequence_mask = get_attn_subsequence_mask(X, self.device)
        # 如果提供了注意力掩码，则合并子序列掩码和注意力掩码
        if attention_mask is not None:
            attention_mask = get_attn_pad_mask(attention_mask, pad_label=self.pad_label)
            attention_mask = torch.gt((attention_mask + subsequence_mask), 0)
        else:
            # 如果没有提供注意力掩码，则直接使用子序列掩码
            attention_mask = subsequence_mask.bool()
        # 初始化自注意力权重列表
        self_attention_weights = []
        # 通过每个解码器块
        for block in self.blocks:
            output, attention_weights = block(output, attention_mask)
            self_attention_weights.append(attention_weights)
        # 返回解码器的输出和自注意力权重列表
        return output, self_attention_weights


class GPTModel(nn.Module):
    """
    GPT模型类，继承自nn.Module。

    该模型主要用于自然语言处理任务，如文本生成等。它基于Transformer架构，包括一个解码器和一个投影层。

    参数:
    - num_hiddens (int): 隐藏层单元数。
    - num_heads (int): 多头注意力机制中的头数。
    - ffn_hiddens (int): 前馈神经网络中的隐藏层单元数。
    - key_size (int): 键向量的维度。
    - value_size (int): 值向量的维度。
    - vocab_size (int): 词汇表大小。
    - max_position (int): 最大位置编码。
    - num_layers (int): 解码器层的数量。
    - device (torch.device): 模型运行的设备，如CPU或GPU。
    """
    def __init__(self, num_hiddens, num_heads, ffn_hiddens, key_size, value_size, vocab_size, max_position, num_layers, device):
        super(GPTModel, self).__init__()
        # 初始化解码器，解码器是Transformer的重要组成部分，用于处理输入信息
        self.decoder = Decoder(num_hiddens, num_heads, ffn_hiddens, key_size, value_size, vocab_size, max_position, num_layers, device)
        # 初始化投影层，用于将解码器的输出映射到词汇表大小的维度，以便进行最终的词汇预测
        self.projection = nn.Linear(num_hiddens, vocab_size)

    def forward(self, X, attention_mask=None):
        """
        前向传播函数。

        参数:
        - X (torch.Tensor): 输入张量。
        - attention_mask (torch.Tensor, 可选): 注意力掩码，用于指定不应对哪些位置进行注意力计算。

        返回:
        - output (torch.Tensor): 模型输出，经过投影层后的张量，形状为(-1, vocab_size)。
        - self_attention_weights (torch.Tensor): 自注意力权重，用于可视化或进一步分析。
        """
        # 通过解码器处理输入，得到解码器输出和自注意力权重
        output, self_attention_weights = self.decoder(X, attention_mask)
        # 通过投影层处理解码器输出，得到最终的输出
        output = self.projection(output)
        # 调整输出形状，以便进行后续的处理或损失计算
        return output.view(-1, output.shape[-1]), self_attention_weights


def main():
    tokenizer = Tokenizer("model_train_data/vocab.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_params = GPTConfig(tokenizer, device).get_config()
    model = GPTModel(**model_params).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}") # 35762905


if __name__ == '__main__':
    main()