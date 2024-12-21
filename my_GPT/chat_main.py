import torch
import numpy as np

from load_jsonl import Tokenizer
from GPT_model import GPTModel
from get_params import GPTConfig

def generate(net, tokenizer, text, max_tokens, device):
    """
    生成文本的函数。

    参数:
    - net (GPTModel): 训练好的GPT模型。
    - tokenizer (Tokenizer): 用于文本编码和解码的分词器。
    - text (str): 输入的文本。
    - max_length (int): 生成文本的最大长度。
    - device (torch.device): 模型运行的设备，如CPU或GPU。

    返回:
    - generated_text (str): 生成的文本。
    """
    net.eval()  # 将模型设置为评估模式
    input_ids, attention_mask = tokenizer.encode(text)  # 对输入文本进行编码
    inputs = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # 将编码后的输入转换为模型所需的格式
    generated_ids = []  # 用于存储生成的token

    # 获取实际输入的长度，排除填充符号
    def get_actual_input_length(input_ids, pad_token):
        if not input_ids or pad_token is None:
            return 0
        non_padding_mask = np.array(input_ids) != pad_token
        actual_input_length = np.sum(non_padding_mask)
        return actual_input_length

    if isinstance(inputs, torch.Tensor):
        actual_input_length = get_actual_input_length(input_ids, tokenizer.pad_token)
        inputs = inputs[:, :actual_input_length]
    else:
        raise ValueError("inputs 必须是 numpy 数组或 torch 张量")


    while len(generated_ids) < max_tokens:
        with torch.no_grad():  # 禁用梯度计算，节省内存
            output, _ = net(inputs)  # 通过模型获取输出
            prob = output.squeeze(0).max(dim=-1, keepdim=False)[1]
            next_word = prob.data[-1]

        if next_word == tokenizer.eos_token:
            break

        generated_ids.append(next_word.item())

        # 只保留实际输入的token，并拼接新的next_word
        inputs = torch.cat(
            [inputs, torch.tensor([[next_word]], dtype=inputs.dtype, device=device)], -1)

    # 解码生成的token序列
    generated_text = tokenizer.decode(generated_ids)
    return "".join(generated_text)


def main():
    # model_path = "../my_GPT/model_output/best_model.pth"
    model_path = "../my_GPT/model_output/last_model.pth"
    vocab_path = "model_train_data/vocab.json"
    tokenizer = Tokenizer(vocab_path)
    max_tokens = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_prams = GPTConfig(tokenizer, device).get_config()
    model = GPTModel(**model_prams).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    while True:
        text = input("(q退出)请输入：")
        if text == "q":
            break
        if text.strip() == "":
            continue
        res = generate(model, tokenizer, text, max_tokens, device)
        print("AI: ", res)


if __name__ == '__main__':
    main()
