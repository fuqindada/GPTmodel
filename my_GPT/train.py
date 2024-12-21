import torch
from tqdm import tqdm
import os

from load_jsonl import Tokenizer
from GPT_model import GPTModel
from QA_dataset import load_data
from get_params import GPTConfig

def train_model(net, train_loader, val_loader, optimizer, criterion, device, num_epochs, save_model_dir):
    """
    训练模型的主要函数。

    参数:
    - net: 模型网络结构。
    - train_loader: 训练数据加载器。
    - val_loader: 验证数据加载器。
    - optimizer: 优化器。
    - criterion: 损失函数。
    - device: 设备类型（CPU或GPU）。
    - num_epochs: 训练的轮数。
    - save_model_dir: 保存模型的目录。

    此函数执行模型的训练和验证过程，包括：
    1. 在训练集上训练模型。
    2. 在验证集上评估模型。
    3. 如果模型表现更好，则保存模型。
    """
    # 初始化最佳验证损失为无穷大
    best_value_loss = float("inf")

    # 开始训练过程
    for epoch in range(num_epochs):
        # 设置网络为训练模式
        net.train()

        # 遍历训练数据集
        for index, data in enumerate(tqdm(train_loader, desc='Train Epoch: ' + str(epoch))):
            # 将输入、标签和注意力掩码数据移动到指定设备
            inputs = data["input_ids"].to(device, dtype=torch.long)
            labels = data["labels"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)

            # 清除之前的梯度
            optimizer.zero_grad()

            # 前向传播
            outputs, dec_self_attn_weights = net(inputs, attention_mask)

            # 计算损失
            loss = criterion(outputs, labels.view(-1))

            # 反向传播
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)

            # 更新权重
            optimizer.step()

            # 在每个epoch的最后一个批次打印训练损失
            if index == len(train_loader) - 1:
                print('Epoch:', '%03d' % (epoch + 1), 'loss =', '{:.4f}'.format(loss))

        # 设置网络为评估模式
        net.eval()

        # 计算验证集损失
        val_loss = evaluate(net, val_loader, criterion, device)

        # 打印验证损失
        print('[Epoch: %d / %d] validation loss : %f' % (epoch + 1, num_epochs, val_loss))

        # 如果当前验证损失是最佳的，则保存模型
        if val_loss < best_value_loss:
            best_value_loss = val_loss
            save_model_path = os.path.join(save_model_dir, 'best_model.pth')
            torch.save(net.state_dict(), save_model_path)
            print(f"保存最优模型到{save_model_path}")

        save_model_path = os.path.join(save_model_dir, 'last_model.pth')
        torch.save(net.state_dict(), save_model_path)
        print(f"保存最新模型到{save_model_path}")


def evaluate(net, val_loader, criterion, device):
    """
    评估模型在验证集上的损失。

    参数:
    net: 模型网络，用于前向传播。
    val_loader: 验证集的数据加载器，用于迭代获取数据。
    criterion: 损失函数，用于计算模型输出和真实标签之间的差异。
    device: 设备信息，指示模型和数据应加载到CPU还是GPU。

    返回:
    平均验证损失。
    """
    # 初始化验证损失为0
    val_loss = 0

    # 迭代验证集数据，计算验证损失
    for index, data in enumerate(tqdm(val_loader, desc='Evaluate')):
        # 将输入数据加载到指定设备
        inputs = data["input_ids"].to(device, dtype=torch.long)
        labels = data["labels"].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(device, dtype=torch.long)

        # 前向传播，获取模型输出和注意力权重
        outputs, dec_self_attn_weights = net(inputs, attention_mask)

        # 计算损失，并累加到总的验证损失中
        loss = criterion(outputs, labels.view(-1))
        val_loss += loss.item()

    # 返回平均验证损失
    return val_loss / len(val_loader)


def main():
    vocab_path = 'model_train_data/vocab.json'
    tokenizer = Tokenizer(vocab_path)
    epochs = 20
    learning_rate = 1e-4
    batch_size = 160 # 128占用大约10G显存, 160占用约13G显存
    train_loader, val_loader = load_data(batch_size)
    save_model_dir = "../my_GPT/model_output"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_prams = GPTConfig(tokenizer, device).get_config()

    model = GPTModel(**model_prams).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token).to(device)
    train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs, save_model_dir)


if __name__ == '__main__':
    main()
