data/train.jsonl文件来自网站：
https://modelscope.cn/datasets/qiaojiedongfeng/qiaojiedongfeng
需要先下载数据集文件到data/train.jsonl
data/extend.jsonl为个人附加的一些模型自我信息

my_GPT/model_output储存模型训练的模型文件
my_GPT/model_train_data存放训练集、测试集以及词汇表
my_GPT/GPT_model.py为整个基于decoder_only的GPT模型代码
my_GPT/QA_dataset.py生成Question&Answer数据集，并且创建训练集生成器和验证集生成器
my_GPT/chat_main.py与训练好的模型进行对话
my_GPT/get_params.py存放模型的参数
my_GPT/load_jsonl.py加载train.jsonl数据集，生成词汇表，创建分词器。运行train.py前需要先在该代码中创建词汇表、划分训练集和测试集。
my_GPT/train.py训练模型的代码
