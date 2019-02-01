基于https://github.com/tensorflow/models/tree/master/official/transformer
论文"attention is all your need" https://arxiv.org/abs/1706.03762
 
1.train
python3 ./train.py
准备数据，格式见raw_data/test.checked
修改的参数（config.py）
raw_data_dir 训练数据所在的文件夹
source_language_type 翻译的输入语言
target_language_type 翻译的目标语言

max_source_length 输入句子的最大长度 
max_target_length 目标句子的最大长度

num_epochs 训练循环次数

若GPU不足，适当调整以下参数(hidden_size 要能够整除num_heads):
batch_size
hidden_size 
num_hidden_layers
num_heads 
filter_size

2.release
python3 release.py

3.test
python3 test.py

4. aan
原始的tansformer模型支持并行化计算，训练速度较快。但是decoder过程速度较慢。
因此，参考论文 Accelerating Neural Transformer via an Average Attention Network
将decoder的self attention层替换为论文中提出的 Average Attention Layer,在准确率影响较小的情况下提升decoder速度。
原始trasformer的复杂度:n*n*d+n*d*d
ann:n*d*d
n:句子长度，d：hidden_size
句子越长，优势越明显

