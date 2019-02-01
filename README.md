### 概述

transformer模型是由Google团队在论文:“ [attention is all your need](https://arxiv.org/pdf/1706.03762.pdf)”中提出，该模型未采用传统的cnn、rnn结构，而是使用position embedding、multi-head self attention、Position-wise Feed-Forward等多种方法构建了一种新型的基于attention的模型。该模型在语言模型上也有了成功的应用，如[BERT](https://arxiv.org/pdf/1810.04805.pdf)。

该模型有以下优点：

* 并行化。抛弃了RNN的递归计算，计算速度快
* 长依赖问题。句子中针对每个词的计算是并行的，不会像RNN那样存在长依赖问题
* 能够充分捕捉词与词之间的关系。与RNN对比，每一层的都是全连接

该模型有以下缺点：

* 模型训练参数较多。如bert模型，参数上亿
* 解码速度慢
* 捉序列信息的能力不如RNN

因此，我建立了三种机器人聊天模型，分别使用了三种模型，用于探索transformer模型与传统的rnn模型之间的优劣。

### transformer-based-chatbot
[第一个模型](./transformer_base)
