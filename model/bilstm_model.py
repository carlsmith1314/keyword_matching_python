import torch
import torch.nn as nn


class TextBiLSTM(nn.Module):
    # 构造函数的参数
    # input_size: 输入数据x的特征值的数目
    # hidden_size: 隐藏层的神经元数量，也就是隐藏层的特征数量
    # num_layer: 循环神经网络的层数，默认是1
    # bias: 默认为True, 如果是FALSE。则表示神经元不使用该参数
    # batch_first: 如果设置为TRUE，则输入数据的维度中第一个维度就是batch值，默认为FALSE，默认状态下第一个维度是序列的长度，第二个才是，第三个是特征数目
    # dropout: 抛弃数据的比例
    # 词嵌入: 用不同的特征来对各个词汇进行表征，不同的单词均有不同的值
    # pytorch中，使用nn.Embedding层来做嵌入词袋模型，Embedding层第一个输入表示我们有多少词，第二个输入表示每一个词使用多少维度的向量表示

    def __init__(self, vocab1, vocab2, embed_size, num_hidden, num_layer):
        super(TextBiLSTM, self).__init__()
        # 嵌入层
        self.embedding1 = nn.Embedding(len(vocab1), embed_size)
        self.embedding2 = nn.Embedding(len(vocab2), embed_size)
        # 隐藏层
        # 处理文献摘要
        self.layer1 = nn.LSTM(
            input_size=embed_size,
            hidden_size=num_hidden,
            num_layers=num_layer,
            bidirectional=True
        )
        # 处理关键术语定义
        self.layer2 = nn.LSTM(
            input_size=embed_size,
            hidden_size=num_hidden,
            num_layers=num_layer,
            bidirectional=True
        )
        # 全连接层
        # Batch：批处理，顾名思义就是对某对象进行批量的处理。训练神经网络时，在数据集很大的情况下，不能一次性载入全部的数据进行训练，电脑会支撑不住，其次全样本训练对于非凸损失函数会出现局部最优，所以要将大的数据集分割进行分批处理。
        # batch_size就是每批处理的样本的个数。
        # nn.Linear()：用于设置网络中的全连接层。
        # in_features指的是输入的二维张量的大小，即输入的[batch_size, size]中的size。
        # out_features指的是输出的二维张量的大小，即输出的二维张量的形状为[batch_size，output_size]，当然，它也代表了该全连接层的神经元个数。
        # 从输入输出的张量的shape角度来理解，相当于一个输入为[batch_size, in_features]的张量变换成了[batch_size, out_features]的输出张量。
        # 如何连接全连接层？？？
        self.layer3 = nn.Linear(4 * num_hidden, 2)

    # 2022.4.1 此部分修改内容如下
    # 根据main函数中出现了参数传递数量的问题，问题如下
    # forward() takes 2 positional arguments but 3 were given
    # 故修改内容如下，将forward()的参数进行修改
    def forward(self, inputs1, inputs2):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列长度(seq_len)作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        # permute 张量的维度换位，简单来说就是交换位置
        # 序列长度是什么？？？
        # 该部分还需要后续修改，词嵌入部分尚不完善
        # embeddings = self.embedding(inputs.permute(1, 0))
        embeddings1 = self.embedding1(inputs1.permute(1, 0))
        embeddings2 = self.embedding2(inputs2.permute(1, 0))
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # output形状是(词数, 批量大小, 2 * 隐藏单元个数)
        # output, (h, c)
        # 此部分输出结果仍然存在怀疑？？？？？？？？
        output1, _1 = self.layer1(embeddings1)
        output2, _2 = self.layer2(embeddings2)
        # 连结文献摘要的隐藏状态和文献关键词的隐藏状态作为全连接层输入。它的形状为？？？
        # torch.cat((A,B),axis)是对A, B两个tensor进行拼接。
        encoding = torch.cat((output1[-1], output2[-1]), -1)
        outs = torch.sigmoid(self.layer3(encoding))
        return outs
