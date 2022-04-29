import os
import time
import torch
from torch import nn as nn
import torch.nn.functional as F
from bilstm_model import TextBiLSTM
from gensim.models import keyedvectors
from dataProcess.processing import abs_vocab
from dataProcess.processing import def_vocab
from dataProcess.processing import abs_train_iter, def_train_iter
from dataProcess.processing import abs_test_iter, def_test_iter

"""
判定是否是可以使用GPU完成计算
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 词嵌入维度
embed_size = 300
# 隐藏层神经元数量
num_hidden = 100
# 隐藏层数量
num_layers = 2
# 调用LSTM模型
bi_lstm = TextBiLSTM(abs_vocab, def_vocab, embed_size, num_hidden, num_layers)


"""
从预训练好的vocab中提取出words对应的词向量
"""


def load_pretrained_embedding(words, pretrained_vocab):
    # 初始化为0
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0])
    # out of vocabulary
    oov_count = 0
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.get_index(word)
            embed[i, :] = torch.from_numpy(pretrained_vocab.vectors)[idx]
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        print("There are %d oov words." % oov_count)
    return embed


"""
模型评价函数，用来测试每一轮测试数据下的准确率
abs_test_data：摘要测试数据
def_test_data：术语测试数据
model_info：模型数据
"""


def evaluate_accuracy(abs_test_data, def_test_data, model_info, support_device=None):
    # 判定是否使用加速设备，如果没有指定设备，则使用默认的模型设备
    if support_device is None and isinstance(model_info, torch.nn.Module):
        support_device = list(model_info.parameters())[0].device

    # 准确率初始化
    acc_num = 0
    n = 0
    with torch.no_grad():
        for i, data in enumerate(zip(abs_test_data, def_test_data)):
            if isinstance(model_info, torch.nn.Module):
                # 启动评估模式
                model_info.eval()
                # 测试数据的标签
                test_label = data[0][1]
                # 根据当前训练轮数的训练参数来使用测试集对模型效果进行拟合的结果
                test_hat = model_info(data[0][0].to(support_device), data[1][0].to(support_device))
                # 准确率求和
                acc_num += (test_hat.argmax(dim=1) == test_label).sum().cpu().item()
                model_info.train()
            n += data[0][1].shape[0]
    return acc_num / n


"""
模型训练函数
abs_train_data：摘要训练数据
abs_test_data：摘要测试数据
def_train_data：术语训练数据
def_test_data：术语测试数据
model_info：模型数据
loss_def：损失函数
optimizer_info：学习率
support_device：支持的服务
epochs：训练轮次
"""


def train(abs_train_data, abs_test_data, def_train_data, def_test_data, model_info, loss_def, optimizer_info, support_device, epochs):
    # pytorch的to()将张量中的数据送入GPU（如果支持CUDA的话）
    model_info = model_info.to(support_device)
    print("training on ", device)
    # batch计数变量
    batch_count = 0
    for epoch in range(epochs):
        # train_l_sum:训练损失率
        train_l_sum = 0.0
        # train_acc_sum:训练准确率
        train_acc_sum = 0.0
        # n:训练批次
        n = 0
        # start:开始时间
        start = time.time()
        for i, data in enumerate(zip(abs_train_data, def_train_data)):
            abs_x = data[0][0].to(device)
            def_x = data[1][0].to(device)
            y_label = data[1][1].to(device)
            # 梯度初始化0
            optimizer_info.zero_grad()
            # y_hat：模型拟合结果
            y_hat = model_info(abs_x, def_x)
            # y_test 拟合结果中的label
            y_test = y_hat[:, 0]
            # loss_：损失率
            # loss_ = loss_def(y_test.to(torch.float32), y_label.to(torch.float32))
            # loss_ = loss_def(y_test, y_label)
            loss_ = F.binary_cross_entropy(y_test.to(torch.float32), y_label.to(torch.float32))
            # 损失率反向传播
            loss_.backward()
            # 学习率不断更新
            optimizer_info.step()
            # 损失率求和
            train_l_sum += loss_.cpu().item()
            train_acc_sum += float(torch.sum(torch.argmax(y_test, dim=0) == y_label))
            # 训练批次迭代
            n += y_label.shape[0]
            batch_count += 1
        # 测试集上准确率的评估
        test_acc = evaluate_accuracy(abs_test_data, def_test_data, model_info)
        # 每一次epoch的结果
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f,\
                        time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
                 test_acc, time.time() - start))


# dic_abs, dic_def是预训练好的词向量
# dic_abs = keyedvectors.load_word2vec_format('../DataSet/model/abstract.bin', binary=True)
# dic_def = keyedvectors.load_word2vec_format('../DataSet/model/definition.bin', binary=True)
dic_abs = keyedvectors.load_word2vec_format('../DataSet/sgns.baidubaike.bigram-char', binary=False)

# 词嵌入
bi_lstm.embedding1.weight.data.copy_(load_pretrained_embedding(abs_vocab.get_itos(), dic_abs))
bi_lstm.embedding2.weight.data.copy_(load_pretrained_embedding(def_vocab.get_itos(), dic_abs))

# 直接加载预训练好的, 所以不需要更新它
bi_lstm.embedding1.weight.requires_grad = True
bi_lstm.embedding2.weight.requires_grad = True
# 学习率
learn_rate = 0.01
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, bi_lstm.parameters()), lr=learn_rate)
# epochs
num_epochs = 50

# 损失函数
# loss = nn.MSELoss()
# loss = nn.CrossEntropyLoss()
loss = nn.NLLLoss()

# 模型训练
train(abs_train_iter, abs_test_iter, def_train_iter, def_test_iter, bi_lstm, loss, optimizer, device, num_epochs)

# 模型保存
torch.save(bi_lstm, '../modelData/BiLSTM.pkl')



