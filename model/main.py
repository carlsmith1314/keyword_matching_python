import os
import time
import torch
from torch import nn as nn
from lstm_model import TextLSTM
from gensim.models import keyedvectors
from dataProcess.processing import abs_vocab
from dataProcess.processing import def_vocab
from dataProcess.processing import abs_train_iter, def_train_iter
from dataProcess.processing import abs_test_iter, def_test_iter

# 判定是否能用GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置词嵌入维度、隐藏层神经元数量、隐藏层数量
# 以上参数如何设置????
embed_size, num_hidden, num_layers = 100, 100, 2
net_model = TextLSTM(abs_vocab, def_vocab, embed_size, num_hidden, num_layers)


# 从预训练好的vocab中提取出words对应的词向量
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


# dic_abs, dic_def是预训练好的词向量
dic_abs = keyedvectors.load_word2vec_format('../DataSet/model/abstract.bin', binary=True)
dic_def = keyedvectors.load_word2vec_format('../DataSet/model/definition.bin', binary=True)

# 此处仍然存在问题，关于word2vec版本的问题，后续只需修改函数即可,dic_abs.itos等作为数据是否正确仍然需要商榷
net_model.embedding1.weight.data.copy_(load_pretrained_embedding(abs_vocab.get_itos(), dic_abs))
net_model.embedding2.weight.data.copy_(load_pretrained_embedding(def_vocab.get_itos(), dic_def))

# 直接加载预训练好的, 所以不需要更新它
net_model.embedding1.weight.requires_grad = False
net_model.embedding2.weight.requires_grad = False


# 模型评价
# 模型评价部分仍需要进行修改和完善，此部分仅为单变量代码！！！！！？？？？？？
def evaluate_accuracy(abs_data_iter, def_data_iter, net, support_device=None):
    # 判定是否指定加速设备，如果没指定device就使用net的device
    if support_device is None and isinstance(net, torch.nn.Module):
        support_device = list(net.parameters())[0].device
    # 初始化相关数据
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for i, data in enumerate(zip(abs_data_iter, def_data_iter)):
            # 用isinstance()判断一个变量是否是某个类型
            if isinstance(net, torch.nn.Module):
                net.eval()
                # mid_data = net(data[0][0].to(support_device), data[1][0].to(support_device))[:, 0]
                # 准确率评价部分存在问题？？？？？也就是下边这行
                mid_data = net(data[0][0].to(support_device), data[1][0].to(support_device))
                # mid_data = torch.transpose(mid_data, dim0=1, dim1=0)
                print(type(mid_data))
                print(type(data[0][1]))
                acc_sum += (mid_data.argmax(dim=1) == data[0][1].to(support_device)).float().sum().cpu().item()  # .__float__().sum().cpu().item()
                net.train()
            else:
                # 自定义模型
                if 'is_training' in net.__code__.co_varnames:
                    # 将is_training设置成False
                    acc_sum += (net(data[0][0], data[1][0], is_trainning=False).argmax(dim=1) ==
                                data[0][1]).__float__().sum().item()
                else:
                    acc_sum += (net(data[0][0], data[1][0], is_trainning=False).argmax(dim=1) ==
                                data[0][1]).__float__().sum().item()
            n += data[0][1].shape[0]
    return acc_sum / n


# 模型训练
# abs_train_iter 迭代后的文献摘要训练数据
# abs_test_iter 迭代后的文献摘要测试数据
# def_train_iter 迭代后的标准术语训练数据
# def_test_iter 迭代后的标准术语测试数据
# net pytorch函数
# loss 损失率
# optimizer 优化器
# support_dev 是否CUDA服务
# num_epochs 数据训练轮次
def train(abs_train_iter, abs_test_iter, def_train_iter, def_test_iter, net, loss, optimizer, support_device,
          num_epochs):
    # pytorch的to()将张量中的数据送入GPU（如果支持CUDA的话）
    net = net.to(support_device)
    print("training on ", device)
    # batch计数变量
    batch_count = 0
    # 循环部分尚未完善
    for epoch in range(num_epochs):
        # train_l_sum:训练损失率
        # train_acc_sum:训练准确率
        # n:训练批次
        # start:开始时间
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for i, data in enumerate(zip(abs_train_iter, def_train_iter)):
            abs_x = data[0][0].to(device)
            abs_y = data[0][1].to(device)
            def_x = data[1][0].to(device)
            def_y = data[1][1].to(device)
            # y_hat究竟是什么？？？这里使用net还是上述定义的model
            y_hat = net(abs_x, def_x)
            y_test = y_hat[:, 0]
            loss_ = loss(y_test, abs_y)
            # 梯度初始化0
            optimizer.zero_grad()
            # 损失率求和
            train_l_sum += loss_.cpu().item()
            # 准确率求和
            train_acc_sum += (y_hat.argmax(dim=1) == def_y).sum().cpu().item()
            n += def_y.shape[0]
            # 训练批次迭代
            batch_count += 1
        test_acc = evaluate_accuracy(abs_test_iter, def_test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f,\
                 time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
                 test_acc, time.time() - start))


lr, num_epochs = 0.01, 5
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net_model.parameters()), lr=lr)
# loss = nn.BCELoss()
loss = nn.MSELoss()
train(abs_train_iter, abs_test_iter, def_train_iter, def_test_iter, net_model, loss, optimizer, device, num_epochs)
torch.save(net_model, '../modelData/LSTM.pkl')

"""
# 预测数据处理部分
abs_info = open('../DataSet/abs_train/true/0.txt', 'rb')
def_info = open('../DataSet/def_train/true/0.txt', 'rb')
abs_data = []
def_data = []
for abs_tok in abs_info.read().decode('utf-8').split(' '):
    abs_data.append(abs_tok)

for def_tok in def_info.read().decode('utf-8').split(' '):
    def_data.append(def_tok)

# 预测分类结果
def predict_sentiment(net, abs_vocab_p, def_vocab_p, sentence1, sentence2):
    #sentence是词语的列表
    support_device = list(net.parameters())[0].device
    se1 = torch.tensor([abs_vocab_p.__getitem__(word1) for word1 in sentence1], device=support_device)
    se2 = torch.tensor([def_vocab_p.__getitem__(word2) for word2 in sentence2], device=support_device)
    label = torch.argmax(net(se1.view((1, -1)), se2.view((1, -1))), dim=1)
    return '是' if label.item() == 1 else '否'
    
# 模型预测实例部分
def main():
    print(predict_sentiment(net_model, abs_vocab, def_vocab, abs_data, def_data))


if __name__ == "__main__":
    main()
"""
