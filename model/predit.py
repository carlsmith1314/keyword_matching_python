import torch
from dataProcess.processing import abs_vocab
from dataProcess.processing import def_vocab


# 预测分类结果
def predict_sentiment(net, abs_vocab_p, def_vocab_p, sentence1, sentence2):
    """sentence是词语的列表"""
    support_device = list(net.parameters())[0].device
    se1 = torch.tensor([abs_vocab_p.__getitem__(word1) for word1 in sentence1], device=support_device)
    se2 = torch.tensor([def_vocab_p.__getitem__(word2) for word2 in sentence2], device=support_device)
    label = torch.argmax(net(se1.view((1, -1)), se2.view((1, -1))), dim=1)
    # 此处修改的返回值要和java部分保持一致
    return '1' if label.item() == 1 else '0'


# 数据处理部分
# abs_info = open('../DataSet/abs_train/true/0.txt', 'rb')
# def_info = open('../DataSet/def_train/true/0.txt', 'rb')
# abs_data = []
# def_data = []
# for abs_tok in abs_info.read().decode('utf-8').split(' '):
# abs_data.append(abs_tok)

# for def_tok in def_info.read().decode('utf-8').split(' '):
# def_data.append(def_tok)

test = torch.load('../modelData/LSTM.pkl')


def result(d1, d2):
    return predict_sentiment(test, abs_vocab, def_vocab, d1, d2)
