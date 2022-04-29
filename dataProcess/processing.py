import collections
import os
import random
import torch
import torchtext.vocab as Vocab
import torch.utils.data as Data
from tqdm import tqdm


# 定义函数读取DataSet
def read_imdb(folder='train', data_root="../DataSet"):
    data = []
    for label in ['true', 'false']:
        folder_name = os.path.join(data_root, folder, label)
        # 遍历文件夹下所有文件,tqdm为显示遍历进度的包
        for file in tqdm(os.listdir(folder_name)):
            with open(os.path.join(folder_name, file), 'rb') as f:
                # 将句子中的回车去除
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'true' else 0])
    random.shuffle(data)
    # 最终返回的data是一个List，里面存放了所有folder下的数据，每个数据还有这个句子的所有单词和标签
    return data


# 获取测试数据和训练数据
data_root = '../DataSet'
abs_train_data, abs_test_data = read_imdb('abs_train', data_root), read_imdb('abs_test', data_root)
def_train_data, def_test_data = read_imdb('def_train', data_root), read_imdb('def_test', data_root)


# 文本数据预处理
def get_tokenized_imdb(data):
    """
    data: list of [string, label]
    """

    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]

    # 获得按照空格分开后的所有词语
    return [tokenizer(review) for review, _ in data]


# 获取数据字典
def get_vocab_imdb(data):
    tokenized_data = get_tokenized_imdb(data)
    # counter是这个数据里所有单词的出现次数
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    # 返回一个vocab，获取counter里单词数量大于等于1的数据
    return Vocab.vocab(counter, min_freq=1)


abs_vocab = get_vocab_imdb(abs_train_data)
def_vocab = get_vocab_imdb(def_train_data)


# 文本数据向量化
def preprocess_imdb(data, vocab):
    max_l = 300

    # 定义评论补全函数
    # 将每条评论通过截断或者补0，使得长度变成500

    def pad(x):
        return x[:max_l] if len(x) > max_l else [0] * (max_l - len(x)) + x

    # tokenized_data是数据按照空格分开后的句子，是一个二维list
    tokenized_data = get_tokenized_imdb(data)
    # features是每个词在字典中的value
    vocab.set_default_index(0)
    features = torch.tensor([pad([vocab.__getitem__(word) for word in words])
                             for words in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels


# 数据迭代期部分是否存在问题，是否是能以一对应？？？？？
# 创建数据迭代器
batch_size = 16
abs_train_set = Data.TensorDataset(*preprocess_imdb(abs_train_data, abs_vocab))
abs_test_set = Data.TensorDataset(*preprocess_imdb(abs_test_data, abs_vocab))
def_train_set = Data.TensorDataset(*preprocess_imdb(def_train_data, def_vocab))
def_test_set = Data.TensorDataset(*preprocess_imdb(def_test_data, def_vocab))
# 其中每个数据集都有16个句子
# 为了一一对应设置不随机是否可行？？？
abs_train_iter = Data.DataLoader(abs_train_set, batch_size, shuffle=False)
abs_test_iter = Data.DataLoader(abs_test_set, batch_size)
def_train_iter = Data.DataLoader(def_train_set, batch_size, shuffle=False)
def_test_iter = Data.DataLoader(def_test_set, batch_size)


def main():
    # 查看数据类型
    for X, y in abs_train_iter:
        print('X', X.shape, 'y', y.shape)
        print(X)
        print('-' * 100)
        print(y)
        break

    for i, data in enumerate(zip(abs_train_iter, def_train_iter)):
        abs_x = data[0][0]
        abs_y = data[0][1]
        print(abs_x[0])
        print('-' * 100)
        print(abs_y)
        break


print(len(abs_train_iter))
# torch.metric()

# f1等数值整出来

if __name__ == "__main__":
    main()
