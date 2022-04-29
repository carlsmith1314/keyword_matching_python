import os
import warnings
import random
from tqdm import tqdm
from model.predit import result
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")


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


data_root = '../DataSet'
# 读取测试数据
abs_basic = read_imdb('abs_test', data_root)
def_basic = read_imdb('def_test', data_root)

abs_test = []
def_test = []
y_label = []
y_predict = []
for a, a_ in abs_basic:
    abs_test.append(a)
    y_label.append(a_)
for b, _ in def_basic:
    def_test.append(b)
for i in range(len(abs_test)):
    y_predict.append(int(result(abs_test[i], def_test[i])))

col_name = ['abs', 'def']
print(classification_report(y_label, y_predict, target_names=col_name))
print(confusion_matrix(y_label, y_predict))
print(type(y_label[0]))
print(type(y_predict[0]))
