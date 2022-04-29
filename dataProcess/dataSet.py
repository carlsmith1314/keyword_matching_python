import jieba
import pymysql
import pandas as pd
from sklearn.model_selection import train_test_split
import random

# 提取文献摘要数据
# TODO 连接数据库
db = pymysql.connect(host='127.0.0.1', user='root', password='123456', db='keyword_matching')

# TODO 创建游标对象
cur1 = db.cursor()
cur2 = db.cursor()
# TODO 执行MySQL查询
cur1.execute(
    "select articleInformation.abstract from articleInformation, model_data where articleinformation.ID = model_data.article_id")
cur2.execute(
    "select technical_term.definition from technical_term, model_data where technical_term.id = model_data.technical_id")

"""
# 分词函数
def cutting(data, i):
    # 对初始语料进行分词处理
    # 设置精确模式
    new_text = jieba.cut(data, cut_all=False)
    # 去掉标点符号
    str_out = ' '.join(new_text).replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
        .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
        .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
        .replace('’', '')
    fo = open(str(i) + ".txt", 'w', encoding='utf-8')
    fo.write(str_out)
    fo.close()
# 采用生成随机数来划分训练数据和测试数据
# testNum: 放入测试集的数据编号
testNum = random.sample(range(0, cur1.rowcount - 1), int(cur1.rowcount * 0.2))
testNum = sorted(testNum)


# TODO 循环读取数据库中内容
sym1 = 0
for res1 in cur1:
    # cutting(res1, sym1)
    sym1 = sym1 + 1
sym2 = 0
for res2 in cur2:
    # cutting(res2, sym2)
    sym2 = sym2 + 1
"""

# TODO 关闭数据库
db.close()


# 使用DataFrame处理初始数据
# 定义从数据库读取数据转换成dataframe函数
def transferSQL(sql):
    conn = pymysql.connect(host='127.0.0.1', user='root', password='123456', db='keyword_matching', charset='utf8')
    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    # 获取连接对象的描述信息
    column_describe = cursor.description
    cursor.close()
    conn.close()
    column_names = [column_describe[i][0] for i in range(len(column_describe))]
    results = pd.DataFrame([list(i) for i in results], columns=column_names)
    return results


# init_data:初始数据
# init_sql:映射表查询语句
init_sql = "select * from model_data"
init_data = transferSQL(init_sql)
# 在DataFrame中添加由article_id和technical_id映射得到的摘要和术语定义
# mid1, mid2用来存放从数据库中取出的数据
mid1 = []
mid2 = []
for res1 in cur1:
    mid1.append(res1[0])
for res2 in cur2:
    mid2.append(res2[0])

init_data['abs'] = mid1
init_data['define'] = mid2


# 分词函数
def cut_word(word):
    cw = jieba.cut(word)
    str_out = ' '.join(cw).replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
        .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
        .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
        .replace('’', '')
    return str_out


# 对文献摘要和标准术语进行分词
init_data['abs'] = init_data['abs'].apply(cut_word)
init_data['define'] = init_data['define'].apply(cut_word)

# 去除不相关的列
# cope_data:处理后的数据
cope_data = init_data.drop(columns=['ID', 'article_id', 'keyword', 'technical_id'])

print(cope_data.tail(5))
Y = cope_data['label']
X = cope_data.drop(columns='label')

print(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=12)

# 合并X_train, Y_train以及X_test, Y_test
div1 = X_train.join(Y_train)
div2 = X_test.join(Y_test)
# 将划分的数据存入txt文本文件
# 处理文献摘要—训练集
sym1 = 0
for row in div1.itertuples():
    if getattr(row, 'label') == 1:
        fo = open('../DataSet/abs_train/true/' + str(sym1) + ".txt", 'w', encoding='utf-8')
        fo.write(getattr(row, 'abs'))
        fo.close()
    else:
        fo = open('../DataSet/abs_train/false/' + str(sym1) + ".txt", 'w', encoding='utf-8')
        fo.write(getattr(row, 'abs'))
        fo.close()
    sym1 = sym1 + 1
# 处理术语定义-训练集
sym2 = 0
for row1 in div1.itertuples():
    if getattr(row1, 'label') == 1:
        fo = open('../DataSet/def_train/true/' + str(sym2) + ".txt", 'w', encoding='utf-8')
        fo.write(getattr(row1, 'define'))
        fo.close()
    else:
        fo = open('../DataSet/def_train/false/' + str(sym2) + ".txt", 'w', encoding='utf-8')
        fo.write(getattr(row1, 'define'))
        fo.close()
    sym2 = sym2 + 1
# 处理文献摘要—测试集
sym3 = 0
for row2 in div2.itertuples():
    if getattr(row2, 'label') == 1:
        fo = open('../DataSet/abs_test/true/' + str(sym3) + ".txt", 'w', encoding='utf-8')
        fo.write(getattr(row2, 'abs'))
        fo.close()
    else:
        fo = open('../DataSet/abs_test/false/' + str(sym3) + ".txt", 'w', encoding='utf-8')
        fo.write(getattr(row2, 'abs'))
        fo.close()
    sym3 = sym3 + 1
# 处理术语定义-测试集
sym4 = 0
for row3 in div2.itertuples():
    if getattr(row3, 'label') == 1:
        fo = open('../DataSet/def_test/true/' + str(sym4) + ".txt", 'w', encoding='utf-8')
        fo.write(getattr(row3, 'define'))
        fo.close()
    else:
        fo = open('../DataSet/def_test/false/' + str(sym4) + ".txt", 'w', encoding='utf-8')
        fo.write(getattr(row3, 'define'))
        fo.close()
    sym4 = sym4 + 1


