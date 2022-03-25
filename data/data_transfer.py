# 使用word2vec预训练词向量
# 导入jieba包
# 导入摘要数据
import jieba
import pymysql


# 提取文献摘要数据
# TODO 连接数据库
db = pymysql.connect(host='127.0.0.1', user='root', password='123456', db='keyword_matching')

# TODO 创建游标对象
cur = db.cursor()

# TODO 执行MySQL查询
cur.execute("select articleInformation.abstract from articleInformation, reflect_tablepro where articleinformation.ID = reflect_TablePro.article_id")

# TODO 循环读取数据库中内容
abs_data=""
for res in cur:
    abs_data += res[0]

# TODO 关闭数据库
db.close()

# 对初始语料进行分词处理
new_text = jieba.cut(abs_data, cut_all=False)  # 精确模式
str_out = ' '.join(new_text).replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
        .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
        .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
        .replace('’', '')     # 去掉标点符号
fo = open("abs_cutting.txt", 'w', encoding='utf-8')
fo.write(str_out)

