import pymysql

# TODO 连接数据库
db = pymysql.connect(host='127.0.0.1', user='root', password='123456', db='keyword_matching')

# TODO 创建游标对象
cur = db.cursor()

# TODO 执行MySQL查询
cur.execute("select articleInformation.abstract from articleInformation, reflect_tablepro where articleinformation.ID = reflect_TablePro.article_id")

# TODO 使用fetchone读取单条数据
# data = cur.fetchone()

# TODO 循环读取数据库中内容
temp = []
for res in cur:
    print(res[0])

# print(data)
# TODO 关闭数据库
db.close()
