import jieba
import torch
from model.lstm_model import *
from model.predit import predict_sentiment
from model.predit import result
from transfer import ModelPredict
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from dataProcess.processing import abs_vocab
from dataProcess.processing import def_vocab


class ModelAPI:
    def __init__(self):
        pass

    @staticmethod
    def cut_data(data):
        text = jieba.cut(data, cut_all=False)
        str_out = ' '.join(text).replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
            .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
            .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
            .replace('’', '')
        return str_out

    def predict(self, abs_word, def_word):
        abs_info = ModelAPI.cut_data(abs_word)
        def_info = ModelAPI.cut_data(def_word)
        input_data1 = []
        input_data2 = []
        for abs_tok in abs_info.split(' '):
            input_data1.append(abs_tok)

        for def_tok in def_info.split(' '):
            input_data2.append(def_tok)
        """
        :param abs_word: 
        :param def_word: 
        :return: 




        test = torch.load('../modelData/LSTM.pkl')

        res = predict_sentiment(test, abs_vocab, def_vocab, input_data1, input_data2)
        """

        return result(input_data1, input_data2)


handler = ModelAPI()
processor = ModelPredict.Processor(handler)
# 服务器端套接字管理
transport = TSocket.TServerSocket("127.0.0.1", 8083)
# 传输方式，使用buffer
t_factory = TTransport.TBufferedTransportFactory()
# 传输的数据类型：二进制
p_factory = TBinaryProtocol.TBinaryProtocolFactory()
# 创建一个thrift 服务~
server = TServer.TThreadPoolServer(processor, transport, t_factory, p_factory)
print("Starting thrift server in python...")
server.serve()
print("done!")
