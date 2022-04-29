import copy
import torch
import torch.nn as nn
from model.toolBar import Config
from model.toolBar.Encoder import Encoder
from model.toolBar.Positional_Encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self,
                 # abs_vocab
                 vocab1=Config.v1,
                 # def_vocab
                 vocab2=Config.v2,
                 # text_length
                 seq_len=Config.seq_len,
                 # GPU or CPU
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 # result of category
                 n_class=Config.class_number,
                 # Dimension of embedding
                 embed_dim=Config.embed_size,
                 # dim_model ????
                 dim_model=Config.dim_model,
                 # num of neuron
                 num_hidden=Config.num_hidden,
                 # dropout
                 dropout=Config.dropout,
                 # num_head
                 num_head=Config.num_head,
                 # num_encoder
                 num_encoder=Config.num_encoder
                 ):
        super(Transformer, self).__init__()
        # 嵌入层
        self.embedding1 = nn.Embedding(len(vocab1), embed_dim)
        self.embedding2 = nn.Embedding(len(vocab2), embed_dim)
        # 位置编码
        self.position_embedding1 = PositionalEncoding(embed_dim, seq_len, dropout, device)
        self.position_embedding2 = PositionalEncoding(embed_dim, seq_len, dropout, device)
        # 没看懂
        self.encoder1 = Encoder(dim_model, num_head, num_hidden, dropout)
        self.encoder2 = Encoder(dim_model, num_head, num_hidden, dropout)
        self.encoders1 = nn.ModuleList([copy.deepcopy(self.encoder1) for _ in range(num_encoder)])
        self.encoders2 = nn.ModuleList([copy.deepcopy(self.encoder2) for _ in range(num_encoder)])
        # 全连接层
        self.layer = nn.Linear(seq_len * dim_model, n_class)

    def forward(self, input1, input2):
        out1 = self.embedding1(input1.permute(1, 0))
        out2 = self.embedding2(input2.permute(1, 0))
        for encoder1 in self.encoders1:
            out1 = encoder1(out1)
        for encoder2 in self.encoders2:
            out2 = encoder2(out2)
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        encoding = torch.cat((out1, out2), -1)
        outs = torch.sigmoid(self.layer(encoding))
        return outs
