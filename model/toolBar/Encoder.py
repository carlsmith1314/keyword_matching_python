import torch.nn as nn
from model.toolBar.Multi_Head_Attention import MultiHeadAttention
from model.toolBar.Position_wise_Feed_Forward import PositionWiseFeedForward


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(dim_model, num_head, dropout)
        self.feed_forward = PositionWiseFeedForward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)

        return out
