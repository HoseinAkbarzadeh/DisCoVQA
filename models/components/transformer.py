
from copy import deepcopy

import torch.nn as nn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model=768, hid_dim=1024, dropout=0.0) -> None:
        super(PositionWiseFeedForward, self).__init__()

        self.l1 = nn.Linear(d_model, hid_dim)
        self.l2 = nn.Linear(hid_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.l2(self.dropout(self.l1(x).relu()))


class EncoderLayer(nn.Module):
    def __init__(self, pff, d_model=768, num_heads=1, dropout=0.0) -> None:
        super(EncoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.feedforward = pff

    def forward(self, x):
        xin = x
        x, attn_w = self.attn(x, x, x)
        x = self.norm(x + xin)
        return self.norm(x + self.feedforward(x)), attn_w

class ResidualTransformerEncoder(nn.Module):
    def __init__(self, N=4, d_model=768, dropout=0.0) -> None:
        super(ResidualTransformerEncoder, self).__init__()

        pff = PositionWiseFeedForward(d_model, dropout=dropout)
        layer = EncoderLayer(pff, d_model, dropout=dropout)
        self.sequential = clones(layer, N)

    def forward(self, x):
        xin = x
        for layer in self.sequential:
            x, _ = layer(x)

        return x + xin


class TransformerDecoder(nn.Module):
    def __init__(self, d_model=768, num_heads=1, hid_dim=1024, dropout=0.0, out_dim=1) -> None:
        super(TransformerDecoder, self).__init__()

        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.l1 = nn.Linear(d_model, hid_dim)
        self.l2 = nn.Linear(hid_dim, out_dim)
        self.norm = nn.LayerNorm(d_model)

        self.act = nn.GELU()

    def forward(self, x, y):
        """Forward path of the nn.Module

        Args:
            x (torch.Tensor): T_{PE} Input tensor from frame-by-frame features extraction
            y (torch.Tensor): T_{EN} Input tensor from transformer encoder
        """
        t_en = y
        # input shape: [nbatch, sequence, features]
        # temporal pooling
        x = x.mean(dim=1).unsqueeze(1)
        # output shpae: [nbatch, 1, features]
        x, _ = self.attn(x.repeat(1,y.size(1),1), y, y)
        # residual connection
        x = self.norm(x + t_en)
        return self.l2(self.act(self.l1(x)))