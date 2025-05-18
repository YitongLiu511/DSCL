import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe
        self.register_buffer('pe', pe)

    def forward(self, data=None, idx=None):
        if data != None:
            p = self.pe[:data].unsqueeze(0)
        else:
            p = self.pe.unsqueeze(0).repeat(idx.shape[0],1,1)[torch.arange(idx.shape[0])[:,None],idx,:]
        return p


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # 修改输入通道数为1，因为我们的输入是[B, T, C, 1]
        self.tokenConv = nn.Conv1d(in_channels=1, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x: [B, T, C, 1]
        # 调整维度顺序为[B*C, 1, T]，因为Conv1d期望输入是[B, C, T]
        B, T, C, _ = x.shape
        x = x.reshape(B*C, 1, T)  # [B*C, 1, T]
        x = self.tokenConv(x)  # [B*C, D, T]
        x = x.transpose(1, 2)  # [B*C, T, D]
        x = x.reshape(B, C, T, -1)  # [B, C, T, D]
        x = x.transpose(1, 2)  # [B, T, C, D]
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.05):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [B, T, C, 1]
        x = self.value_embedding(x)  # [B, T, C, D]
        # 添加位置编码
        pos_emb = self.position_embedding(data=x.shape[1])  # [1, T, D]
        x = x + pos_emb.unsqueeze(2)  # [B, T, C, D]
        return self.dropout(x)