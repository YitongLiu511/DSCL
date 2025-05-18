import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .embed import DataEmbedding, PositionalEmbedding

class MultiheadAttention(nn.Module):
    '''For temporal input'''

    def __init__(
        self,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        factor=5,
        batch_size=-1,
    ) -> None:
        super().__init__()
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.n_heads = n_heads
        self.batch_size = batch_size

        self.q = nn.Linear(d_model, dim_k * n_heads)
        self.k = nn.Linear(d_model, dim_k * n_heads)
        self.v = nn.Linear(d_model, dim_v * n_heads)
        self.o = nn.Linear(dim_v * n_heads, d_model)
        self.norm_fact = 1 / math.sqrt(d_model)

    def attention(self, Q, K, V):
        B, L = Q.shape[:2]
        scores = torch.einsum("blhe,bshe->bhls", Q, K) * self.norm_fact
        scores = scores.softmax(dim=-1)
        output = torch.einsum("bhls,bshd->blhd", scores, V).reshape(B, L, -1)
        return output, scores

    def forward(self, x, y):
        '''x : (B, L, D)'''
        B, L, _ = x.shape
        Q = self.q(x).reshape(B, L, self.n_heads, -1)  # (N, B, L, K)
        K = self.k(x).reshape(B, L, self.n_heads, -1)  # (N, B, L, K)
        V = self.v(y).reshape(B, L, self.n_heads, -1)  # (N, B, L, K)
        output, scores = self.attention(Q, K, V)
        return self.o(output), scores

class TemporalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        dim_fc: int = 128,
        dropout: float = 0.1,
        half: bool = False,
        return_attn: bool = False,
    ) -> None:
        super().__init__()
        self.half_ = half
        self.d_model = d_model
        self.attn = MultiheadAttention(d_model, dim_k, dim_v, n_heads)
        self.conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=dim_fc,
            kernel_size=1,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if not half:
            self.conv2 = nn.Conv1d(
                in_channels=dim_fc,
                out_channels=d_model,
                kernel_size=1,
            )
            self.norm2 = nn.LayerNorm(d_model)
        self.return_attn = return_attn

    def forward(self, x):
        x_, attn = self.attn(x, x)
        y = x = self.norm1(self.dropout(x + x_))
        y = self.dropout(torch.relu(self.conv1(y.transpose(-1, 1))))
        if self.half_:
            if self.return_attn:
                return y, attn
            return y
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x + y)
        if self.return_attn:
            return output, attn
        return output

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, T, D]
        attlist = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x)
            attlist.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attlist

class FreEnc(nn.Module):
    def __init__(self, c_in, c_out, d_model, e_layers, win_size, fr):
        super(FreEnc, self).__init__()

        self.emb = DataEmbedding(c_in, d_model)

        self.enc = Encoder(
            [
                TemporalTransformer(
                    d_model=d_model,
                    dim_k=d_model // 8,
                    dim_v=d_model // 8,
                    n_heads=8,
                    dim_fc=d_model,
                    dropout=0.1,
                    half=False,
                    return_attn=True
                ) for l in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.mask_token = nn.Parameter(torch.zeros(1,d_model,1, dtype=torch.cfloat))

        self.fr = fr
    
    def forward(self, x):
        # x: [B, T, C]
        ex = self.emb(x) # [B, T, D]

        # converting to frequency domain and calculating the mag
        cx = torch.fft.rfft(ex.transpose(1,2))
        mag = torch.sqrt(cx.real ** 2 + cx.imag ** 2) # [B, D, Mag]

        # masking smaller mag
        quantile = torch.quantile(mag, self.fr, dim=2, keepdim=True)
        idx = torch.argwhere(mag<quantile)
        cx[mag<quantile] = self.mask_token.repeat(ex.shape[0], 1, mag.shape[-1])[idx[:,0],idx[:,1],idx[:,2]]

        # converting to time domain
        ix = torch.fft.irfft(cx).transpose(1,2)

        # encoding tokens
        dx, att = self.enc(ix)

        rec = self.pro(dx)
        att.append(rec)

        return att # att(list): [B, T, T]
    
class TemEnc(nn.Module):
    def __init__(self, c_in, c_out, d_model, e_layers, win_size, seq_size, tr):
        super(TemEnc, self).__init__()

        self.emb = DataEmbedding(c_in, d_model)
        self.pos_emb = PositionalEmbedding(d_model)

        self.enc = Encoder(
            [
                TemporalTransformer(
                    d_model=d_model,
                    dim_k=d_model // 8,
                    dim_v=d_model // 8,
                    n_heads=8,
                    dim_fc=d_model,
                    dropout=0.1,
                    half=False,
                    return_attn=True
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.dec = Encoder(
            [
                TemporalTransformer(
                    d_model=d_model,
                    dim_k=d_model // 8,
                    dim_v=d_model // 8,
                    n_heads=8,
                    dim_fc=d_model,
                    dropout=0.1,
                    half=False,
                    return_attn=True
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.mask_token = nn.Parameter(torch.zeros(1,1,d_model))
        self.tr = int(tr * win_size)
        self.seq_size = seq_size
    
    def forward(self, x):
        # x: [B, T, C]
        ex = self.emb(x) # [B, T, D]
        filters = torch.ones(1,1,self.seq_size).to(device)
        ex2 = ex ** 2

        # calculating summation of ex and ex2
        ltr = F.conv1d(ex.transpose(1,2).reshape(-1, ex.shape[1]).unsqueeze(1), filters, padding=self.seq_size-1)
        ltr[:,:,:self.seq_size-1] /= torch.arange(1,self.seq_size).to(device)
        ltr[:,:,self.seq_size-1:] /= self.seq_size
        ltr2 = F.conv1d(ex2.transpose(1,2).reshape(-1, ex.shape[1]).unsqueeze(1), filters, padding=self.seq_size-1)
        ltr2[:,:,:self.seq_size-1] /= torch.arange(1,self.seq_size).to(device)
        ltr2[:,:,self.seq_size-1:] /= self.seq_size
        
        # calculating mean and variance
        ltrd = (ltr2 - ltr ** 2)[:,:,:ltr.shape[-1]-self.seq_size+1].squeeze(1).reshape(ex.shape[0],ex.shape[-1],-1).transpose(1,2)
        ltrm = ltr[:,:,:ltr.shape[-1]-self.seq_size+1].squeeze(1).reshape(ex.shape[0],ex.shape[-1],-1).transpose(1,2)
        score = ltrd.sum(-1) / ltrm.sum(-1)

        # mask time points
        masked_idx, unmasked_idx = score.topk(self.tr, dim=1, sorted=False)[1], (-1*score).topk(x.shape[1]-self.tr, dim=1, sorted=False)[1]
        unmasked_tokens = ex[torch.arange(ex.shape[0])[:,None],unmasked_idx,:]
        
        # encoding unmasked tokens and getting masked tokens
        ux, _ = self.enc(unmasked_tokens)
        masked_tokens = self.mask_token.repeat(ex.shape[0], masked_idx.shape[1], 1) + self.pos_emb(idx = masked_idx)
        
        tokens = torch.zeros(ex.shape,device=device)
        tokens[torch.arange(ex.shape[0])[:,None],unmasked_idx,:] = ux
        tokens[torch.arange(ex.shape[0])[:,None],masked_idx,:] = masked_tokens

        # decoding tokens
        dx, att = self.dec(tokens)

        rec = self.pro(dx)
        att.append(rec)

        return att # att(list): [B, T, T]

class MTFA(nn.Module):
    def __init__(self, win_size, seq_size, c_in, c_out, d_model=512, e_layers=3, fr=0.4, tr=0.5, dev=None):
        super(MTFA, self).__init__()
        global device
        device = dev
        self.tem = TemEnc(1, c_out, d_model, e_layers, win_size, seq_size, tr)
        self.fre = FreEnc(1, c_out, d_model, e_layers, win_size, fr)

    def forward(self, x):
        # x: [B, T, C]
        # 确保输入维度正确
        if len(x.shape) == 4:  # 如果输入是[B, 1, T, C]
            x = x.squeeze(1)  # 移除多余的维度，变成[B, T, C]
        # 调整维度顺序为[B, T, 1]
        x = x.unsqueeze(-1)  # [B, T, 1]
        tematt = self.tem(x) # tematt: [B, T, T]
        freatt = self.fre(x) # freatt: [B, T, T]
        return tematt, freatt