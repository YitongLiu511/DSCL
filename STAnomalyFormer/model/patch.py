import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
import math

from .tsfm import TemporalTransformer
from .module import SingleGCN, MultiheadAttention
from .embed import TemporalEmbedding
from .revin import RevIN
from .contrastive import compute_contrastive_loss, compute_anomaly_score


def random_masking(xb, mask_ratio):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(
        bs,
        L,
        nvars,
        device=xb.device,
    )  # noise in [0, 1], bs x L x nvars

    # sort noise for each sample
    ids_shuffle = torch.argsort(
        noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle,
                                dim=1)  # ids_restore: [bs x L x nvars]

    # keep the first subset
    ids_keep = ids_shuffle[:, :
                           len_keep, :]  # ids_keep: [bs x len_keep x nvars]
    x_kept = torch.gather(
        x,
        dim=1,
        index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D),
    )  # x_kept: [bs x len_keep x nvars  x patch_len]

    # removed x
    x_removed = torch.zeros(
        bs,
        L - len_keep,
        nvars,
        D,
        device=xb.device,
    )  # x_removed: [bs x (L-len_keep) x nvars x patch_len]
    x_ = torch.cat(
        [x_kept, x_removed],
        dim=1,
    )  # x_: [bs x L x nvars x patch_len]

    # combine the kept part and the removed one
    x_masked = torch.gather(
        x_,
        dim=1,
        index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, D),
    )  # x_masked: [bs x num_patch x nvars x patch_len]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones(
        [bs, L, nvars],
        device=x.device,
    )  # mask: [bs x num_patch x nvars]
    mask[:, :len_keep, :] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(
        mask,
        dim=1,
        index=ids_restore,
    )  # [bs x num_patch x nvars]
    return x_masked, x_kept, mask, ids_restore


def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty(
            (q_len, d_model)
        )  # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'sincos':
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(
            f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)


class Patch(nn.Module):

    def __init__(self, seq_len, patch_len, stride):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
        tgt_len = patch_len + stride * (self.num_patch - 1)
        self.s_begin = seq_len - tgt_len

    def forward(self, x):
        """
        x: [bs x seq_len x n_vars]
        """
        x = x[:, self.s_begin:, :]
        x = x.unfold(
            dimension=1,
            size=self.patch_len,
            step=self.stride,
        )  # xb: [bs x num_patch x n_vars x patch_len]
        return x


class PatchEncoder(nn.Module):

    def __init__(
        self,
        c_in,
        num_patch,
        patch_len,
        d_model=128,
        n_heads=16,
        d_ff=256,
        shared_embedding=True,
        attn_dropout=0.,
        dropout=0.,
        half: bool = False,
        pe='zeros',
        learn_pe=True,
    ):

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding
        self.n_heads = n_heads

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not shared_embedding:
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars):
                self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model)

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        self.encoder = TemporalTransformer(
            d_model=d_model,
            dim_k=d_model // n_heads,
            dim_v=d_model // n_heads,
            n_heads=n_heads,
            dim_fc=d_ff,
            dropout=attn_dropout,
            half=half,
            return_attn=False,
        )

    def forward(self, x):
        bs, num_patch, n_vars, patch_len = x.shape
        print(f"\nPatchEncoder维度追踪:")
        print(f"1. 输入维度: {x.shape}")
        
        # Input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars):
                z = self.W_P[i](x[:, :, i, :])  # [bs, num_patch, d_model]
                x_out.append(z)
            x = torch.stack(x_out, dim=2)  # [bs, num_patch, n_vars, d_model]
        else:
            x = x.reshape(-1, patch_len)  # [bs*num_patch*n_vars, patch_len]
            x = self.W_P(x)  # [bs*num_patch*n_vars, d_model]
            x = x.reshape(bs, num_patch, n_vars, self.d_model)  # [bs, num_patch, n_vars, d_model]
        print(f"2. 编码后维度: {x.shape}")
        
        # 调整维度顺序
        x = x.transpose(1, 2)  # [bs, n_vars, num_patch, d_model]
        print(f"3. 维度重排后: {x.shape}")
        
        u = torch.reshape(x, (bs * n_vars, num_patch, self.d_model))  # [bs*n_vars, num_patch, d_model]
        print(f"4. 重塑后维度: {u.shape}")
        
        u = self.dropout(u + self.W_pos)  # [bs*n_vars, num_patch, d_model]
        print(f"5. 位置编码后: {u.shape}")

        # Encoder
        z = self.encoder(u)  # [bs*n_vars, num_patch, d_model]
        print(f"6. Transformer编码后: {z.shape}")
        
        z = z.reshape(bs, n_vars, num_patch, self.d_model)  # [bs, n_vars, num_patch, d_model]
        print(f"7. 最终输出维度: {z.shape}")
        
        return z


class RandomMasking(nn.Module):

    def __init__(self, mask_ratio: float = 0.4) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, xb):
        if self.training:
            x_masked, _, mask, _ = random_masking(xb, self.mask_ratio)
            self.mask = mask.bool()
        else:
            x_masked = xb
        return x_masked


class TimeFreqMasking(nn.Module):
    def __init__(self, mask_ratio: float = 0.4, time_ratio: float = 0.5, freq_ratio: float = 0.4, patch_size: int = 12):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.time_ratio = time_ratio
        self.freq_ratio = freq_ratio
        self.patch_size = patch_size  # 分片大小
        
        # 可学习的掩码token，使用浮点类型
        self.time_mask_token = nn.Parameter(torch.zeros(1, 1, 1))
        self.freq_mask_token = nn.Parameter(torch.zeros(1, 1, 1))
        
    def time_masking(self, x):
        """
        时间域掩码
        Args:
            x: 输入数据 [bs, num_patch, n_vars, patch_len]
        Returns:
            x_masked: 掩码后的数据
            mask: 掩码矩阵
        """
        bs, num_patch, n_vars, patch_len = x.shape
        device = x.device
        
        print(f"\n时间掩码处理维度追踪:")
        print(f"1. 输入维度: {x.shape}")
        
        # 计算每个patch的变异系数
        cv = torch.std(x, dim=3) / (torch.mean(x, dim=3) + 1e-6)  # [bs, num_patch, n_vars]
        print(f"2. 变异系数维度: {cv.shape}")
        
        # 对每个变量选择变异系数最大的patch进行掩码
        patch_mask = torch.ones((bs, num_patch, n_vars), dtype=torch.bool, device=device)  # [bs, num_patch, n_vars]
        num_masks = int(num_patch * self.time_ratio)
        print(f"3. 掩码数量: {num_masks}")
        
        # 获取变异系数最大的patch索引
        _, mask_idx = torch.topk(cv, k=num_masks, dim=1)  # [bs, num_masks, n_vars]
        print(f"4. 掩码索引维度: {mask_idx.shape}")
        
        # 应用掩码
        for i in range(bs):
            for j in range(n_vars):
                for k in range(num_masks):
                    if mask_idx[i,k,j] < num_patch:  # 确保索引在有效范围内
                        patch_mask[i, mask_idx[i,k,j], j] = 0
        
        print(f"5. 掩码矩阵维度: {patch_mask.shape}")
        print(f"5.1 掩码矩阵中True的数量: {patch_mask.sum().item()}")
        print(f"5.2 掩码矩阵中False的数量: {(~patch_mask).sum().item()}")
        
        # 分离掩码和未掩码的token
        unmasked_tokens = x[patch_mask]  # [num_unmasked, patch_len]
        masked_tokens = self.time_mask_token.repeat(bs, num_masks, n_vars, 1)
        print(f"6. 未掩码token维度: {unmasked_tokens.shape}")
        print(f"7. 掩码token维度: {masked_tokens.shape}")
        
        # 重塑未掩码的token以适应编码器
        num_unmasked = unmasked_tokens.shape[0]
        unmasked_tokens = unmasked_tokens.reshape(-1, 1, patch_len)  # [num_unmasked, 1, patch_len]
        print(f"8. 重塑后未掩码token维度: {unmasked_tokens.shape}")
        
        # 投影到模型维度
        unmasked_tokens = self.patch2emb(unmasked_tokens)  # [num_unmasked, 1, d_model]
        print(f"9. 投影后未掩码token维度: {unmasked_tokens.shape}")
        
        # 编码未掩码的token
        encoded_tokens = unmasked_tokens
        for i, encoder in enumerate(self.time_encoder):
            print(f"\n10. 编码器 {i+1} 输入维度: {encoded_tokens.shape}")
            encoded_tokens, _ = encoder(encoded_tokens)
            print(f"    编码器 {i+1} 输出维度: {encoded_tokens.shape}")
        
        # 重塑编码后的token
        encoded_tokens = encoded_tokens.reshape(-1, self.d_model)  # [num_unmasked, d_model]
        print(f"\n11. 编码后token维度: {encoded_tokens.shape}")
        
        # 组合编码后的token和掩码token
        tokens = torch.zeros(bs, num_patch, n_vars, self.d_model, device=device)
        print(f"\n12. 组合前维度检查:")
        print(f"12.1 tokens维度: {tokens.shape}")
        print(f"12.2 patch_mask维度: {patch_mask.shape}")
        print(f"12.3 encoded_tokens维度: {encoded_tokens.shape}")
        print(f"12.4 time_mask_token维度: {self.time_mask_token.shape}")
        
        # 创建索引映射
        unmasked_indices = torch.where(patch_mask.reshape(-1))[0]  # [num_unmasked]
        print(f"\n13. unmasked_indices维度: {unmasked_indices.shape}")
        
        # 使用索引赋值
        tokens.reshape(-1, self.d_model)[unmasked_indices] = encoded_tokens
        tokens.reshape(-1, self.d_model)[~unmasked_indices] = self.time_mask_token
        print(f"14. 组合后tokens维度: {tokens.shape}")
        
        # 解码所有token
        decoded_tokens = tokens
        for i, decoder in enumerate(self.time_decoder):
            print(f"\n15. 解码器 {i+1} 输入维度: {decoded_tokens.shape}")
            # 调整维度以适应解码器
            decoded_tokens = decoded_tokens.reshape(bs * n_vars, num_patch, self.d_model)  # [bs*n_vars, num_patch, d_model]
            print(f"    解码器 {i+1} 调整后输入维度: {decoded_tokens.shape}")
            decoded_tokens, _ = decoder(decoded_tokens)
            print(f"    解码器 {i+1} 输出维度: {decoded_tokens.shape}")
            # 恢复原始维度
            decoded_tokens = decoded_tokens.reshape(bs, num_patch, n_vars, self.d_model)  # [bs, num_patch, n_vars, d_model]
            print(f"    解码器 {i+1} 恢复后维度: {decoded_tokens.shape}")
        
        # 投影回原始维度
        decoded_tokens = self.emb2patch(decoded_tokens)  # [bs, num_patch, n_vars, patch_len]
        print(f"\n16. 最终输出维度: {decoded_tokens.shape}")
        
        return decoded_tokens, patch_mask
        
    def freq_masking(self, x):
        """
        频域掩码
        Args:
            x: 输入数据 [bs, num_patch, n_vars, patch_len]
        Returns:
            x_masked: 掩码后的数据
            mask: 掩码矩阵
        """
        bs, num_patch, n_vars, patch_len = x.shape
        device = x.device
        
        print(f"\n频域掩码处理维度追踪:")
        print(f"1. 输入维度: {x.shape}")
        
        # 对每个patch进行FFT
        cx = torch.fft.rfft(x, dim=-1)  # [bs, num_patch, n_vars, patch_len//2+1]
        print(f"2. FFT后维度: {cx.shape}")
        
        # 计算每个patch的频率幅度
        mag = torch.sqrt(cx.real**2 + cx.imag**2)  # [bs, num_patch, n_vars, patch_len//2+1]
        print(f"3. 频率幅度维度: {mag.shape}")
        
        # 计算每个patch的频率重要性得分
        patch_freq_score = mag.mean(dim=-1)  # [bs, num_patch, n_vars]
        print(f"4. 频率重要性得分维度: {patch_freq_score.shape}")
        
        # 选择频率重要性最低的patch进行掩码
        num_masks = int(num_patch * self.freq_ratio)
        _, mask_idx = torch.topk(patch_freq_score, k=num_masks, dim=1, largest=False)  # [bs, num_masks, n_vars]
        print(f"5. 掩码索引维度: {mask_idx.shape}")
        
        # 生成掩码
        mask = torch.ones((bs, num_patch, n_vars), dtype=torch.bool, device=device)  # [bs, num_patch, n_vars]
        print(f"6. 初始掩码维度: {mask.shape}")
        
        # 应用掩码
        for i in range(bs):
            for j in range(n_vars):
                for k in range(num_masks):
                    if mask_idx[i,k,j] < num_patch:  # 确保索引在有效范围内
                        mask[i, mask_idx[i,k,j], j] = 0
        
        print(f"7. 应用掩码后:")
        print(f"7.1 掩码中True的数量: {mask.sum().item()}")
        print(f"7.2 掩码中False的数量: {(~mask).sum().item()}")
        
        # 使用可学习的掩码token替换被掩码的位置
        x_masked = x.clone()
        
        # 投影到模型维度
        x_masked = x_masked.reshape(-1, 1, patch_len)  # [bs*num_patch*n_vars, 1, patch_len]
        x_masked = self.patch2emb(x_masked)  # [bs*num_patch*n_vars, 1, d_model]
        x_masked = x_masked.reshape(bs, num_patch, n_vars, self.d_model)  # [bs, num_patch, n_vars, d_model]
        print(f"8. 投影后维度: {x_masked.shape}")
        
        # 应用掩码
        mask = mask.unsqueeze(-1)  # [bs, num_patch, n_vars, 1]
        mask = mask.expand(-1, -1, -1, self.d_model)  # [bs, num_patch, n_vars, d_model]
        
        # 扩展掩码token到正确的维度
        freq_mask_token = self.freq_mask_token.expand(bs, num_patch, n_vars, self.d_model)  # [bs, num_patch, n_vars, d_model]
        x_masked = torch.where(mask, x_masked, freq_mask_token)  # 使用where替代直接索引
        print(f"9. 掩码后维度: {x_masked.shape}")
        
        # 投影回原始维度
        x_masked = x_masked.reshape(-1, self.d_model)  # [bs*num_patch*n_vars, d_model]
        x_masked = self.emb2patch(x_masked)  # [bs*num_patch*n_vars, patch_len]
        x_masked = x_masked.reshape(bs, num_patch, n_vars, patch_len)  # [bs, num_patch, n_vars, patch_len]
        print(f"10. 最终输出维度: {x_masked.shape}")
        
        return x_masked, mask
        
    def forward(self, x):
        if self.training:
            # 应用时间掩码
            x_time_masked, time_mask = self.time_masking(x)
            
            # 应用频域掩码
            x_freq_masked, freq_mask = self.freq_masking(x)
            
            # 组合两种掩码的结果
            x_masked = x_time_masked * 0.5 + x_freq_masked * 0.5
            
            # 保存掩码信息用于后续重建
            self.time_mask = time_mask
            self.freq_mask = freq_mask
            
            return x_masked
        else:
            return x


class DynamicTimeFreqMasking(TimeFreqMasking):
    def __init__(self, mask_ratio: float = 0.4, time_ratio: float = 0.5, freq_ratio: float = 0.4, 
                 patch_size: int = 12, d_model: int = 512, n_heads: int = 8, n_layers: int = 3):
        super().__init__(mask_ratio, time_ratio, freq_ratio, patch_size)
        
        # 保存d_model参数
        self.d_model = d_model
        
        # 添加投影层
        self.patch2emb = nn.Linear(patch_size, d_model)  # patch_len -> d_model
        self.emb2patch = nn.Linear(d_model, patch_size)  # d_model -> patch_len
        
        # 时间掩码编码器-解码器
        self.time_encoder = nn.ModuleList([
            TemporalTransformer(
                d_model=d_model,
                dim_k=d_model // n_heads,
                dim_v=d_model // n_heads,
                n_heads=n_heads,
                dim_fc=d_model,
                dropout=0.1,
                half=False,
                return_attn=True
            ) for _ in range(n_layers)
        ])
        
        self.time_decoder = nn.ModuleList([
            TemporalTransformer(
                d_model=d_model,
                dim_k=d_model // n_heads,
                dim_v=d_model // n_heads,
                n_heads=n_heads,
                dim_fc=d_model,
                dropout=0.1,
                half=False,
                return_attn=True
            ) for _ in range(n_layers)
        ])
        
        # 频率掩码解码器
        self.freq_decoder = nn.ModuleList([
            TemporalTransformer(
                d_model=d_model,
                dim_k=d_model // n_heads,
                dim_v=d_model // n_heads,
                n_heads=n_heads,
                dim_fc=d_model,
                dropout=0.1,
                half=False,
                return_attn=True
            ) for _ in range(n_layers)
        ])
        
        # 投影层
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.freq_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # 可学习的掩码token
        self.time_mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.freq_mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
    def time_masking(self, x):
        """
        时间域掩码
        Args:
            x: 输入数据 [bs, num_patch, n_vars, patch_len]
        Returns:
            x_masked: 掩码后的数据
            mask: 掩码矩阵
        """
        bs, num_patch, n_vars, patch_len = x.shape
        device = x.device
        
        print(f"\n时间掩码处理维度追踪:")
        print(f"1. 输入维度: {x.shape}")
        
        # 计算每个patch的变异系数
        cv = torch.std(x, dim=3) / (torch.mean(x, dim=3) + 1e-6)  # [bs, num_patch, n_vars]
        print(f"2. 变异系数维度: {cv.shape}")
        
        # 对每个变量选择变异系数最大的patch进行掩码
        patch_mask = torch.ones((bs, num_patch, n_vars), dtype=torch.bool, device=device)  # [bs, num_patch, n_vars]
        num_masks = int(num_patch * self.time_ratio)
        print(f"3. 掩码数量: {num_masks}")
        
        # 获取变异系数最大的patch索引
        _, mask_idx = torch.topk(cv, k=num_masks, dim=1)  # [bs, num_masks, n_vars]
        print(f"4. 掩码索引维度: {mask_idx.shape}")
        
        # 应用掩码
        for i in range(bs):
            for j in range(n_vars):
                for k in range(num_masks):
                    if mask_idx[i,k,j] < num_patch:  # 确保索引在有效范围内
                        patch_mask[i, mask_idx[i,k,j], j] = 0
        
        print(f"5. 掩码矩阵维度: {patch_mask.shape}")
        print(f"5.1 掩码矩阵中True的数量: {patch_mask.sum().item()}")
        print(f"5.2 掩码矩阵中False的数量: {(~patch_mask).sum().item()}")
        
        # 分离掩码和未掩码的token
        unmasked_tokens = x[patch_mask]  # [num_unmasked, patch_len]
        masked_tokens = self.time_mask_token.repeat(bs, num_masks, n_vars, 1)
        print(f"6. 未掩码token维度: {unmasked_tokens.shape}")
        print(f"7. 掩码token维度: {masked_tokens.shape}")
        
        # 重塑未掩码的token以适应编码器
        num_unmasked = unmasked_tokens.shape[0]
        unmasked_tokens = unmasked_tokens.reshape(-1, 1, patch_len)  # [num_unmasked, 1, patch_len]
        print(f"8. 重塑后未掩码token维度: {unmasked_tokens.shape}")
        
        # 投影到模型维度
        unmasked_tokens = self.patch2emb(unmasked_tokens)  # [num_unmasked, 1, d_model]
        print(f"9. 投影后未掩码token维度: {unmasked_tokens.shape}")
        
        # 编码未掩码的token
        encoded_tokens = unmasked_tokens
        for i, encoder in enumerate(self.time_encoder):
            print(f"\n10. 编码器 {i+1} 输入维度: {encoded_tokens.shape}")
            encoded_tokens, _ = encoder(encoded_tokens)
            print(f"    编码器 {i+1} 输出维度: {encoded_tokens.shape}")
        
        # 重塑编码后的token
        encoded_tokens = encoded_tokens.reshape(-1, self.d_model)  # [num_unmasked, d_model]
        print(f"\n11. 编码后token维度: {encoded_tokens.shape}")
        
        # 组合编码后的token和掩码token
        tokens = torch.zeros(bs, num_patch, n_vars, self.d_model, device=device)
        print(f"\n12. 组合前维度检查:")
        print(f"12.1 tokens维度: {tokens.shape}")
        print(f"12.2 patch_mask维度: {patch_mask.shape}")
        print(f"12.3 encoded_tokens维度: {encoded_tokens.shape}")
        print(f"12.4 time_mask_token维度: {self.time_mask_token.shape}")
        
        # 创建索引映射
        unmasked_indices = torch.where(patch_mask.reshape(-1))[0]  # [num_unmasked]
        print(f"\n13. unmasked_indices维度: {unmasked_indices.shape}")
        
        # 使用索引赋值
        tokens.reshape(-1, self.d_model)[unmasked_indices] = encoded_tokens
        tokens.reshape(-1, self.d_model)[~unmasked_indices] = self.time_mask_token
        print(f"14. 组合后tokens维度: {tokens.shape}")
        
        # 解码所有token
        decoded_tokens = tokens
        for i, decoder in enumerate(self.time_decoder):
            print(f"\n15. 解码器 {i+1} 输入维度: {decoded_tokens.shape}")
            # 调整维度以适应解码器
            decoded_tokens = decoded_tokens.reshape(bs * n_vars, num_patch, self.d_model)  # [bs*n_vars, num_patch, d_model]
            print(f"    解码器 {i+1} 调整后输入维度: {decoded_tokens.shape}")
            decoded_tokens, _ = decoder(decoded_tokens)
            print(f"    解码器 {i+1} 输出维度: {decoded_tokens.shape}")
            # 恢复原始维度
            decoded_tokens = decoded_tokens.reshape(bs, num_patch, n_vars, self.d_model)  # [bs, num_patch, n_vars, d_model]
            print(f"    解码器 {i+1} 恢复后维度: {decoded_tokens.shape}")
        
        # 投影回原始维度
        decoded_tokens = self.emb2patch(decoded_tokens)  # [bs, num_patch, n_vars, patch_len]
        print(f"\n16. 最终输出维度: {decoded_tokens.shape}")
        
        return decoded_tokens, patch_mask


class STPatchFormer(nn.Module):
    def __init__(
        self,
        c_in,
        seq_len,
        patch_len,
        stride,
        max_seq_len,
        n_layers,
        d_model,
        n_heads,
        d_ff,
        shared_embedding=True,
        attn_dropout=0.,
        dropout=0.,
        act='gelu',
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding
        self.n_layers = n_layers
        self.c_in = c_in

        print(f"\n模型初始化参数:")
        print(f"- 输入特征维度 (c_in): {c_in}")
        print(f"- 序列长度 (seq_len): {seq_len}")
        print(f"- Patch长度 (patch_len): {patch_len}")
        print(f"- 步长 (stride): {stride}")
        print(f"- 模型维度 (d_model): {d_model}")

        # 创建分片模块
        self.patch = Patch(seq_len, patch_len, stride)
        num_patch = self.patch.num_patch
        print(f"- 分片数量 (num_patch): {num_patch}")

        # 创建分片编码器
        self.patch_tsfm = PatchEncoder(
            c_in=c_in,
            num_patch=num_patch,
            patch_len=patch_len,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            attn_dropout=attn_dropout,
            shared_embedding=shared_embedding,
        )

        # 创建时间位置编码
        self.temporal_embedding = TemporalEmbedding(
            c_in=d_model,
            d_model=d_model,
            dropout=dropout,
        )

        # 创建可学习的掩码token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # 添加输出投影层
        self.output_proj = nn.Linear(d_model, c_in)  # 修改为投影到原始特征维度

        self.revin = RevIN(c_in)
        
    def forward(self, x):
        print(f"\n前向传播维度追踪:")
        print(f"1. 输入数据维度: {x.shape}")
        
        # 1. 分片处理
        patch_x = self.patch(x)  # [bs, num_patch, n_vars, patch_len]
        print(f"2. 分片后维度: {patch_x.shape}")
        
        # 2. 数据归一化
        x = self.revin(patch_x.transpose(2, 3), 'norm').transpose(2, 3)
        print(f"3. 归一化后维度: {x.shape}")
        
        # 3. 分片编码
        z = self.patch_tsfm(x)  # [bs, n_vars, num_patch, d_model]
        print(f"4. 编码后维度: {z.shape}")
        
        # 4. 输出投影
        z = self.output_proj(z)  # [bs, n_vars, num_patch, c_in]
        print(f"5. 投影后维度: {z.shape}")
        
        # 5. 还原到原始维度
        bs, n_vars, num_patch, c_in = z.shape
        z = z.transpose(1, 2)  # [bs, num_patch, n_vars, c_in]
        print(f"6. 维度重排后: {z.shape}")
        
        # 6. 还原分片
        z = z.reshape(bs, -1, n_vars)  # [bs, num_patch*c_in, n_vars]
        print(f"7. 还原分片后: {z.shape}")
        z = z[:, :self.seq_len, :]  # 截取到原始序列长度
        print(f"8. 截取长度后: {z.shape}")
        
        # 7. 反归一化
        z = self.revin(z, 'denorm')
        print(f"9. 最终输出维度: {z.shape}")
        
        return z


class STPatchMaskFormer(STPatchFormer):
    def __init__(self, c_in, seq_len, patch_len, stride, max_seq_len, n_layers, d_model, n_heads, d_ff, 
                 shared_embedding=True, attn_dropout=0., dropout=0., act='gelu', mask_ratio: float = 0.4, 
                 time_ratio: float = 0.5, freq_ratio: float = 0.4, patch_size: int = 12):
        super().__init__(c_in, seq_len, patch_len, stride, max_seq_len, n_layers, d_model, n_heads, d_ff,
                        shared_embedding, attn_dropout, dropout, act)
        
        # 使用新的 DynamicTimeFreqMasking 替换原来的 TimeFreqMasking
        self.mask = DynamicTimeFreqMasking(mask_ratio, time_ratio, freq_ratio, patch_size, d_model, n_heads, n_layers)
        
        # 添加训练相关的组件
        self.criterion = nn.MSELoss()
        
        # 导入对比损失函数
        self.compute_contrastive_loss = compute_contrastive_loss
        self.compute_anomaly_score = compute_anomaly_score
        
    def forward(self, x, return_dict=False):
        print(f"\nSTPatchMaskFormer前向传播维度追踪:")
        print(f"1. 输入数据维度: {x.shape}")
        
        # 1. 分片处理
        patch_x = self.patch(x)  # [bs, num_patch, n_vars, patch_len]
        print(f"2. 分片后维度: {patch_x.shape}")
        
        # 2. 数据归一化
        x = self.revin(patch_x.transpose(2, 3), 'norm').transpose(2, 3)
        print(f"3. 归一化后维度: {x.shape}")
        
        # 3. 应用掩码
        if self.training:
            x_masked = self.mask(x)  # 使用新的掩码机制
            print(f"4. 掩码后维度: {x_masked.shape}")
        else:
            x_masked = x
        
        # 4. 分片编码
        z = self.patch_tsfm(x_masked)  # [bs, n_vars, num_patch, d_model]
        print(f"5. 编码后维度: {z.shape}")
        
        # 5. 输出投影
        z = self.output_proj(z)  # [bs, n_vars, num_patch, c_in]
        print(f"6. 投影后维度: {z.shape}")
        
        # 6. 还原到原始维度
        bs, n_vars, num_patch, c_in = z.shape
        z = z.transpose(1, 2)  # [bs, num_patch, n_vars, c_in]
        print(f"7. 维度重排后: {z.shape}")
        
        # 7. 还原分片
        z = z.reshape(bs, -1, n_vars)  # [bs, num_patch*c_in, n_vars]
        print(f"8. 还原分片后: {z.shape}")
        z = z[:, :self.seq_len, :]  # 截取到原始序列长度
        print(f"9. 截取长度后: {z.shape}")
        
        # 8. 反归一化
        z = self.revin(z, 'denorm')
        print(f"10. 最终输出维度: {z.shape}")
        
        if return_dict:
            print("\n异常分数计算维度追踪:")
            # 计算异常分数
            # 确保x和z的维度匹配
            x_reshaped = x.reshape(bs, -1, n_vars)  # [bs, seq_len, n_vars]
            x_reshaped = x_reshaped[:, :self.seq_len, :]  # 截取到原始序列长度
            print(f"11. x_reshaped维度: {x_reshaped.shape}")
            
            # 确保z的维度正确
            if len(z.shape) == 4:  # 如果z是[bs, bs, seq_len, n_vars]
                z = z[0]  # 取第一个batch的预测结果
            print(f"12. z调整后维度: {z.shape}")
            
            # 计算重建损失
            recon_loss = self.criterion(z, x_reshaped)
            
            # 计算异常分数
            anomaly_scores = torch.abs(z - x_reshaped)  # [bs, seq_len, n_vars]
            print(f"13. anomaly_scores维度: {anomaly_scores.shape}")
            
            # 分离时间和空间异常分数
            region_scores = anomaly_scores.mean(dim=1)  # [bs, n_vars]
            time_scores = anomaly_scores.mean(dim=2)    # [bs, seq_len]
            print(f"14. region_scores维度: {region_scores.shape}")
            print(f"15. time_scores维度: {time_scores.shape}")
            
            # 调整维度以匹配
            region_scores = region_scores.unsqueeze(1).expand(-1, self.seq_len, -1)  # [bs, seq_len, n_vars]
            time_scores = time_scores.unsqueeze(2).expand(-1, -1, n_vars)  # [bs, seq_len, n_vars]
            print(f"16. region_scores调整后维度: {region_scores.shape}")
            print(f"17. time_scores调整后维度: {time_scores.shape}")
            
            # 计算阈值
            region_thresh = torch.quantile(region_scores, 0.9)
            time_thresh = torch.quantile(time_scores, 0.9)
            print(f"18. region_thresh: {region_thresh}")
            print(f"19. time_thresh: {time_thresh}")
            
            # 预测异常
            region_pred = (region_scores > region_thresh).float()
            time_pred = (time_scores > time_thresh).float()
            print(f"20. region_pred维度: {region_pred.shape}")
            print(f"21. time_pred维度: {time_pred.shape}")
            
            # 获取异常区域和时间戳
            anomaly_regions = torch.where(region_pred == 1)[1]
            anomaly_timestamps = torch.where(time_pred == 1)[1]
            print(f"22. anomaly_regions数量: {len(anomaly_regions)}")
            print(f"23. anomaly_timestamps数量: {len(anomaly_timestamps)}")
            
            return {
                'reconstruction': z,
                'anomaly_scores': torch.stack([region_scores, time_scores], dim=-1),
                'region_scores': region_scores,
                'time_scores': time_scores,
                'region_pred': region_pred,
                'time_pred': time_pred,
                'anomaly_regions': anomaly_regions,
                'anomaly_timestamps': anomaly_timestamps,
                'region_threshold': region_thresh,
                'time_threshold': time_thresh,
                'recon_loss': recon_loss
            }
        
        return z

    def evaluate(self, x, y_true):
        """
        评估模型性能
        Args:
            x: 输入数据 [N, T, D]
            y_true: 真实标签 [N, T, D, 2]
        Returns:
            metrics: 评估指标字典
        """
        self.eval()
        with torch.no_grad():
            print("\n=== 评估阶段详细调试信息 ===")
            print(f"1. 输入数据维度:")
            print(f"   x: {x.shape}")
            print(f"   y_true: {y_true.shape}")
            
            # 获取模型输出
            output_dict = self(x, return_dict=True)
            
            # 获取异常分数
            anomaly_scores = output_dict['anomaly_scores']  # [N, T, D, 2]
            region_scores = anomaly_scores[:, :, :, 0]     # [N, T, D]
            time_scores = anomaly_scores[:, :, :, 1]       # [N, T, D]
            
            print(f"\n2. 模型输出维度:")
            print(f"   anomaly_scores: {anomaly_scores.shape}")
            print(f"   region_scores: {region_scores.shape}")
            print(f"   time_scores: {time_scores.shape}")
            
            # 将张量移动到CPU并转换为numpy数组
            y_true = y_true.cpu().numpy()
            region_scores = region_scores.cpu().numpy()
            time_scores = time_scores.cpu().numpy()
            
            print(f"\n3. 转换为numpy后的维度:")
            print(f"   y_true: {y_true.shape}")
            print(f"   region_scores: {region_scores.shape}")
            print(f"   time_scores: {time_scores.shape}")
            
            # 确保维度匹配
            y_true_region = y_true[:, :, :, 0]  # [N, T, D]
            y_true_time = y_true[:, :, :, 1]    # [N, T, D]
            
            print(f"\n4. 调整维度后:")
            print(f"   y_true_region: {y_true_region.shape}")
            print(f"   region_scores: {region_scores.shape}")
            print(f"   y_true_time: {y_true_time.shape}")
            print(f"   time_scores: {time_scores.shape}")
            
            try:
                # 展平数组以计算AUC
                y_true_region_flat = y_true_region.reshape(-1)
                region_scores_flat = region_scores.reshape(-1)
                y_true_time_flat = y_true_time.reshape(-1)
                time_scores_flat = time_scores.reshape(-1)
                
                print(f"\n5. 展平后维度:")
                print(f"   y_true_region_flat: {y_true_region_flat.shape}")
                print(f"   region_scores_flat: {region_scores_flat.shape}")
                print(f"   y_true_time_flat: {y_true_time_flat.shape}")
                print(f"   time_scores_flat: {time_scores_flat.shape}")
                
                # 打印一些样本值以验证数据
                print(f"\n6. 数据样本:")
                print(f"   y_true_region_flat前5个值: {y_true_region_flat[:5]}")
                print(f"   region_scores_flat前5个值: {region_scores_flat[:5]}")
                print(f"   y_true_time_flat前5个值: {y_true_time_flat[:5]}")
                print(f"   time_scores_flat前5个值: {time_scores_flat[:5]}")
                
                # 计算AUC分数
                region_auc = roc_auc_score(y_true_region_flat, region_scores_flat)
                time_auc = roc_auc_score(y_true_time_flat, time_scores_flat)
                
                print(f"\n7. AUC分数:")
                print(f"   region_auc: {region_auc:.4f}")
                print(f"   time_auc: {time_auc:.4f}")
                
                return {
                    'region_auc': region_auc,
                    'time_auc': time_auc
                }
            except Exception as e:
                print(f"\n计算AUC时出错: {str(e)}")
                print(f"y_true_region_flat唯一值: {np.unique(y_true_region_flat)}")
                print(f"region_scores_flat唯一值: {np.unique(region_scores_flat)}")
                print(f"y_true_time_flat唯一值: {np.unique(y_true_time_flat)}")
                print(f"time_scores_flat唯一值: {np.unique(time_scores_flat)}")
                raise e
