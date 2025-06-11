import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
import math

from .tsfm import TemporalTransformer
from .module import SingleGCN, MultiheadAttention, SoftClusterLayer, MultipleGCN
from .embed import TemporalEmbedding
from .revin import RevIN
from .contrastive import compute_contrastive_loss, compute_anomaly_score, compute_adversarial_contrastive_loss


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
        self.patch_size = patch_size
        
        # 初始化编码器输出
        self.time_encoder_output = None
        self.freq_encoder_output = None
        
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
        
            # 保存最后一个编码器的输出
            if i == len(self.time_encoder) - 1:
                # 创建完整的token张量
                full_tokens = torch.zeros(bs, num_patch, n_vars, self.d_model, device=device)
                # 获取未掩码位置的索引
                unmasked_indices = torch.where(patch_mask.reshape(-1))[0]
                # 将编码后的token放入正确的位置
                full_tokens.reshape(-1, self.d_model)[unmasked_indices] = encoded_tokens.reshape(-1, self.d_model)
                # 保存编码器输出
                self.time_encoder_output = full_tokens
                print(f"    保存时间编码器输出维度: {self.time_encoder_output.shape}")
        
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
        tokens.reshape(-1, self.d_model)[unmasked_indices] = encoded_tokens.reshape(-1, self.d_model)
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
        
        print(f"\n频率掩码处理维度追踪:")
        print(f"1. 输入维度: {x.shape}")
        
        # 计算每个patch的变异系数
        cv = torch.std(x, dim=3) / (torch.mean(x, dim=3) + 1e-6)  # [bs, num_patch, n_vars]
        print(f"2. 变异系数维度: {cv.shape}")
        
        # 对每个变量选择变异系数最大的patch进行掩码
        patch_mask = torch.ones((bs, num_patch, n_vars), dtype=torch.bool, device=device)  # [bs, num_patch, n_vars]
        num_masks = int(num_patch * self.freq_ratio)
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
        masked_tokens = self.freq_mask_token.repeat(bs, num_masks, n_vars, 1)
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
        for i, encoder in enumerate(self.freq_encoder):
            print(f"\n10. 编码器 {i+1} 输入维度: {encoded_tokens.shape}")
            encoded_tokens, _ = encoder(encoded_tokens)
            print(f"    编码器 {i+1} 输出维度: {encoded_tokens.shape}")
            
            # 保存最后一个编码器的输出
            if i == len(self.freq_encoder) - 1:
                # 创建完整的token张量
                full_tokens = torch.zeros(bs, num_patch, n_vars, self.d_model, device=device)
                # 获取未掩码位置的索引
                unmasked_indices = torch.where(patch_mask.reshape(-1))[0]
                # 将编码后的token放入正确的位置
                full_tokens.reshape(-1, self.d_model)[unmasked_indices] = encoded_tokens.reshape(-1, self.d_model)
                # 保存编码器输出
                self.freq_encoder_output = full_tokens
                print(f"    保存频率编码器输出维度: {self.freq_encoder_output.shape}")
        
        # 组合编码后的token和掩码token
        tokens = torch.zeros(bs, num_patch, n_vars, self.d_model, device=device)
        print(f"\n12. 组合前维度检查:")
        print(f"12.1 tokens维度: {tokens.shape}")
        print(f"12.2 patch_mask维度: {patch_mask.shape}")
        print(f"12.3 encoded_tokens维度: {encoded_tokens.shape}")
        print(f"12.4 freq_mask_token维度: {self.freq_mask_token.shape}")
        
        # 创建索引映射
        unmasked_indices = torch.where(patch_mask.reshape(-1))[0]  # [num_unmasked]
        print(f"\n13. unmasked_indices维度: {unmasked_indices.shape}")
        
        # 使用索引赋值
        tokens.reshape(-1, self.d_model)[unmasked_indices] = encoded_tokens.reshape(-1, self.d_model)
        tokens.reshape(-1, self.d_model)[~unmasked_indices] = self.freq_mask_token
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
        
    def forward(self, x):
        if self.training:
            print("\n=== 开始时间-频率掩码处理 ===")
            # 应用时间掩码
            x_time_masked, time_mask = self.time_masking(x)
            print("时间掩码处理完成")
            
            # 应用频域掩码
            x_freq_masked, freq_mask = self.freq_masking(x)
            print("频率掩码处理完成")
            
            # 组合两种掩码的结果
            x_masked = x_time_masked * 0.5 + x_freq_masked * 0.5
            
            # 保存掩码信息用于后续重建
            self.time_mask = time_mask
            self.freq_mask = freq_mask
            
            print("掩码处理完成，返回组合结果")
            return x_masked, self.time_encoder_output, self.freq_encoder_output
        else:
            return x, None, None


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
        
        # 频率掩码编码器
        self.freq_encoder = nn.ModuleList([
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
        
            # 保存最后一个编码器的输出
            if i == len(self.time_encoder) - 1:
                # 创建完整的token张量
                full_tokens = torch.zeros(bs, num_patch, n_vars, self.d_model, device=device)
                # 获取未掩码位置的索引
                unmasked_indices = torch.where(patch_mask.reshape(-1))[0]
                # 将编码后的token放入正确的位置
                full_tokens.reshape(-1, self.d_model)[unmasked_indices] = encoded_tokens.reshape(-1, self.d_model)
                # 保存编码器输出
                self.time_encoder_output = full_tokens
                print(f"    保存时间编码器输出维度: {self.time_encoder_output.shape}")
        
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
        tokens.reshape(-1, self.d_model)[unmasked_indices] = encoded_tokens.reshape(-1, self.d_model)
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
        """
        前向传播
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, input_dim]
        Returns:
            重构后的张量，形状为 [batch_size, seq_len, input_dim]
        """
        # 打印输入形状
        print(f"输入形状: {x.shape}")
        
        # 输入投影
        x = self.input_projection(x)
        print(f"投影后形状: {x.shape}")
        
        # 空间Transformer处理
        spatial_out = self.spatial_transformer(x)
        print(f"空间Transformer输出形状: {spatial_out.shape}")
        
        # 时间Transformer处理
        temporal_out = self.temporal_transformer(spatial_out)
        print(f"时间Transformer输出形状: {temporal_out.shape}")
        
        # 输出投影
        output = self.output_projection(temporal_out)
        print(f"最终输出形状: {output.shape}")
        
        return output


class STPatchMaskFormer(STPatchFormer):
    def __init__(self, c_in, seq_len, patch_len, stride, max_seq_len, n_layers, d_model, n_heads, d_ff, 
                 shared_embedding=True, attn_dropout=0., dropout=0., act='gelu', mask_ratio: float = 0.4, 
                 time_ratio: float = 0.5, freq_ratio: float = 0.4, patch_size: int = 12,
                 n_clusters: int = 8, temperature: float = 0.07, n_gcn: int = 3,
                 poi_sim=None, dist_mat=None):
        super().__init__(c_in, seq_len, patch_len, stride, max_seq_len, n_layers, d_model, n_heads, d_ff,
                        shared_embedding, attn_dropout, dropout, act)
        
        # 初始化掩码模块
        self.time_freq_masking = TimeFreqMasking(
            mask_ratio=mask_ratio,
            time_ratio=time_ratio,
            freq_ratio=freq_ratio,
            patch_size=patch_size
        )
        
        # 初始化异常检测器
        self.anomaly_detector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        # 初始化区域检测器
        self.region_detector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        # 初始化时间检测器
        self.time_detector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        # 添加输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, c_in)
        )
        
        # 其他初始化代码保持不变
        self.n_clusters = n_clusters
        self.temperature = temperature
        self.n_gcn = n_gcn
        self.poi_sim = poi_sim
        self.dist_mat = dist_mat
        
        # 初始化num_patch属性
        self.num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
        
        # 使用新的 DynamicTimeFreqMasking 替换原来的 TimeFreqMasking
        self.mask = DynamicTimeFreqMasking(mask_ratio, time_ratio, freq_ratio, patch_size, d_model, n_heads, n_layers)
        
        # 添加空间异常检测组件，传入POI和距离信息
        self.soft_cluster = SoftClusterLayer(
            c_in=d_model,
            nmb_prototype=n_clusters,
            tau=temperature,
            poi_sim=poi_sim,
            dist_mat=dist_mat
        )
        
        # 创建一个默认的邻接矩阵
        default_adj = torch.ones(c_in, c_in) / c_in
        default_adj = default_adj.unsqueeze(0)
        
        self.multiple_gcn = MultipleGCN(
            in_channels=d_model,
            out_channels=d_model,
            matrices=default_adj,
            n_layers=n_gcn
        )
        
        # 添加训练相关的组件
        self.criterion = nn.MSELoss()
        
        # 导入对比损失函数
        self.compute_contrastive_loss = compute_contrastive_loss
        self.compute_anomaly_score = compute_anomaly_score
        
        # 添加空间特征投影层
        self.spatial_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # 添加特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # 添加空间异常检测器
        self.spatial_anomaly_detector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # 添加区域特征存储
        self.region_features = None
        self.fusion_features = None
        self.projection = None
        
    def forward(self, x, return_dict=False):
        """
        前向传播
        Args:
            x: 输入数据 [batch_size, seq_len, n_vars]
            return_dict: 是否返回字典格式的输出
        Returns:
            如果return_dict为True，返回包含重建结果和异常分数的字典
            否则返回重建结果
        """
        print("\n" + "="*50)
        print("STPatchMaskFormer forward方法开始:")
        print(f"1. 输入数据维度: {x.shape}")
        print(f"2. 模型训练状态: {self.training}")
        print(f"3. 模型模式: {'训练模式' if self.training else '评估模式'}")
        print("="*50 + "\n")

        # 保存原始训练状态
        original_training = self.training
        print(f"original_training的值: {original_training}")

        print("\nSTPatchMaskFormer前向传播维度追踪:")
        print(f"1. 输入数据维度: {x.shape}")
        
        # 分片处理
        x = self.patch(x)
        print(f"2. 分片后维度: {x.shape}")
        
        # 归一化
        x = self.revin(x.transpose(2, 3), 'norm').transpose(2, 3)
        print(f"3. 归一化后维度: {x.shape}")
        
        # 根据训练状态决定是否应用掩码
        if self.training:
            print("\n=== 训练模式 - 应用掩码 ===")
            print(f"1. 掩码前模型状态: {self.training}")
            print(f"2. 掩码前模型模式: {'训练模式' if self.training else '评估模式'}")
            
            # 应用时间-频率掩码
            x, time_encoder_output, freq_encoder_output = self.mask(x)
            
            print("\n掩码分支输出检查:")
            print(f"1. 时间掩码分支输出是否存在: {time_encoder_output is not None}")
            print(f"2. 频率掩码分支输出是否存在: {freq_encoder_output is not None}")
            if time_encoder_output is not None:
                print(f"3. 时间掩码分支输出维度: {time_encoder_output.shape}")
            if freq_encoder_output is not None:
                print(f"4. 频率掩码分支输出维度: {freq_encoder_output.shape}")
        else:
            print("\n=== 评估模式 - 不应用掩码 ===")
            print(f"1. 评估模式下的模型状态: {self.training}")
            print(f"2. 评估模式下的模型模式: {'训练模式' if self.training else '评估模式'}")
            time_encoder_output = None
            freq_encoder_output = None

        print("\n=== 开始分片编码 ===")
        print(f"1. 编码前模型状态: {self.training}")
        print(f"2. 编码前模型模式: {'训练模式' if self.training else '评估模式'}")
        
        # 分片编码
        x = self.patch_tsfm(x)
        print(f"3. 编码后维度: {x.shape}")
        print(f"4. 编码后模型状态: {self.training}")
        print(f"5. 编码后模型模式: {'训练模式' if self.training else '评估模式'}")

        print("\n=== 开始空间异常检测 ===")
        print(f"1. 检测前模型状态: {self.training}")
        print(f"2. 检测前模型模式: {'训练模式' if self.training else '评估模式'}")
        
        # 空间异常检测
        self.region_features = x  # 保存区域特征
        z = self.spatial_anomaly_detector(x)
        
        # 特征融合
        if self.region_features is not None:
            self.fusion_features = self.fusion_layer(
                torch.cat([self.region_features, z], dim=-1)
            )
        else:
            self.fusion_features = z
            
        # 特征投影
        self.projection = self.spatial_proj(self.fusion_features)
        
        # 调整维度
        z = self.projection.reshape(self.projection.shape[0], -1, self.projection.shape[-1])
        z = z[:, :self.seq_len]
        
        print(f"3. 区域特征维度: {self.region_features.shape}")
        print(f"4. 空间特征维度: {x.shape}")
        print(f"5. 融合特征维度: {self.fusion_features.shape}")
        print(f"6. 投影后维度: {self.projection.shape}")
        print(f"7. 维度重排后: {self.projection.permute(0, 2, 1, 3).shape}")
        print(f"8. 还原分片后: {self.projection.reshape(self.projection.shape[0], -1, self.projection.shape[-1]).shape}")
        print(f"9. 截取长度后: {z.shape}")
        print(f"10. 最终输出维度: {self.projection.reshape(self.projection.shape[0], -1, self.projection.shape[-1])[:, :self.seq_len].shape}")
        print(f"11. 调整后z维度: {z.shape}")
        
        # 调整x的维度以匹配z
        x_reshaped = x.reshape(x.shape[0], -1, x.shape[-1])
        x_reshaped = x_reshaped[:, :self.seq_len]
        
        print(f"12. 调整后x维度: {x_reshaped.shape}")
        
        # 计算重建损失
        recon_loss = F.mse_loss(z, x_reshaped)

            # 计算异常分数
        anomaly_scores = self.anomaly_detector(z)  # [bs, seq_len, d_model]
        region_scores = self.region_detector(z)  # [bs, d_model]
        time_scores = self.time_detector(x)  # [bs, seq_len]
        
        # 调整维度以匹配
        region_scores = region_scores.unsqueeze(1).unsqueeze(2)  # [bs, 1, 1, d_model]
        time_scores = time_scores.unsqueeze(-1)  # [bs, seq_len, 1]
        
        # 计算最终的异常分数
        final_scores = region_scores * time_scores  # [bs, seq_len, d_model]
        
        print("\n准备检查original_training条件")
        print(f"1. original_training值: {original_training}")
        print(f"2. 当前模型状态: {self.training}")
        print(f"3. 当前模型模式: {'训练模式' if self.training else '评估模式'}")
            
        # 如果是在训练模式下，计算对抗对比损失
        if original_training:
            print("\n=== 开始计算对抗对比损失 ===")
            print("1. 检查时间编码器输出:")
            print(f"   - 是否存在: {time_encoder_output is not None}")
            if time_encoder_output is not None:
                print(f"   - 维度: {time_encoder_output.shape}")
            
            print("\n2. 检查频率编码器输出:")
            print(f"   - 是否存在: {freq_encoder_output is not None}")
            if freq_encoder_output is not None:
                print(f"   - 维度: {freq_encoder_output.shape}")
            
            # 确保模型处于训练模式
            self.train()
            
            # 将编码器输出转换为列表
            time_features = [time_encoder_output] if time_encoder_output is not None else []
            freq_features = [freq_encoder_output] if freq_encoder_output is not None else []
            
                # 计算对抗对比损失
            total_loss, adv_loss, con_loss = compute_adversarial_contrastive_loss(
                time_features, freq_features
            )
            
            print("\n3. 对抗对比损失计算结果:")
            print(f"   - 总损失值: {total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss}")
            print(f"   - 对抗损失值: {adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss}")
            print(f"   - 对比损失值: {con_loss.item() if isinstance(con_loss, torch.Tensor) else con_loss}")
            
            # 将对抗对比损失添加到总损失中
            recon_loss = recon_loss + total_loss
            
            print("\n4. 更新后的总损失:")
            print(f"   - 损失值: {recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss}")

        print("\n7. 异常分数计算:")
        print(f"   anomaly_scores维度: {anomaly_scores.shape}")
        print(f"   region_scores维度: {region_scores.shape}")
        print(f"   time_scores维度: {time_scores.shape}")
            
        print("\n=== forward方法结束 ===")
        print(f"1. 最终模型状态: {self.training}")
        print(f"2. 最终模型模式: {'训练模式' if self.training else '评估模式'}")

        if return_dict:
            return {
                'z': z,
                'x': x,
                'time_encoder_output': time_encoder_output,
                'freq_encoder_output': freq_encoder_output,
                'anomaly_scores': final_scores,
                'region_scores': region_scores,
                'time_scores': time_scores
            }
        else:
            return self.output_projection(z)

    def evaluate(self, x, y_true):
        """评估模型性能"""
        self.eval()
        with torch.no_grad():
            # 获取模型输出
            output_dict = self(x, return_dict=True)
            anomaly_scores = output_dict['anomaly_scores']  # [N, T, 2]
            region_scores = anomaly_scores[:, :, 0]        # [N, T]
            time_scores = anomaly_scores[:, :, 1]          # [N, T]
            
            # 转换为numpy数组
            y_true = y_true.cpu().numpy()
            region_scores = region_scores.cpu().numpy()
            time_scores = time_scores.cpu().numpy()
            
            # 调整维度以匹配
            y_true_region = y_true[:, :, :, 0]  # [N, T, D]
            y_true_time = y_true[:, :, :, 1]    # [N, T, D]
            
            # 确保region_scores和time_scores的维度与y_true匹配
            region_scores = region_scores.reshape(region_scores.shape[0], -1)  # [N, T*D]
            time_scores = time_scores.reshape(time_scores.shape[0], -1)        # [N, T*D]
            
            # 展平所有数组
            y_true_region_flat = y_true_region.reshape(-1)
            region_scores_flat = region_scores.reshape(-1)
            y_true_time_flat = y_true_time.reshape(-1)
            time_scores_flat = time_scores.reshape(-1)
            
            # 确保所有数组长度相同
            min_length = min(len(y_true_region_flat), len(region_scores_flat))
            y_true_region_flat = y_true_region_flat[:min_length]
            region_scores_flat = region_scores_flat[:min_length]
            y_true_time_flat = y_true_time_flat[:min_length]
            time_scores_flat = time_scores_flat[:min_length]
            
            # 计算AUC
            try:
                region_auc = roc_auc_score(y_true_region_flat, region_scores_flat)
                time_auc = roc_auc_score(y_true_time_flat, time_scores_flat)
            except Exception as e:
                print(f"计算AUC时出错: {str(e)}")
                print(f"y_true_region_flat长度: {len(y_true_region_flat)}")
                print(f"region_scores_flat长度: {len(region_scores_flat)}")
                print(f"y_true_time_flat长度: {len(y_true_time_flat)}")
                print(f"time_scores_flat长度: {len(time_scores_flat)}")
                raise e
            
            return {
                'region_auc': region_auc,
                'time_auc': time_auc
            }
