import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional
import os

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.05):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # 确保输入维度正确
        if len(x.shape) == 4:  # [B, T, N, F]
            B, T, N, F = x.shape
            x = x.reshape(B, T, -1)  # [B, T, N*F]
            x = self.value_embedding(x)  # [B, T, D]
        elif len(x.shape) == 3:  # [B, T, C]
            x = self.value_embedding(x)  # [B, T, D]
        else:
            x = self.value_embedding(x)
        return self.dropout(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:(d_model//2)])
        self.register_buffer('pe', pe)

    def forward(self, idx):
        return self.pe[idx]

class TemporalFrequencyMasking(nn.Module):
    def __init__(
        self,
        window_size: int,
        temporal_mask_ratio: float = 0.1,
        frequency_mask_ratio: float = 0.1,
        d_model: int = 263,
        n_features: int = 2,
        device: Optional[str] = None
    ):
        super().__init__()
        self.window_size = window_size
        self.temporal_mask_ratio = temporal_mask_ratio
        self.frequency_mask_ratio = frequency_mask_ratio
        self.d_model = d_model
        self.n_features = n_features
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 数据嵌入 - 输入维度为特征数
        self.emb = DataEmbedding(c_in=n_features, d_model=d_model)
        self.pos_emb = PositionalEmbedding(d_model=d_model)
        
        # 时间掩蔽的可学习参数
        self.temporal_mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.temporal_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # 频率掩蔽的可学习参数
        self.frequency_mask_token = nn.Parameter(torch.zeros(1, d_model, 1, dtype=torch.cfloat))
        self.frequency_projection = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # 将模型移动到指定设备
        self.to(self.device)
        
    def temporal_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MTFAE风格时域掩码，直接支持(N, T, F)输入（区域, 时间, 特征）
        Args:
            x: 输入张量 [N, T, F]
        Returns:
            掩蔽后的张量和掩蔽位置
        """
        x = x.to(self.device)
        if x.ndim != 3:
            raise ValueError(f"输入张量维度应为3，实际为{x.ndim}")
        N, T, C = x.shape
        ex = self.emb(x)  # [N, T, D]
        filters = torch.ones(1, 1, self.window_size, device=self.device)
        ex2 = ex ** 2
        ltr = F.conv1d(ex.transpose(1,2).reshape(-1, ex.shape[1]).unsqueeze(1), filters, padding=self.window_size-1)
        ltr[:,:,:self.window_size-1] /= torch.arange(1, self.window_size, device=self.device)
        ltr[:,:,self.window_size-1:] /= self.window_size
        ltr2 = F.conv1d(ex2.transpose(1,2).reshape(-1, ex2.shape[1]).unsqueeze(1), filters, padding=self.window_size-1)
        ltr2[:,:,:self.window_size-1] /= torch.arange(1, self.window_size, device=self.device)
        ltr2[:,:,self.window_size-1:] /= self.window_size
        ltrd = (ltr2 - ltr ** 2)[:,:,:ltr.shape[-1]-self.window_size+1].squeeze(1).reshape(ex.shape[0],ex.shape[-1],-1).transpose(1,2)
        ltrm = ltr[:,:,:ltr.shape[-1]-self.window_size+1].squeeze(1).reshape(ex.shape[0],ex.shape[-1],-1).transpose(1,2)
        score = ltrd.sum(-1) / (ltrm.sum(-1) + 1e-6)
        num_mask = int(T * self.temporal_mask_ratio)
        masked_indices = torch.zeros(N, num_mask, dtype=torch.long, device=self.device)
        for n in range(N):
            masked_idx = score[n].topk(num_mask, dim=0, sorted=False)[1]
            masked_indices[n] = masked_idx
        tokens = ex.clone()
        for n in range(N):
            tokens[n, masked_indices[n]] = self.temporal_mask_token + self.pos_emb(masked_indices[n])
        return tokens, masked_indices
    
    def frequency_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MTFAE风格频域掩码，直接支持(N, T, C)输入（区域, 时间, 特征）
        Args:
            x: 输入张量 [N, T, C]
        Returns:
            掩蔽后的张量和掩蔽位置
        """
        x = x.to(self.device)
        if x.ndim != 3:
            raise ValueError(f"输入张量维度应为3，实际为{x.ndim}")
        N, T, C = x.shape
        ex = self.emb(x)  # [N, T, D]
        cx = torch.fft.rfft(ex.transpose(1,2))  # [N, D, Freq]
        mag = torch.sqrt(cx.real ** 2 + cx.imag ** 2)  # [N, D, Freq]
        quantile = torch.quantile(mag, self.frequency_mask_ratio, dim=2, keepdim=True)
        idx = torch.argwhere(mag < quantile)
        mask_token = self.frequency_mask_token.repeat(N, cx.shape[1], mag.shape[-1])
        cx[mag < quantile] = mask_token[idx[:,0], idx[:,1], idx[:,2]]
        ix = torch.fft.irfft(cx).transpose(1,2)  # [N, T, D]
        return ix, mag < quantile
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            x: 输入张量 [B, T, N, F]
        Returns:
            时间掩蔽后的张量、时间掩蔽位置、频率掩蔽后的张量、频率掩蔽位置
        """
        # 时间掩蔽
        temporal_masked_x, temporal_mask_indices = self.temporal_masking(x)
        
        # 频率掩蔽
        frequency_masked_x, frequency_mask_indices = self.frequency_masking(x)
        
        # 定义输出目录
        output_dir = 'data/processed'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存掩码后的数据和索引到指定目录
        np.save(os.path.join(output_dir, 'tfm_temporal_masked_data.npy'), temporal_masked_x.detach().cpu().numpy())
        np.save(os.path.join(output_dir, 'tfm_frequency_masked_data.npy'), frequency_masked_x.detach().cpu().numpy())
        np.save(os.path.join(output_dir, 'tfm_temporal_mask_indices.npy'), temporal_mask_indices.detach().cpu().numpy())
        np.save(os.path.join(output_dir, 'tfm_frequency_mask_indices.npy'), frequency_mask_indices.detach().cpu().numpy())
        
        return temporal_masked_x, temporal_mask_indices, frequency_masked_x, frequency_mask_indices

def temporal_frequency_masking(x, temporal_mask_ratio=0.1, freq_mask_ratio=0.1):
    """
    同时应用时间掩码和频率掩码
    
    参数:
        x: 输入数据，形状为 [B, T, N, F, 2]
        temporal_mask_ratio: 时间掩码比例
        freq_mask_ratio: 频率掩码比例
    """
    batch_size, num_days, num_nodes, num_freq, num_features = x.shape
    
    # 合并14天和144个时间槽，保持特征维度
    x = x.reshape(batch_size, -1, num_nodes, num_features)  # [B, T*F, N, 2]
    
    # 应用时间掩码
    x = temporal_masking(x, temporal_mask_ratio)
    
    # 应用频率掩码
    x = frequency_masking(x, freq_mask_ratio)
    
    # 恢复原始形状
    x = x.reshape(batch_size, num_days, num_nodes, num_freq, num_features)
    
    return x

