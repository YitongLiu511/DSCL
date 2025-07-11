import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional

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
        
        # 对每个特征分别计算统计量
        x_masked = x.clone()  # 复制原始数据
        num_mask = min(int(T * self.temporal_mask_ratio), T)  # 确保不超过时间步数
        masked_indices = torch.zeros(N, num_mask, dtype=torch.long, device=self.device)
        
        for c in range(C):
            # 计算滑动窗口统计量
            x_feature = x[:,:,c]  # [N, T]
            x_feature_2 = x_feature ** 2
            
            # 使用卷积计算滑动平均
            filters = torch.ones(1, 1, self.window_size, device=self.device)
            ltr = F.conv1d(x_feature.unsqueeze(1), filters, padding=self.window_size-1)
            ltr2 = F.conv1d(x_feature_2.unsqueeze(1), filters, padding=self.window_size-1)
            
            # 归一化
            ltr[:,:,:self.window_size-1] /= torch.arange(1, self.window_size, device=self.device)
            ltr[:,:,self.window_size-1:] /= self.window_size
            ltr2[:,:,:self.window_size-1] /= torch.arange(1, self.window_size, device=self.device)
            ltr2[:,:,self.window_size-1:] /= self.window_size
            
            # 计算方差
            ltrd = (ltr2 - ltr ** 2)[:,:,:ltr.shape[-1]-self.window_size+1].squeeze(1)
            ltrm = ltr[:,:,:ltr.shape[-1]-self.window_size+1].squeeze(1)
            
            # 计算异常分数
            score = ltrd.sum(-1) / (ltrm.sum(-1) + 1e-6)
            
            # 选择掩码位置
            for n in range(N):
                if c == 0:  # 只在第一个特征时计算掩码位置
                    # 确保不会超出范围
                    score_len = score[n].numel() if score[n].dim() > 0 else 1
                    actual_num_mask = min(num_mask, score_len)
                    if actual_num_mask > 0:
                        masked_idx = score[n].topk(actual_num_mask, dim=0, sorted=False)[1]
                        masked_indices[n, :actual_num_mask] = masked_idx
                
                # 应用可学习掩码标记
                if c == 0 and masked_indices[n, 0] != 0:  # 确保有掩码位置
                    # 使用可学习的掩码标记替换被掩码的值
                    mask_token = self.temporal_mask_token[0, 0, :]  # [d_model]
                    # 将掩码标记投影到当前特征维度
                    mask_value = self.temporal_projection(mask_token.unsqueeze(0))[0, c]  # 标量
                    x_masked[n, masked_indices[n, :actual_num_mask], c] = mask_value
        
        return x_masked, masked_indices
    
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
        # 对每个特征分别做rfft和掩码
        x_masked = torch.zeros_like(x)
        mask_indices = torch.zeros((N, C, T//2+1), dtype=torch.bool, device=self.device)
        for c in range(C):
            # [N, T] -> [N, Freq]
            cx = torch.fft.rfft(x[:,:,c])
            mag = torch.abs(cx)  # [N, Freq]
            quantile = torch.quantile(mag, self.frequency_mask_ratio, dim=1, keepdim=True)  # [N, 1]
            mask = mag < quantile  # [N, Freq]
            mask_indices[:,c,:] = mask
            # 使用可学习掩码标记
            cx_masked = cx.clone()
            # 获取可学习的频率掩码标记
            freq_mask_token = self.frequency_mask_token[0, :, 0]  # [d_model]
            # 将掩码标记投影到合适的维度
            mask_values = self.frequency_projection(freq_mask_token.unsqueeze(-1).float())  # [d_model, 1]
            # 使用投影后的值替换被掩码的频率
            for n in range(N):
                if mask[n].any():  # 如果有掩码位置
                    # 选择掩码位置对应的值
                    mask_positions = torch.where(mask[n])[0]
                    for pos in mask_positions:
                        if pos < len(mask_values):
                            cx_masked[n, pos] = mask_values[pos, 0]
            # 逆变换
            ix = torch.fft.irfft(cx_masked, n=T)  # [N, T]
            x_masked[:,:,c] = ix
        return x_masked, mask_indices
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            x: 输入张量 [N, T, F]
        Returns:
            时间掩蔽后的张量、时间掩蔽位置、频率掩蔽后的张量、频率掩蔽位置
        """
        # 时间掩蔽
        temporal_masked_x, temporal_mask_indices = self.temporal_masking(x)
        
        # 频率掩蔽
        frequency_masked_x, frequency_mask_indices = self.frequency_masking(x)
        
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

