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
    
    def multi_feature_fusion(self, x: torch.Tensor) -> torch.Tensor:
        """
        直接做embedding和位置编码，不做加权融合
        Args:
            x: 输入张量 [N, T, F]
        Returns:
            融合后的特征 [N, T, d_model]
        """
        N, T, F = x.shape
        embedded_x = self.emb(x)  # [N, T, d_model]
        pos_emb = self.pos_emb(torch.arange(T, device=self.device))  # [T, d_model]
        embedded_x = embedded_x + pos_emb.unsqueeze(0)  # [N, T, d_model]
        return embedded_x
    
    def temporal_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        改进的时域掩码，参考MTFAE模型在嵌入空间中进行多特征融合
        Args:
            x: 输入张量 [N, T, F]
        Returns:
            掩蔽后的张量和掩蔽位置
        """
        x = x.to(self.device)
        if x.ndim != 3:
            raise ValueError(f"输入张量维度应为3，实际为{x.ndim}")
        N, T, C = x.shape
        
        # 1. 使用高级多特征融合方法
        x_embedded = self.multi_feature_fusion(x)  # [N, T, d_model] - 多特征融合到嵌入空间
        
        # 2. 在嵌入空间中计算统计量（参考MTFAE的TemEnc）
        x_embedded_2 = x_embedded ** 2  # [N, T, d_model]
        
        # 3. 使用卷积计算滑动窗口统计量（在嵌入空间中进行）
        # 修复：需要重塑数据以匹配卷积的输入格式
        x_embedded_reshaped = x_embedded.transpose(1, 2)  # [N, d_model, T]
        x_embedded_2_reshaped = x_embedded_2.transpose(1, 2)  # [N, d_model, T]
        
        # 对每个d_model维度分别进行卷积
        ltr_list = []
        ltr2_list = []
        
        for d in range(x_embedded_reshaped.shape[1]):  # 遍历d_model维度
            # 提取当前维度的数据 [N, 1, T]
            x_d = x_embedded_reshaped[:, d:d+1, :]
            x_d_2 = x_embedded_2_reshaped[:, d:d+1, :]
            
            # 创建卷积核
            filters = torch.ones(1, 1, self.window_size, device=self.device)
            
            # 执行卷积
            ltr_d = F.conv1d(x_d, filters, padding=self.window_size-1)  # [N, 1, T+window_size-1]
            ltr2_d = F.conv1d(x_d_2, filters, padding=self.window_size-1)  # [N, 1, T+window_size-1]
            
            # 归一化处理
            ltr_d[:,:,:self.window_size-1] /= torch.arange(1, self.window_size, device=self.device).unsqueeze(0).unsqueeze(0)
            ltr_d[:,:,self.window_size-1:] /= self.window_size
            ltr2_d[:,:,:self.window_size-1] /= torch.arange(1, self.window_size, device=self.device).unsqueeze(0).unsqueeze(0)
            ltr2_d[:,:,self.window_size-1:] /= self.window_size
            
            ltr_list.append(ltr_d)
            ltr2_list.append(ltr2_d)
        
        # 合并所有维度的结果
        ltr = torch.cat(ltr_list, dim=1)  # [N, d_model, T+window_size-1]
        ltr2 = torch.cat(ltr2_list, dim=1)  # [N, d_model, T+window_size-1]
        
        # 4. 计算方差（在嵌入空间中进行）
        ltrd = (ltr2 - ltr ** 2)[:,:,:ltr.shape[-1]-self.window_size+1]  # [N, d_model, T]
        ltrm = ltr[:,:,:ltr.shape[-1]-self.window_size+1]  # [N, d_model, T]
        
        # 5. 计算异常分数（在嵌入空间中进行，然后聚合）
        score = ltrd.sum(dim=1) / (ltrm.sum(dim=1) + 1e-6)  # [N, T] - 每个节点每个时间步的异常分数
        
        # 6. 选择掩码位置
        x_masked = x.clone()  # 复制原始数据
        num_mask = min(int(T * self.temporal_mask_ratio), T)
        masked_indices = torch.zeros(N, num_mask, dtype=torch.long, device=self.device)
        
        for n in range(N):
            actual_num_mask = min(num_mask, score.shape[1])
            if actual_num_mask > 0:
                masked_idx = score[n].topk(actual_num_mask, dim=0, sorted=False)[1]
                masked_indices[n, :actual_num_mask] = masked_idx
        
        # 7. 应用可学习掩码标记（对每个特征分别应用）
        for n in range(N):
            if masked_indices[n, 0] != 0:
                for c in range(C):
                    # 使用可学习的掩码标记替换被掩码的值
                    mask_token = self.temporal_mask_token[0, 0, :]  # [d_model]
                    # 将掩码标记投影到当前特征维度
                    mask_value = self.temporal_projection(mask_token.unsqueeze(0))[0, c]  # 标量
                    x_masked[n, masked_indices[n, :actual_num_mask], c] = mask_value
        
        return x_masked, masked_indices
    
    def frequency_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        改进的频域掩码，参考MTFAE模型在嵌入空间中进行多特征融合
        Args:
            x: 输入张量 [N, T, C]
        Returns:
            掩蔽后的张量和掩蔽位置
        """
        x = x.to(self.device)
        if x.ndim != 3:
            raise ValueError(f"输入张量维度应为3，实际为{x.ndim}")
        N, T, C = x.shape
        
        # 1. 使用高级多特征融合方法
        x_embedded = self.multi_feature_fusion(x)  # [N, T, d_model] - 多特征融合到嵌入空间
        
        # 2. 在嵌入空间中进行频域变换（参考MTFAE的FreEnc）
        # 转换为频域: [N, T, d_model] -> [N, d_model, Freq]
        cx = torch.fft.rfft(x_embedded.transpose(1, 2))  # [N, d_model, Freq]
        
        # 3. 计算频域幅度（在嵌入空间中进行）
        mag = torch.sqrt(cx.real ** 2 + cx.imag ** 2)  # [N, d_model, Freq]
        
        # 4. 基于分位数的掩码选择（在嵌入空间中进行）
        quantile = torch.quantile(mag, self.frequency_mask_ratio, dim=2, keepdim=True)  # [N, d_model, 1]
        mask = mag < quantile  # [N, d_model, Freq]
        
        # 5. 使用可学习掩码标记
        cx_masked = cx.clone()
        freq_mask_token = self.frequency_mask_token[0, :, 0]  # [d_model]
        # 修复：正确处理复数到实数的转换，避免警告
        if freq_mask_token.is_complex():
            freq_mask_token_real = freq_mask_token.real
        else:
            freq_mask_token_real = freq_mask_token
        mask_values = self.frequency_projection(freq_mask_token_real.unsqueeze(-1))  # [d_model, 1]
        
        # 6. 替换被掩码的频率分量
        for n in range(N):
            if mask[n].any():  # 如果有掩码位置
                mask_positions = torch.where(mask[n])
                for d_model_idx, freq_idx in zip(mask_positions[0], mask_positions[1]):
                    if d_model_idx < len(mask_values):
                        cx_masked[n, d_model_idx, freq_idx] = mask_values[d_model_idx, 0]
        
        # 7. 逆傅里叶变换恢复时域信号（在嵌入空间中）
        ix = torch.fft.irfft(cx_masked, n=T)  # [N, d_model, T]
        x_masked_embedded = ix.transpose(1, 2)  # [N, T, d_model]
        
        # 8. 将嵌入空间的掩码结果投影回原始特征空间
        # 简化处理：直接对原始特征进行频域掩码
        x_masked = torch.zeros_like(x)
        
        # 对每个特征应用掩码
        for c in range(C):
            # 对原始特征进行频域变换
            cx_orig = torch.fft.rfft(x[:,:,c])  # [N, Freq]
            mag_orig = torch.abs(cx_orig)  # [N, Freq]
            
            # 基于分位数的掩码选择
            quantile_orig = torch.quantile(mag_orig, self.frequency_mask_ratio, dim=1, keepdim=True)
            mask_orig = mag_orig < quantile_orig
            
            # 应用掩码
            cx_masked_orig = cx_orig.clone()
            for n in range(N):
                if mask_orig[n].any():
                    mask_positions = torch.where(mask_orig[n])[0]
                    for pos in mask_positions:
                        if pos < len(mask_values):
                            cx_masked_orig[n, pos] = mask_values[pos % len(mask_values), 0]
            
            # 逆变换
            ix_orig = torch.fft.irfft(cx_masked_orig, n=T)  # [N, T]
            x_masked[:,:,c] = ix_orig
        
        # 保存掩码位置信息
        mask_indices = mask.float().mean(dim=1)  # [N, Freq] - 平均掩码位置
        
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

