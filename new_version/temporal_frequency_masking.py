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
        if len(x.shape) == 3:  # [B, T, C]
            B, T, C = x.shape
            x = x.reshape(-1, C)  # [B*T, C]
            x = self.value_embedding(x)  # [B*T, D]
            x = x.reshape(B, T, -1)  # [B, T, D]
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
        d_model: int = 512,
        device: Optional[str] = None
    ):
        """
        时频掩蔽模块
        
        Args:
            window_size: 时间掩蔽的窗口大小
            temporal_mask_ratio: 时间掩蔽的比例
            frequency_mask_ratio: 频率掩蔽的比例
            d_model: 模型维度
            device: 设备
        """
        super().__init__()
        self.window_size = window_size
        self.temporal_mask_ratio = temporal_mask_ratio
        self.frequency_mask_ratio = frequency_mask_ratio
        self.d_model = d_model
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 数据嵌入
        self.emb = DataEmbedding(c_in=263, d_model=d_model)
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
        # 修改频率投影层，使其能够处理单个值
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
        时间掩蔽
        
        Args:
            x: 输入张量 [B, T, C]
            
        Returns:
            掩蔽后的张量和掩蔽位置
        """
        # 确保输入在正确的设备上
        x = x.to(self.device)
        B, T, C = x.shape
        
        # 数据嵌入
        ex = self.emb(x)  # [B, T, D]
        ex = ex + self.pos_emb(torch.arange(T, device=self.device))
        
        # 计算滑动窗口统计
        filters = torch.ones(1, 1, self.window_size, device=self.device)
        ex2 = ex ** 2
        
        # 计算滑动窗口和
        ltr = F.conv1d(ex.transpose(1, 2).reshape(-1, ex.shape[1]).unsqueeze(1), 
                      filters, padding=self.window_size-1)
        ltr[:,:,:self.window_size-1] /= torch.arange(1, self.window_size, device=self.device)
        ltr[:,:,self.window_size-1:] /= self.window_size
        
        ltr2 = F.conv1d(ex2.transpose(1, 2).reshape(-1, ex.shape[1]).unsqueeze(1), 
                       filters, padding=self.window_size-1)
        ltr2[:,:,:self.window_size-1] /= torch.arange(1, self.window_size, device=self.device)
        ltr2[:,:,self.window_size-1:] /= self.window_size
        
        # 计算均值和方差
        ltrd = (ltr2 - ltr ** 2)[:,:,:ltr.shape[-1]-self.window_size+1].squeeze(1)
        ltrd = ltrd.reshape(ex.shape[0], ex.shape[-1], -1).transpose(1, 2)
        ltrm = ltr[:,:,:ltr.shape[-1]-self.window_size+1].squeeze(1)
        ltrm = ltrm.reshape(ex.shape[0], ex.shape[-1], -1).transpose(1, 2)
        
        score = ltrd.sum(-1) / (ltrm.sum(-1) + 1e-6)
        
        # 选择掩码位置 - 修改为考虑所有天数
        num_mask = int(T * self.temporal_mask_ratio)  # 每天掩码的时间戳数量
        masked_indices = torch.zeros(B, num_mask, dtype=torch.long, device=self.device)
        for b in range(B):
            # 对每天选择num_mask个时间戳进行掩码
            day_masked_idx = score[b].topk(num_mask, dim=0, sorted=False)[1]
            masked_indices[b] = day_masked_idx
        
        # 创建掩码张量
        masked_x = ex.clone()
        for b in range(B):
            masked_x[b, masked_indices[b]] = self.temporal_mask_token
            
        # 对未掩码位置进行投影
        unmasked_indices = torch.ones(B, T, dtype=torch.bool, device=self.device)
        for b in range(B):
            unmasked_indices[b, masked_indices[b]] = False
            
        unmasked_values = masked_x[unmasked_indices]
        projected_values = self.temporal_projection(unmasked_values)
        masked_x[unmasked_indices] = projected_values
        
        return masked_x, masked_indices

    def frequency_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        频率掩蔽
        
        Args:
            x: 输入张量 [B, T, C]
            
        Returns:
            掩蔽后的张量和掩蔽位置
        """
        # 确保输入在正确的设备上
        x = x.to(self.device)
        B, T, C = x.shape
        print(f"输入张量形状: {x.shape}")
        
        # 数据嵌入
        ex = self.emb(x)  # [B, T, D]
        print(f"嵌入后形状: {ex.shape}")
        ex = ex + self.pos_emb(torch.arange(T, device=self.device))
        
        # FFT变换
        cx = torch.fft.rfft(ex.transpose(1, 2))  # [B, D, T//2 + 1]
        print(f"FFT后形状: {cx.shape}")
        
        # 计算幅度
        mag = torch.sqrt(cx.real ** 2 + cx.imag ** 2)  # [B, D, T//2 + 1]
        
        # 选择掩码位置 - 修改为考虑所有天数
        num_mask = int(T * self.frequency_mask_ratio)  # 每天掩码的时间戳数量
        masked_indices = torch.zeros(B, num_mask, dtype=torch.long, device=self.device)
        for b in range(B):
            # 对每天选择num_mask个频率分量进行掩码
            day_mag = mag[b].mean(dim=0)  # 对每个频率分量取平均
            day_masked_idx = day_mag.topk(num_mask, dim=0, sorted=False)[1]
            masked_indices[b] = day_masked_idx
        
        # 应用掩码
        masked_fft = cx.clone()
        for b in range(B):
            masked_fft[b, :, masked_indices[b]] = self.frequency_mask_token.repeat(1, 1, 1)
        
        # 逆FFT变换
        masked_x = torch.fft.irfft(masked_fft).transpose(1, 2)  # [B, T, D]
        print(f"逆FFT后形状: {masked_x.shape}")
        
        # 将频率域掩码转换为时域掩码
        time_domain_mask = torch.zeros(B, T, C, dtype=torch.bool, device=self.device)
        for b in range(B):
            for c in range(C):
                # 获取当前区域的频率掩码
                freq_mask = torch.zeros(T//2 + 1, dtype=torch.bool, device=self.device)
                freq_mask[masked_indices[b]] = True
                # 将频率掩码转换为时域掩码
                time_domain_mask[b, :, c] = torch.fft.irfft(freq_mask.float()).bool()
        print(f"时域掩码形状: {time_domain_mask.shape}")
        
        # 对未掩码位置进行投影
        # 首先将masked_x转换回原始维度
        masked_x = self.emb.value_embedding.weight.T @ masked_x.transpose(1, 2)  # [B, C, T]
        masked_x = masked_x.transpose(1, 2)  # [B, T, C]
        print(f"转换回原始维度后形状: {masked_x.shape}")
        
        # 获取未掩码位置的值
        unmasked_values = masked_x[~time_domain_mask]  # [N]
        print(f"未掩码值形状: {unmasked_values.shape}")
        
        # 对未掩码值进行投影
        projected_values = self.frequency_projection(unmasked_values.unsqueeze(-1))  # [N, 1]
        print(f"投影后形状: {projected_values.shape}")
        
        # 将投影后的值放回原位置
        masked_x[~time_domain_mask] = projected_values.squeeze(-1)
        
        return masked_x, masked_indices

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, T, C]
            
        Returns:
            时间掩蔽后的张量、时间掩蔽位置、频率掩蔽后的张量、频率掩蔽位置
        """
        # 时间掩蔽
        temporal_masked_x, temporal_mask_indices = self.temporal_masking(x)
        
        # 频率掩蔽
        frequency_masked_x, frequency_mask_indices = self.frequency_masking(x)
        
        return temporal_masked_x, temporal_mask_indices, frequency_masked_x, frequency_mask_indices 