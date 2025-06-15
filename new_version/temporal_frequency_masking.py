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
        
        # 数据嵌入 - 输入维度为 区域数*特征数
        self.emb = DataEmbedding(c_in=d_model * n_features, d_model=d_model)
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
        时间掩蔽 - 同时考虑所有特征
        Args:
            x: 输入张量 [B, T, N, F]
        Returns:
            掩蔽后的张量和掩蔽位置
        """
        x = x.to(self.device)
        B, T, N, f = x.shape
        
        # 合并所有特征进行嵌入
        x_reshaped = x.reshape(B, T, -1)  # [B, T, N*f]
        ex = self.emb(x_reshaped)  # [B, T, D]
        ex = ex + self.pos_emb(torch.arange(T, device=self.device))
        
        # 计算滑动窗口统计
        filters = torch.ones(1, 1, self.window_size, device=self.device)
        ex2 = ex ** 2
        
        # 计算滑动窗口和
        ex_reshaped = ex.transpose(1, 2).reshape(-1, ex.shape[1]).unsqueeze(1)  # [B*D, 1, T]
        ltr = F.conv1d(ex_reshaped, filters, padding=self.window_size-1)
        ltr[:,:,:self.window_size-1] /= torch.arange(1, self.window_size, device=self.device)
        ltr[:,:,self.window_size-1:] /= self.window_size
        
        ex2_reshaped = ex2.transpose(1, 2).reshape(-1, ex2.shape[1]).unsqueeze(1)  # [B*D, 1, T]
        ltr2 = F.conv1d(ex2_reshaped, filters, padding=self.window_size-1)
        ltr2[:,:,:self.window_size-1] /= torch.arange(1, self.window_size, device=self.device)
        ltr2[:,:,self.window_size-1:] /= self.window_size
        
        # 计算均值和方差
        ltrd = (ltr2 - ltr ** 2)[:,:,:ltr.shape[-1]-self.window_size+1].squeeze(1)
        ltrd = ltrd.reshape(ex.shape[0], ex.shape[-1], -1).transpose(1, 2)
        ltrm = ltr[:,:,:ltr.shape[-1]-self.window_size+1].squeeze(1)
        ltrm = ltrm.reshape(ex.shape[0], ex.shape[-1], -1).transpose(1, 2)
        
        score = ltrd.sum(-1) / (ltrm.sum(-1) + 1e-6)
        
        # 选择掩码位置
        num_mask = int(T * self.temporal_mask_ratio)
        masked_indices = torch.zeros(B, num_mask, dtype=torch.long, device=self.device)
        for b in range(B):
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
        
        # 将掩码后的嵌入转换回原始特征空间
        masked_x = self.emb.value_embedding.weight.T @ masked_x.transpose(1, 2)
        masked_x = masked_x.transpose(1, 2)
        masked_x = masked_x.reshape(B, T, N, f)
        
        return masked_x, masked_indices
    
    def frequency_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        频率掩蔽 - 同时考虑所有特征
        Args:
            x: 输入张量 [B, T, N, F]
        Returns:
            掩蔽后的张量和掩蔽位置
        """
        x = x.to(self.device)
        B, T, N, f = x.shape
        
        # 合并所有特征进行嵌入
        x_reshaped = x.reshape(B, T, -1)  # [B, T, N*f]
        ex = self.emb(x_reshaped)  # [B, T, D]
        ex = ex + self.pos_emb(torch.arange(T, device=self.device))
        
        # FFT变换
        cx = torch.fft.rfft(ex.transpose(1, 2))  # [B, D, T//2 + 1]
        
        # 计算幅度
        mag = torch.sqrt(cx.real ** 2 + cx.imag ** 2)  # [B, D, T//2 + 1]
        
        # 选择掩码位置
        num_mask = int(T * self.frequency_mask_ratio)
        masked_indices = torch.zeros(B, num_mask, dtype=torch.long, device=self.device)
        for b in range(B):
            day_mag = mag[b].mean(dim=0)
            day_masked_idx = day_mag.topk(num_mask, dim=0, sorted=False)[1]
            masked_indices[b] = day_masked_idx
        
        # 应用掩码
        masked_fft = cx.clone()
        for b in range(B):
            masked_fft[b, :, masked_indices[b]] = self.frequency_mask_token.repeat(1, 1, 1)
        
        # 逆FFT变换
        masked_x = torch.fft.irfft(masked_fft).transpose(1, 2)  # [B, T, D]
        
        # 将频率域掩码转换为时域掩码
        time_domain_mask = torch.zeros(B, T, N*f, dtype=torch.bool, device=self.device)
        for b in range(B):
            for n in range(N*f):
                freq_mask = torch.zeros(T//2 + 1, dtype=torch.bool, device=self.device)
                freq_mask[masked_indices[b]] = True
                time_domain_mask[b, :, n] = torch.fft.irfft(freq_mask.float()).bool()
        
        # 将掩码后的嵌入转换回原始特征空间
        masked_x = self.emb.value_embedding.weight.T @ masked_x.transpose(1, 2)
        masked_x = masked_x.transpose(1, 2)
        masked_x = masked_x.reshape(B, T, N, f)
        
        # 对未掩码位置进行投影
        unmasked_values = masked_x[~time_domain_mask.reshape(B, T, N, f)]
        projected_values = self.frequency_projection(unmasked_values.unsqueeze(-1))
        masked_x[~time_domain_mask.reshape(B, T, N, f)] = projected_values.squeeze(-1)
        
        return masked_x, masked_indices
    
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
        
        # 保存掩码后的数据为.npy格式
        np.save('new_version/temporal_masked_data.npy', temporal_masked_x.detach().cpu().numpy())
        np.save('new_version/frequency_masked_data.npy', frequency_masked_x.detach().cpu().numpy())
        np.save('new_version/temporal_mask_indices.npy', temporal_mask_indices.detach().cpu().numpy())
        np.save('new_version/frequency_mask_indices.npy', frequency_mask_indices.detach().cpu().numpy())
        
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

def main():
    # 加载原始数据
    train_data = np.load('data/train.npy')
    test_data = np.load('data/test.npy')
    
    # 应用掩码
    temporal_masked_train, temporal_mask_indices_train, freq_masked_train, freq_mask_indices_train = temporal_frequency_masking(train_data)
    temporal_masked_test, temporal_mask_indices_test, freq_masked_test, freq_mask_indices_test = temporal_frequency_masking(test_data)
    
    # 保存掩码后的数据
    np.save('data/temporal_masked_train.npy', temporal_masked_train)
    np.save('data/temporal_masked_test.npy', temporal_masked_test)
    np.save('data/frequency_masked_train.npy', freq_masked_train)
    np.save('data/frequency_masked_test.npy', freq_masked_test)
    
    # 保存掩码索引
    np.save('data/temporal_mask_indices_train.npy', temporal_mask_indices_train)
    np.save('data/temporal_mask_indices_test.npy', temporal_mask_indices_test)
    np.save('data/frequency_mask_indices_train.npy', freq_mask_indices_train)
    np.save('data/frequency_mask_indices_test.npy', freq_mask_indices_test)
    
    print("掩码后的数据集和掩码索引已保存到data/目录下")

if __name__ == '__main__':
    main() 