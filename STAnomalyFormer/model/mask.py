import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalMask(nn.Module):
    def __init__(self, window_size=10):
        super().__init__()
        self.window_size = window_size
        self.mask_param = nn.Parameter(torch.randn(1))
        
    def forward(self, x):
        # 计算局部变异系数
        mean = F.avg_pool1d(x, self.window_size, stride=1, padding=self.window_size//2)
        var = F.avg_pool1d(x.pow(2), self.window_size, stride=1, padding=self.window_size//2) - mean.pow(2)
        cv = torch.sqrt(var) / (mean + 1e-6)
        
        # 生成掩码
        mask = (cv > self.mask_param).float()
        return x * (1 - mask) + self.mask_param * mask

class FrequencyMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_param = nn.Parameter(torch.randn(1))
        
    def forward(self, x):
        # FFT
        fft = torch.fft.fft(x, dim=-1)
        magnitude = torch.abs(fft)
        
        # 生成掩码
        mask = (magnitude < self.mask_param).float()
        fft_masked = fft * (1 - mask) + self.mask_param * mask
        
        # 逆FFT
        return torch.fft.ifft(fft_masked, dim=-1).real 