import torch

class SlidingWindowDataset(torch.utils.data.Dataset):
    """动态滑动窗口数据集，避免一次性加载所有窗口到内存"""
    
    def __init__(self, data, seq_len, target_len=1, sample_indices=None):
        """
        Args:
            data: (N, T, D) 原始数据
            seq_len: 滑动窗口长度
            target_len: 目标长度（通常为1）
            sample_indices: 可选的采样索引，如果提供则只使用指定的窗口
        """
        self.data = data
        self.seq_len = seq_len
        self.target_len = target_len
        self.N, self.T, self.D = data.shape
        # 窗口数与训练循环保持一致：允许最后一个窗口的目标为窗口内最后一步
        self.num_windows = self.T - self.seq_len + 1
        
        # 🆕 支持采样索引
        if sample_indices is not None:
            self.sample_indices = sample_indices
            self.use_sampling = True
            print(f"   📊 启用窗口采样: 从{self.num_windows}个窗口中采样{len(sample_indices)}个")
        else:
            self.sample_indices = None
            self.use_sampling = False
        
    def __len__(self):
        if self.use_sampling:
            return len(self.sample_indices)
        else:
            return max(0, self.num_windows)
    
    def __getitem__(self, idx):
        """动态生成滑动窗口样本"""
        if self.use_sampling:
            actual_idx = int(self.sample_indices[idx].item())
        else:
            actual_idx = int(idx)
            
        # 计算时间戳范围
        start_idx = actual_idx
        end_idx = start_idx + self.seq_len
        # 目标为窗口内最后一个时间步
        target_start = end_idx - 1
        target_end = target_start + self.target_len
        
        # 切片
        window = self.data[:, start_idx:end_idx, :]  # (N, seq_len, D)
        target = self.data[:, target_start:target_end, :]  # (N, target_len, D)
        
        return window, target


class AnomalyRecoveryDataset(torch.utils.data.Dataset):
    """异常恢复预测数据集：从异常数据恢复到正常数据"""
    
    def __init__(self, anomaly_data, clean_data, seq_len, target_len=1, sample_indices=None):
        """
        Args:
            anomaly_data: (N, T, D) 异常数据
            clean_data: (N, T, D) 正常数据
            seq_len: 滑动窗口长度
            target_len: 目标长度（通常为1）
            sample_indices: 可选的采样索引，如果提供则只使用指定的窗口
        """
        self.anomaly_data = anomaly_data
        self.clean_data = clean_data
        self.seq_len = seq_len
        self.target_len = target_len
        self.N, self.T, self.D = anomaly_data.shape
        # 与训练循环保持一致
        self.num_windows = self.T - self.seq_len + 1
        
        # 🆕 支持采样索引
        if sample_indices is not None:
            self.sample_indices = sample_indices
            self.use_sampling = True
            print(f"   📊 启用异常恢复窗口采样: 从{self.num_windows}个窗口中采样{len(sample_indices)}个")
        else:
            self.sample_indices = None
            self.use_sampling = False
        
    def __len__(self):
        if self.use_sampling:
            return len(self.sample_indices)
        else:
            return max(0, self.num_windows)
    
    def __getitem__(self, idx):
        """动态生成异常恢复样本"""
        if self.use_sampling:
            actual_idx = int(self.sample_indices[idx].item())
        else:
            actual_idx = int(idx)
            
        # 计算时间戳范围
        start_idx = actual_idx
        end_idx = start_idx + self.seq_len
        # 目标为窗口内最后一个时间步
        target_start = end_idx - 1
        target_end = target_start + self.target_len
        
        # 切片
        window = self.anomaly_data[:, start_idx:end_idx, :]  # (N, seq_len, D)
        target = self.clean_data[:, target_start:target_end, :]  # (N, target_len, D)
        
        return window, target


class SequenceRecoveryDataset(torch.utils.data.Dataset):
    """整段序列数据集：不进行时间滑窗，单样本整段训练/推理"""

    def __init__(self, anomaly_data, clean_data=None, target_len=1):
        """
        Args:
            anomaly_data: (N, T, D) 异常数据（或原始数据）
            clean_data: (N, T, D) 正常数据（可选，用于恢复范式）
            target_len: 目标长度（通常为1，取最后一个时间步）
        """
        self.anomaly_data = anomaly_data
        self.clean_data = clean_data
        self.target_len = target_len
        self.N, self.T, self.D = anomaly_data.shape

    def __len__(self):
        # 整段即单样本
        return 1

    def __getitem__(self, idx):
        # 整段窗口 (N, T, D)
        window = self.anomaly_data  # (N, T, D)
        # 目标：最后一个时间步 (N, 1, D)
        if self.clean_data is not None:
            target = self.clean_data[:, -self.target_len:, :]
        else:
            target = self.anomaly_data[:, -self.target_len:, :]
        return window, target 