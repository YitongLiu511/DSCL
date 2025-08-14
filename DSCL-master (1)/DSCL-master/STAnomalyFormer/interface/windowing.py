import torch

class SlidingWindowDataset(torch.utils.data.Dataset):
    """动态滑动窗口数据集，避免一次性加载所有窗口到内存"""
    
    def __init__(self, data, seq_len, target_len=1):
        """
        Args:
            data: (N, T, D) 原始数据
            seq_len: 滑动窗口长度
            target_len: 目标长度（通常为1）
        """
        self.data = data
        self.seq_len = seq_len
        self.target_len = target_len
        self.N, self.T, self.D = data.shape
        self.num_windows = self.T - self.seq_len
        
    def __len__(self):
        # 确保有足够的数据来预测下一个时间戳
        return max(0, self.num_windows)
    
    def __getitem__(self, idx):
        """动态生成滑动窗口样本"""
        # 计算时间戳范围
        start_idx = idx
        end_idx = start_idx + self.seq_len
        target_start = end_idx  # 预测下一个时间戳
        target_end = target_start + self.target_len
        
        # 动态切片，不预先加载
        window = self.data[:, start_idx:end_idx, :]  # (N, seq_len, D)
        target = self.data[:, target_start:target_end, :]  # (N, target_len, D)
        
        return window, target


class AnomalyRecoveryDataset(torch.utils.data.Dataset):
    """异常恢复预测数据集：从异常数据恢复到正常数据"""
    
    def __init__(self, anomaly_data, clean_data, seq_len, target_len=1):
        """
        Args:
            anomaly_data: (N, T, D) 异常数据
            clean_data: (N, T, D) 正常数据
            seq_len: 滑动窗口长度
            target_len: 目标长度（通常为1）
        """
        self.anomaly_data = anomaly_data
        self.clean_data = clean_data
        self.seq_len = seq_len
        self.target_len = target_len
        self.N, self.T, self.D = anomaly_data.shape
        self.num_windows = self.T - self.seq_len
        
    def __len__(self):
        # 确保有足够的数据来预测下一个时间戳
        return max(0, self.num_windows)
    
    def __getitem__(self, idx):
        """动态生成异常恢复样本"""
        # 计算时间戳范围
        start_idx = idx
        end_idx = start_idx + self.seq_len
        target_start = end_idx  # 预测下一个时间戳
        target_end = target_start + self.target_len
        
        # 输入：注入异常数据的滑动窗口 (N, seq_len, D)
        window = self.anomaly_data[:, start_idx:end_idx, :]  # (N, seq_len, D)
        
        # 目标：未注入异常数据中对应时间点的值 (N, 1, D)
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