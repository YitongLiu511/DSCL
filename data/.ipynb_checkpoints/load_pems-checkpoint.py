import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import os

def load_pems_dataset(args, dataset_name):
    """
    加载PEMS数据集
    Args:
        args: argparse.Namespace对象，包含所有参数
        dataset_name: 'PEMS03' 或 'PEMS08'
    Returns:
        X: 训练数据
        val_X: 验证数据
        test_X: 测试数据
        (adj, dist): 邻接矩阵和距离矩阵
        y: 标签
    """
    # 加载数据
    data_path = f"data1/{dataset_name}/{dataset_name}.npz"
    data = np.load(data_path)
    
    # 加载距离和连通性信息
    dist_path = f"data1/{dataset_name}/{dataset_name}_distance.npz"
    dist_data = np.load(dist_path)
    dist = dist_data['distance']
    
    adj_path = f"data1/{dataset_name}/{dataset_name}_connectivity.npz"
    adj_data = np.load(adj_path)
    adj = adj_data['connectivity']
    
    # 获取流量数据
    flow_data = data['data']  # 假设数据存储在'data'键中
    print(f"原始数据形状: {flow_data.shape}")
    
    # 使用min-max归一化
    if args.normalize:
        scaler = MinMaxScaler()
        flow_data = scaler.fit_transform(flow_data.reshape(-1, flow_data.shape[-1])).reshape(flow_data.shape)
    
    # 确保数据点数量能被每天的时间槽数量整除
    time_slots_per_day = 24 * 12  # 5分钟一个时间槽
    num_days = len(flow_data) // time_slots_per_day
    if len(flow_data) % time_slots_per_day != 0:
        print(f"警告：数据点数量({len(flow_data)})不能被每天的时间槽数量({time_slots_per_day})整除")
        print("将截断到最接近的整数天数")
        flow_data = flow_data[:(num_days * time_slots_per_day)]
    
    flow_data = flow_data.reshape(num_days, time_slots_per_day, -1)
    print("重塑后数据形状:", flow_data.shape)
    
    # 按时间顺序划分数据集
    train_size = 14  # 训练集大小
    val_size = 14    # 验证集大小
    test_size = 14   # 测试集大小
    
    # 确保有足够的数据
    if num_days < (train_size + val_size + test_size):
        raise ValueError(f"数据天数不足，需要至少{train_size + val_size + test_size}天的数据，但只有{num_days}天")
    
    # 按时间顺序划分
    X = flow_data[:train_size]
    val_X = flow_data[train_size:train_size + val_size]
    test_X = flow_data[train_size + val_size:train_size + val_size + test_size]
    
    # 生成标签（这里需要根据实际需求修改）
    y = np.zeros(len(test_X))
    
    return X, val_X, test_X, (adj, dist), y

class PemsDataset(torch.utils.data.Dataset):
    def __init__(self, x):
        super(PemsDataset, self).__init__()
        self.x = x

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.x.shape[0]

if __name__ == "__main__":
    # 测试数据加载
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--dataset', type=str, default='PEMS03', choices=['PEMS03', 'PEMS08'])
    args = parser.parse_args()
    
    X, val_X, test_X, (adj, dist), y = load_pems_dataset(args, args.dataset)
    print(f"\n最终数据形状:")
    print(f"X shape: {X.shape}")
    print(f"val_X shape: {val_X.shape}")
    print(f"test_X shape: {test_X.shape}")
    print(f"adj shape: {adj.shape}")
    print(f"dist shape: {dist.shape}")
    print(f"y shape: {y.shape}") 