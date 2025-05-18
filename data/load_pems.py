import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import os
from scipy.sparse import csr_matrix

def load_dataset(args):
    """
    加载PEMS数据集
    Args:
        args: argparse.Namespace对象，包含所有参数
    Returns:
        X: 训练数据，形状为 (n_nodes, time_slots_per_day, n_days)
        val_X: 验证数据，形状为 (n_nodes, time_slots_per_day, n_days)
        test_X: 测试数据，形状为 (n_nodes, time_slots_per_day, n_days)
        adj: 邻接矩阵，形状为 (n_nodes, n_nodes)
        y: 标签，形状为 (n_nodes,)
    """
    # 根据数据集编号选择对应的文件夹
    dataset_name = f"PEMS0{args.dataset}"
    data_dir = os.path.join("data1", f"pems0{args.dataset}")
    
    # 加载交通流量数据
    data_path = os.path.join(data_dir, f"{dataset_name}.npz")
    data = np.load(data_path)
    flow_data = data['data']  # 数据存储在'data'键中
    
    # 加载邻接矩阵
    adj_path = os.path.join(data_dir, "adj.npz")
    adj_data = np.load(adj_path)
    # 从稀疏矩阵格式构建邻接矩阵
    adj = csr_matrix((adj_data['data'], (adj_data['row'], adj_data['col'])), 
                    shape=adj_data['shape']).toarray()
    
    # 打印原始数据维度
    print(f"原始数据维度: {flow_data.shape}")
    print(f"邻接矩阵维度: {adj.shape}")
    
    # 确保数据维度正确
    n_nodes = adj.shape[0]  # 节点数
    if flow_data.shape[1] != n_nodes:
        raise ValueError(f"数据维度不匹配：流量数据节点数({flow_data.shape[1]})与邻接矩阵节点数({n_nodes})不一致")
    
    # 如果数据是三维的，需要调整维度顺序
    if len(flow_data.shape) == 3:
        # 将数据从 (time_steps, n_nodes, features) 转换为 (time_steps, n_nodes)
        flow_data = flow_data.squeeze(-1)
    
    # 计算时间步数
    time_slots_per_day = 24 * 12  # 5分钟一个时间步，每天288个时间步
    num_days = len(flow_data) // time_slots_per_day
    
    # 按时间顺序划分数据集
    train_size = args.n_day  # 训练集大小
    val_size = args.n_day    # 验证集大小
    test_size = args.n_day   # 测试集大小
    
    # 确保有足够的数据
    if num_days < (train_size + val_size + test_size):
        raise ValueError(f"数据天数不足，需要至少{train_size + val_size + test_size}天的数据，但只有{num_days}天")
    
    # 计算每个集合的起始和结束索引
    train_end = train_size * time_slots_per_day
    val_end = (train_size + val_size) * time_slots_per_day
    test_end = (train_size + val_size + test_size) * time_slots_per_day
    
    # 分别处理每个数据集，避免一次性加载所有数据
    if args.normalize:
        scaler = MinMaxScaler()
        # 只使用训练集数据来拟合scaler
        train_data = flow_data[:train_end].reshape(-1, n_nodes)
        scaler.fit(train_data)
        
        # 分别转换每个数据集，保持正确的维度
        X = scaler.transform(flow_data[:train_end].reshape(-1, n_nodes)).reshape(train_size, time_slots_per_day, n_nodes)
        val_X = scaler.transform(flow_data[train_end:val_end].reshape(-1, n_nodes)).reshape(val_size, time_slots_per_day, n_nodes)
        test_X = scaler.transform(flow_data[val_end:test_end].reshape(-1, n_nodes)).reshape(test_size, time_slots_per_day, n_nodes)
    else:
        X = flow_data[:train_end].reshape(train_size, time_slots_per_day, n_nodes)
        val_X = flow_data[train_end:val_end].reshape(val_size, time_slots_per_day, n_nodes)
        test_X = flow_data[val_end:test_end].reshape(test_size, time_slots_per_day, n_nodes)
    
    # 转置数据维度，使其与NYC数据集格式一致
    # 从 (n_days, time_slots_per_day, n_nodes) 转换为 (n_nodes, time_slots_per_day, n_days)
    X = X.transpose(2, 1, 0)
    val_X = val_X.transpose(2, 1, 0)
    test_X = test_X.transpose(2, 1, 0)
    
    # 打印处理后的数据维度
    print(f"\n处理后的数据维度:")
    print(f"X shape: {X.shape}")
    print(f"val_X shape: {val_X.shape}")
    print(f"test_X shape: {test_X.shape}")
    print(f"adj shape: {adj.shape}")
    
    # 释放不需要的数据
    del flow_data
    del data
    
    # 生成标签（使用随机生成的异常标签）
    np.random.seed(42)  # 设置随机种子以确保可重复性
    y = np.zeros(n_nodes)  # 初始化所有节点为正常
    anomaly_ratio = 0.1  # 设置异常节点比例为10%
    n_anomalies = int(n_nodes * anomaly_ratio)
    anomaly_indices = np.random.choice(n_nodes, n_anomalies, replace=False)
    y[anomaly_indices] = 1  # 将选中的节点标记为异常
    
    print(f"\n标签统计:")
    print(f"正常节点数量: {np.sum(y == 0)}")
    print(f"异常节点数量: {np.sum(y == 1)}")
    
    # 将数据转换为PyTorch张量
    X = torch.from_numpy(X).float()
    val_X = torch.from_numpy(val_X).float()
    test_X = torch.from_numpy(test_X).float()
    adj = torch.from_numpy(adj).float()
    y = torch.from_numpy(y).float()
    
    return X, val_X, test_X, adj, y

if __name__ == "__main__":
    # 测试数据加载
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=int, choices=[3, 8])
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--n_day', type=int, default=14)
    args = parser.parse_args()
    
    X, val_X, test_X, adj, y = load_dataset(args)
    print(f"\n最终数据形状:")
    print(f"X shape: {X.shape}")
    print(f"val_X shape: {val_X.shape}")
    print(f"test_X shape: {test_X.shape}")
    print(f"adj shape: {adj.shape}")
    print(f"y shape: {y.shape}") 