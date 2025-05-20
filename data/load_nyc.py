import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import os

def inject_temporal_anomaly(X, mu=0.5, anomaly_ratio=0.1):
    """注入时间异常
    Args:
        X: 输入数据，形状为 (N, T, D)
        mu: 流量限制阈值
        anomaly_ratio: 异常区域比例
    """
    N, T, D = X.shape
    n_anomaly = int(N * anomaly_ratio)
    
    # 随机选择区域
    anomaly_regions = np.random.choice(N, n_anomaly, replace=False)
    
    # 对选中的区域注入异常
    for n in anomaly_regions:
        max_flow = np.max(X[n])
        X[n] = np.minimum(X[n], mu * max_flow)
    
    return X, anomaly_regions

def inject_spatial_anomaly(X, k=0.1, anomaly_ratio=0.1):
    """注入空间异常
    Args:
        X: 输入数据，形状为 (N, T, D)
        k: 采样节点比例
        anomaly_ratio: 异常区域比例
    """
    N, T, D = X.shape
    n_anomaly = int(N * anomaly_ratio)
    k_nodes = int(N * k)
    
    # 随机选择区域
    anomaly_regions = np.random.choice(N, n_anomaly, replace=False)
    
    # 对选中的区域注入异常
    for n in anomaly_regions:
        # 随机采样k个节点
        sampled_nodes = np.random.choice(N, k_nodes, replace=False)
        # 计算与当前节点的最大差异
        max_diff = 0
        max_diff_node = None
        for i in sampled_nodes:
            diff = np.linalg.norm(X[n] - X[i])
            if diff > max_diff:
                max_diff = diff
                max_diff_node = i
        # 替换流量
        X[n] = X[max_diff_node]
    
    return X, anomaly_regions

def load_dataset(args):
    """
    加载纽约出租车数据集
    Args:
        args: 参数字典或argparse.Namespace对象，包含所有参数
            normalize: 是否归一化数据
            n_day: 训练集天数
    Returns:
        X: 训练数据
        val_X: 验证数据
        test_X: 测试数据
        (adj, dist, poi_sim): 邻接矩阵、距离矩阵和POI相似度矩阵
        y: 标签
    """
    # 首先加载邻接矩阵以获取基准区域集合
    adj_data = np.load("data/static_adjacency.npz")
    adj = adj_data['connectivity']  # 连通性矩阵
    dist = adj_data['distance']     # 距离矩阵
    poi_sim = adj_data['poi_similarity']  # POI相似度矩阵
    n_zones = adj.shape[0]  # 应该是263
    print(f"邻接矩阵中的区域数量: {n_zones}")
    
    # 读取出租车数据（1月和2月）
    df_jan = pd.read_parquet("data/yellow_tripdata_2023-01.parquet")
    df_feb = pd.read_parquet("data/yellow_tripdata_2023-02.parquet")
    df = pd.concat([df_jan, df_feb], ignore_index=True)
    print("原始数据形状:", df.shape)
    
    # 读取taxi zones数据
    zone_lookup = pd.read_csv("data/taxi _zone_lookup.csv")
    print("区域查找表中的区域数量:", len(zone_lookup))
    
    # 获取邻接矩阵中使用的区域ID
    valid_zones = np.arange(1, n_zones + 1)  # 假设区域ID是从1开始连续的
    print("有效区域ID范围:", valid_zones[0], "到", valid_zones[-1])
    
    # 转换时间戳
    df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    
    # 按10分钟聚合数据，并确保时间槽对齐
    df['time_slot'] = df['pickup_datetime'].dt.floor('10min')
    df['day'] = df['pickup_datetime'].dt.date
    
    # 确保LocationID在有效范围内
    df = df[df['PULocationID'].isin(valid_zones) & df['DOLocationID'].isin(valid_zones)]
    print("过滤后数据形状:", df.shape)
    
    # 创建完整的时间槽索引
    time_slots_per_day = 24 * 6  # 10分钟一个时间槽，每天144个时间槽
    all_days = sorted(df['day'].unique())
    
    # 计算每个区域的流量
    flow_data = df.groupby(['day', 'time_slot', 'PULocationID']).size().unstack(fill_value=0)
    print("初始聚合后数据形状:", flow_data.shape)
    
    # 确保所有区域都存在，并且只包含有效区域
    missing_zones = set(valid_zones) - set(flow_data.columns)
    extra_zones = set(flow_data.columns) - set(valid_zones)
    
    print(f"缺失的区域: {len(missing_zones)}")
    print(f"多余的区域: {len(extra_zones)}")
    
    # 添加缺失的区域
    for zone in missing_zones:
        flow_data[zone] = 0
        
    # 移除多余的区域
    flow_data = flow_data[sorted(valid_zones)]
    
    # 重置索引，确保数据按天和时间槽排序
    flow_data = flow_data.reset_index()
    flow_data['day'] = pd.to_datetime(flow_data['day']).dt.date
    flow_data = flow_data.sort_values(['day', 'time_slot'])
    
    # 确保每天都有完整的时间槽
    complete_data = []
    for day in all_days:
        day_data = flow_data[flow_data['day'] == day].copy()
        if len(day_data) < time_slots_per_day:
            # 创建完整的时间槽
            time_slots = pd.date_range(
                start=pd.Timestamp(day),
                end=pd.Timestamp(day) + pd.Timedelta(days=1) - pd.Timedelta(minutes=10),
                freq='10min'
            ).time
            # 创建完整的数据框
            complete_day = pd.DataFrame({
                'day': [day] * time_slots_per_day,
                'time_slot': time_slots
            })
            # 修正：类型统一
            day_data['time_slot'] = pd.to_datetime(day_data['time_slot']).dt.time
            # 合并现有数据
            complete_day = complete_day.merge(
                day_data,
                on=['day', 'time_slot'],
                how='left'
            ).fillna(0)
            complete_data.append(complete_day)
        else:
            complete_data.append(day_data)
    
    # 合并所有天的数据
    flow_data = pd.concat(complete_data, ignore_index=True)
    print("处理后数据形状:", flow_data.shape)
    
    # 使用min-max归一化
    if isinstance(args, dict):
        normalize = args.get('normalize', False)
    else:
        normalize = args.normalize
        
    if normalize:
        scaler = MinMaxScaler()
        flow_data.iloc[:, 2:] = scaler.fit_transform(flow_data.iloc[:, 2:])
    
    # 转换为numpy数组并重塑
    flow_values = flow_data.iloc[:, 2:].values  # 只取数值列
    print("转换为numpy数组后形状:", flow_values.shape)
    
    # 重塑数据为 (days, time_slots_per_day, n_zones)
    n_days = len(all_days)
    flow_values = flow_values.reshape(n_days, time_slots_per_day, n_zones)
    print("重塑后数据形状:", flow_values.shape)
    
    # 按时间顺序划分数据集
    if isinstance(args, dict):
        train_size = args.get('n_day', 14)
    else:
        train_size = args.n_day
        
    val_size = train_size    # 验证集大小与训练集相同
    test_size = train_size   # 测试集大小与训练集相同
    
    # 确保有足够的数据
    required_days = train_size + val_size + test_size
    if n_days < required_days:
        raise ValueError(f"数据天数不足，需要至少{required_days}天的数据，但只有{n_days}天")
    
    # 只使用前required_days天的数据
    flow_values = flow_values[:required_days]
    
    # 注入时间异常
    flow_values, temporal_anomalies = inject_temporal_anomaly(
        flow_values.copy(), 
        mu=0.5, 
        anomaly_ratio=args.anormly_ratio
    )
    
    # 注入空间异常
    flow_values, spatial_anomalies = inject_spatial_anomaly(
        flow_values.copy(),
        k=0.1,
        anomaly_ratio=args.anormly_ratio
    )
    
    # 按时间顺序划分
    X = flow_values[:train_size]
    val_X = flow_values[train_size:train_size + val_size]
    test_X = flow_values[train_size + val_size:train_size + val_size + test_size]
    
    # 生成标签（这里需要根据实际需求修改）
    n_zones = adj.shape[0]  # 区域数量
    n_time_slots = test_X.shape[1]  # 时间槽数量
    y = np.zeros((n_zones, n_time_slots, 2))  # 初始化所有区域和时间槽为正常
    
    # 设置时间异常标签
    for n in temporal_anomalies:
        y[n, :, 0] = 1
    
    # 设置空间异常标签
    for n in spatial_anomalies:
        y[n, :, 1] = 1
    
    print(f"\n标签统计:")
    print(f"总单元格数: {n_zones * n_time_slots}")
    print(f"正常单元格数: {np.sum(y[:, :, 0] == 0)}")
    print(f"异常单元格数: {np.sum(y[:, :, 1] == 1)}")
    print(f"异常比例: {np.mean(y[:, :, 1] == 1) * 100:.2f}%")
    print(f"每个区域的异常时间槽数量: {np.sum(y[:, :, 0], axis=1)}")
    print(f"每个时间槽的异常区域数量: {np.sum(y[:, :, 0], axis=0)}")
    
    # 将标签重塑为与输入数据相同的形状
    y = y.reshape(n_zones, n_time_slots, 2)  # 确保标签形状正确
    y = y.transpose(1, 0, 2)  # 转置为 (时间槽, 区域, 异常类型)
    y = y.reshape(1, n_time_slots, n_zones, 2)  # 添加天数维度
    y = np.repeat(y, test_X.shape[0], axis=0)  # 复制到与test_X相同的天数
    
    return X, val_X, test_X, (adj, dist, poi_sim), y

def get_loader_segment(data, patch_len, stride, batch_size, shuffle=True):
    """
    创建数据加载器，使用与STPatchFormer相同的patch切分方式
    Args:
        data: 输入数据，形状为 [days, time_slots, n_zones]
        patch_len: patch长度
        stride: patch步长
        batch_size: 批次大小
        shuffle: 是否打乱数据
    Returns:
        DataLoader对象
    """
    class TimeSeriesDataset(Dataset):
        def __init__(self, data, patch_len, stride):
            self.data = data
            self.patch_len = patch_len
            self.stride = stride
            # 计算每个时间槽可以生成的patch数量
            self.num_patches = (data.shape[1] - patch_len) // stride + 1
            # 计算总patch数量
            self.total_patches = self.num_patches * data.shape[0]
            
        def __len__(self):
            return self.total_patches
            
        def __getitem__(self, idx):
            # 计算当前patch属于哪一天
            day_idx = idx // self.num_patches
            # 计算当前patch在该天中的位置
            patch_idx = idx % self.num_patches
            # 计算起始时间槽
            start_slot = patch_idx * self.stride
            end_slot = start_slot + self.patch_len
            # 获取数据并调整维度顺序为 [1, patch_len, n_zones]
            x = self.data[day_idx, start_slot:end_slot]  # [patch_len, n_zones]
            # 将数据转换为[B, T, C]格式，其中B=1, T=patch_len, C=n_zones
            x = x.reshape(1, x.shape[0], x.shape[1])  # [1, patch_len, n_zones]
            # 确保数据类型正确
            x = x.astype(np.float32)
            # 转换为PyTorch张量
            x = torch.FloatTensor(x)
            # 确保维度顺序正确
            x = x.permute(0, 1, 2)  # [B, T, C]
            return x, x  # 返回相同的x作为输入和目标
    
    dataset = TimeSeriesDataset(data, patch_len, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    # 测试数据加载
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--n_day', type=int, default=14)  # 默认值改为14
    args = parser.parse_args()
    
    X, val_X, test_X, (adj, dist, poi_sim), y = load_dataset(args)
    print(f"\n最终数据形状:")
    print(f"X shape: {X.shape}")
    print(f"val_X shape: {val_X.shape}")
    print(f"test_X shape: {test_X.shape}")
    print(f"adj shape: {adj.shape}")
    print(f"dist shape: {dist.shape}")
    print(f"poi_sim shape: {poi_sim.shape}")
    print(f"y shape: {y.shape}") 