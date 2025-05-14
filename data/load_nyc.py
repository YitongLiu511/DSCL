import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import os

def load_dataset(args):
    """
    加载纽约出租车数据集
    Args:
        args: argparse.Namespace对象，包含所有参数
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
    
    # 按10分钟聚合数据
    df['time_slot'] = df['pickup_datetime'].dt.floor('10min')
    df['day'] = df['pickup_datetime'].dt.date
    
    # 确保LocationID在有效范围内
    df = df[df['PULocationID'].isin(valid_zones) & df['DOLocationID'].isin(valid_zones)]
    print("过滤后数据形状:", df.shape)
    
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
    
    print("处理后数据形状:", flow_data.shape)
    
    # 使用min-max归一化
    if args.normalize:
        scaler = MinMaxScaler()
        flow_data = pd.DataFrame(
            scaler.fit_transform(flow_data),
            index=flow_data.index,
            columns=flow_data.columns
        )
    
    # 转换为numpy数组并重塑
    flow_data = flow_data.values
    print("转换为numpy数组后形状:", flow_data.shape)
    
    # 确保数据点数量能被每天的时间槽数量整除
    time_slots_per_day = 24 * 6  # 10分钟一个时间槽
    num_days = len(flow_data) // time_slots_per_day
    if len(flow_data) % time_slots_per_day != 0:
        print(f"警告：数据点数量({len(flow_data)})不能被每天的时间槽数量({time_slots_per_day})整除")
        print("将截断到最接近的整数天数")
        flow_data = flow_data[:(num_days * time_slots_per_day)]
    
    flow_data = flow_data.reshape(num_days, time_slots_per_day, n_zones)
    print("重塑后数据形状:", flow_data.shape)
    
    # 按时间顺序划分数据集
    train_size = 14  # 训练集大小改为14天
    val_size = 14    # 验证集大小改为14天
    test_size = 14   # 测试集大小改为14天
    
    # 确保有足够的数据
    if num_days < (train_size + val_size + test_size):
        raise ValueError(f"数据天数不足，需要至少{train_size + val_size + test_size}天的数据，但只有{num_days}天")
    
    # 按时间顺序划分
    X = flow_data[:train_size]
    val_X = flow_data[train_size:train_size + val_size]
    test_X = flow_data[train_size + val_size:train_size + val_size + test_size]
    
    # 生成标签（这里需要根据实际需求修改）
    y = np.zeros(len(test_X))
    
    return X, val_X, test_X, (adj, dist, poi_sim), y

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