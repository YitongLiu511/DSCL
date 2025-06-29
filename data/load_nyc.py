import numpy as np
import pandas as pd
import torch
import random
import geopandas as gpd
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from new_version.temporal_frequency_masking import TemporalFrequencyMasking
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from new_version.temporal_attention import process_temporal_masked_data

def inject_anomalies(data, anomaly_ratio=0.1, random_seed=42):
    """
    注入异常：通过替换连续3个时间戳的片段
    Args:
        data: 形状为 (n_slots, n_zones, n_features) 的三维数据
        anomaly_ratio: 异常比例，默认0.1
        random_seed: 随机种子
    Returns:
        data_with_anomalies: 注入异常后的数据
        anomaly_labels: 异常标签，1表示异常，0表示正常
    """
    np.random.seed(random_seed)
    total_slots, n_zones, n_features = data.shape
    
    data_with_anomalies = data.copy()
    anomaly_labels = np.zeros_like(data)
    
    # 计算总数据点数
    total_points = total_slots * n_zones
    # 计算目标异常数据点数
    target_anomaly_points = int(total_points * anomaly_ratio)
    # 计算异常片段数（每个片段3个点）
    n_anomaly_fragments = target_anomaly_points // 3
    
    print(f"\n=== 异常注入统计 ===")
    print(f"区域数量: {n_zones}")
    print(f"总时间槽数: {total_slots}")
    print(f"特征数量: {n_features}")
    print(f"总数据点数: {total_points}")
    print(f"目标异常数据点数: {target_anomaly_points}")
    print(f"目标异常片段数: {n_anomaly_fragments}")
    print(f"预期异常比例: {anomaly_ratio}")
    
    def random_sample_fragments(k, exclude_zone):
        """随机采样k个候选片段，并确保不在指定区域
        Args:
            k: 采样数量
            exclude_zone: 需要排除的区域
        Returns:
            fragments: 候选片段列表，每个元素为(slot, zone)
        """
        fragments = []
        for _ in range(k):
            zone = np.random.randint(0, n_zones)
            # 如果随机选到了要排除的区域，就重新选
            while zone == exclude_zone:
                zone = np.random.randint(0, n_zones)
            slot = np.random.randint(0, total_slots - 2)  # 确保有3个连续时间戳
            fragments.append((slot, zone))
        return fragments
    
    # 计算k值
    k = 200  # 使用一个固定的、合理的k值
    print(f"每个目标片段的候选片段数 (固定值): {k}")
    
    injected_fragments = 0
    while injected_fragments < n_anomaly_fragments:
        target_zone = np.random.randint(0, n_zones)
        target_slot = np.random.randint(0, total_slots - 2)

        # 检查该位置是否已经是异常
        if np.any(anomaly_labels[target_slot:target_slot+3, target_zone] == 1):
            continue

        target_fragment = data[target_slot:target_slot+3, target_zone]
        
        # 候选片段不能来自目标区域
        candidate_fragments = random_sample_fragments(k, target_zone)
        
        candidate_distances = []
        for candidate_slot, candidate_zone in candidate_fragments:
            # 避免候选片段与目标片段重叠（虽然跨区域，保险起见）
            if candidate_zone == target_zone and abs(candidate_slot - target_slot) < 3:
                continue
            candidate_fragment = data[candidate_slot:candidate_slot+3, candidate_zone]
            distance = np.linalg.norm(target_fragment - candidate_fragment)
            candidate_distances.append((distance, candidate_slot, candidate_zone))
            
        if not candidate_distances:
            continue
            
        best_candidate = max(candidate_distances, key=lambda x: x[0])
        _, best_slot, best_zone = best_candidate
        
        # 替换数据并打上标签
        data_with_anomalies[target_slot:target_slot+3, target_zone] = \
            data[best_slot:best_slot+3, best_zone]
        anomaly_labels[target_slot:target_slot+3, target_zone] = 1
        
        # 在注入片段的边缘进行平滑处理，以减小突变
        if target_slot > 0:
            # 平滑开始点 (80%原数据 + 20%异常数据)
            data_with_anomalies[target_slot, target_zone] = \
                0.8 * data[target_slot, target_zone] + \
                0.2 * data_with_anomalies[target_slot, target_zone]
        
        if target_slot < total_slots - 3:
            # 平滑结束点 (80%原数据 + 20%异常数据)
            data_with_anomalies[target_slot + 2, target_zone] = \
                0.8 * data[target_slot + 2, target_zone] + \
                0.2 * data_with_anomalies[target_slot + 2, target_zone]

        injected_fragments += 1
        if injected_fragments % 100 == 0:
            print(f"已注入异常片段数: {injected_fragments}/{n_anomaly_fragments}")
    
    # 统计异常点数
    actual_anomaly_points = np.sum(np.any(anomaly_labels == 1, axis=-1))
    actual_ratio = actual_anomaly_points / total_points
    print(f"\n=== 异常注入完成 ===")
    print(f"实际注入的异常片段数: {injected_fragments}")
    print(f"实际异常数据点数: {actual_anomaly_points}")
    print(f"实际异常比例: {actual_ratio:.4f}")

    return data_with_anomalies, anomaly_labels

def load_dataset(args, run_post_processing=True):
    """
    加载纽约出租车数据集
    Args:
        args: 参数字典或argparse.Namespace对象，包含所有参数
            normalize: 是否归一化数据
            n_day: 训练集天数
            inject_anomaly: 是否注入异常
            anomaly_ratio: 异常比例
        run_post_processing: 是否运行掩码和注意力等后处理步骤
    Returns:
        根据 run_post_processing 的值，返回不同数量的结果
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
    
    # 只保留2023年1月和2月的数据
    df = df[(df['pickup_datetime'].dt.year == 2023) & (df['pickup_datetime'].dt.month.isin([1, 2]))]
    
    # 按10分钟聚合数据，并确保时间槽对齐
    df['time_slot'] = df['pickup_datetime'].dt.floor('10min')
    df['day'] = df['pickup_datetime'].dt.date
    
    # 确保LocationID在有效范围内
    df = df[df['PULocationID'].isin(valid_zones) & df['DOLocationID'].isin(valid_zones)]
    print("过滤后数据形状:", df.shape)
    
    # 创建完整的时间槽索引
    time_slots_per_day = 24 * 6  # 10分钟一个时间槽，每天144个时间槽
    all_days = sorted(df['day'].unique())
    
    # 分别计算流入和流出流量
    inflow_data = df.groupby(['day', 'time_slot', 'DOLocationID']).size().unstack(fill_value=0.0)  # 流入流量
    outflow_data = df.groupby(['day', 'time_slot', 'PULocationID']).size().unstack(fill_value=0.0)  # 流出流量
    print("初始聚合后数据形状 - 流入:", inflow_data.shape, "流出:", outflow_data.shape)
    
    # 打印列名信息
    print("\n流入数据列名:", sorted(inflow_data.columns))
    print("流出数据列名:", sorted(outflow_data.columns))
    
    # 确保所有区域都存在，并且只包含有效区域
    for zone in valid_zones:
        if zone not in inflow_data.columns:
            inflow_data[zone] = 0.0
        if zone not in outflow_data.columns:
            outflow_data[zone] = 0.0
    
    # 只保留有效区域并排序
    inflow_data = inflow_data[sorted(valid_zones)]
    outflow_data = outflow_data[sorted(valid_zones)]
    
    print("\n处理后的列名:")
    print("流入数据列名:", sorted(inflow_data.columns))
    print("流出数据列名:", sorted(outflow_data.columns))
    
    # 重置索引，确保数据按天和时间槽排序
    inflow_data = inflow_data.reset_index()
    outflow_data = outflow_data.reset_index()
    
    inflow_data['day'] = pd.to_datetime(inflow_data['day']).dt.date
    outflow_data['day'] = pd.to_datetime(outflow_data['day']).dt.date
    
    inflow_data = inflow_data.sort_values(['day', 'time_slot'])
    outflow_data = outflow_data.sort_values(['day', 'time_slot'])
    
    # 确保每天都有完整的时间槽
    complete_inflow_data = []
    complete_outflow_data = []
    
    for day in all_days:
        # 处理流入数据
        day_inflow = inflow_data[inflow_data['day'] == day].copy()
        if len(day_inflow) < time_slots_per_day:
            time_slots = pd.date_range(
                start=pd.Timestamp(day),
                end=pd.Timestamp(day) + pd.Timedelta(days=1) - pd.Timedelta(minutes=10),
                freq='10min'
            ).time
            complete_day = pd.DataFrame({
                'day': [day] * time_slots_per_day,
                'time_slot': time_slots
            })
            day_inflow['time_slot'] = pd.to_datetime(day_inflow['time_slot']).dt.time
            complete_day = complete_day.merge(
                day_inflow,
                on=['day', 'time_slot'],
                how='left'
            ).fillna(0.0)
            complete_inflow_data.append(complete_day)
        else:
            complete_inflow_data.append(day_inflow)
            
        # 处理流出数据
        day_outflow = outflow_data[outflow_data['day'] == day].copy()
        if len(day_outflow) < time_slots_per_day:
            time_slots = pd.date_range(
                start=pd.Timestamp(day),
                end=pd.Timestamp(day) + pd.Timedelta(days=1) - pd.Timedelta(minutes=10),
                freq='10min'
            ).time
            complete_day = pd.DataFrame({
                'day': [day] * time_slots_per_day,
                'time_slot': time_slots
            })
            day_outflow['time_slot'] = pd.to_datetime(day_outflow['time_slot']).dt.time
            complete_day = complete_day.merge(
                day_outflow,
                on=['day', 'time_slot'],
                how='left'
            ).fillna(0.0)
            complete_outflow_data.append(complete_day)
        else:
            complete_outflow_data.append(day_outflow)
    
    inflow_data = pd.concat(complete_inflow_data, ignore_index=True)
    outflow_data = pd.concat(complete_outflow_data, ignore_index=True)
    
    print("处理后数据形状 - 流入:", inflow_data.shape, "流出:", outflow_data.shape)
    
    # 转换为numpy数组
    inflow_values = inflow_data.iloc[:, 2:].values  # 只取数值列
    outflow_values = outflow_data.iloc[:, 2:].values  # 只取数值列
    
    print("转换为numpy数组后形状 - 流入:", inflow_values.shape, "流出:", outflow_values.shape)
    
    # -- 注释掉三维变四维的操作 --
    # n_days = len(all_days)
    # inflow_values = inflow_values.reshape(n_days, time_slots_per_day, n_zones)
    # outflow_values = outflow_values.reshape(n_days, time_slots_per_day, n_zones)
    
    # 合并流入和流出数据
    flow_values = np.stack([inflow_values, outflow_values], axis=-1)
    print(f"合并后数据形状 (三维): {flow_values.shape}")
    
    # 1. 划分训练集和测试集
    # 训练集天数 * 每天时间槽数 = 总时间槽数
    n_train_slots = 14 * time_slots_per_day
    n_test_slots = 14 * time_slots_per_day
    
    X_train = flow_values[:n_train_slots]
    X_test = flow_values[n_train_slots : n_train_slots + n_test_slots]
    
    # 提前初始化标签数组，确保它们总是存在
    y_train = np.zeros_like(X_train)
    y_test = np.zeros_like(X_test)
    
    print(f"\n数据集划分:")
    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    
    # 2. 归一化
    if args.normalize:
        print("\n开始归一化...")
        scaler = MinMaxScaler()
        # 使用训练集拟合归一化参数
        X_train_reshaped = X_train.reshape(-1, 2)  # 重塑为2D数组，每行包含流入和流出两个特征
        scaler.fit(X_train_reshaped)
        # 转换训练集和测试集
        X_train = scaler.transform(X_train_reshaped).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, 2)).reshape(X_test.shape)
        print("归一化完成")
        
        # 保存归一化后的数据
        np.save('data/normalized_train.npy', X_train)
        np.save('data/normalized_test.npy', X_test)
        print("已保存归一化后的训练集和测试集")
    
    # 3. 在归一化后的训练集中注入异常
    if args.inject_anomaly:
        print("\n开始注入异常...")
        X_train, y_train = inject_anomalies(X_train, anomaly_ratio=args.anomaly_ratio)
        
    # 如果不需要后处理，则直接返回
    if not run_post_processing:
        return X_train, X_test, (adj, dist, poi_sim), y_train, y_test

    # --- 后续处理（时频掩码、注意力等） ---
    print("\n=== 开始时频掩蔽处理 ===")
    
    # 创建时频掩蔽模块
    masking_module = TemporalFrequencyMasking(
        window_size=10,  # 可以根据需要调整窗口大小
        temporal_mask_ratio=0.1,
        frequency_mask_ratio=0.1,
        d_model=263  # 使用区域数量作为模型维度
    )
    
    # 将数据转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    
    print("处理训练集...")
    temporal_masked_X_train, temporal_mask_indices_train = masking_module.temporal_masking(X_train_tensor)
    frequency_masked_X_train, frequency_mask_indices_train = masking_module.frequency_masking(X_train_tensor)
    print(f"训练集时间掩蔽位置数量: {temporal_mask_indices_train.shape[0] * temporal_mask_indices_train.shape[1]}")
    print(f"训练集频率掩蔽位置数量: {frequency_mask_indices_train.shape[0] * frequency_mask_indices_train.shape[1]}")
    
    print("\n处理测试集...")
    temporal_masked_X_test, temporal_mask_indices_test = masking_module.temporal_masking(X_test_tensor)
    frequency_masked_X_test, frequency_mask_indices_test = masking_module.frequency_masking(X_test_tensor)
    print(f"测试集时间掩蔽位置数量: {temporal_mask_indices_test.shape[0] * temporal_mask_indices_test.shape[1]}")
    print(f"测试集频率掩蔽位置数量: {frequency_mask_indices_test.shape[0] * frequency_mask_indices_test.shape[1]}")
    
    # 将掩蔽后的数据转换为numpy数组
    X_train = temporal_masked_X_train.cpu().detach().numpy()
    X_test = temporal_masked_X_test.cpu().detach().numpy()
    
    # 添加时间注意力处理
    print("\n=== 开始时间注意力处理 ===")
    
    print("处理训练集时间掩码数据...")
    # 将numpy数组转回PyTorch张量
    temporal_masked_X_train_tensor = torch.FloatTensor(X_train)
    # 应用时间注意力处理
    processed_X_train, train_attention_weights = process_temporal_masked_data(
        temporal_masked_X_train_tensor,
        d_model=263,  # 使用区域数量作为模型维度
        n_heads=8,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"训练集时间注意力处理完成，输出形状: {processed_X_train.shape}")
    print(f"注意力权重数量: {len(train_attention_weights)}")
    
    print("\n处理测试集时间掩码数据...")
    temporal_masked_X_test_tensor = torch.FloatTensor(X_test)
    processed_X_test, test_attention_weights = process_temporal_masked_data(
        temporal_masked_X_test_tensor,
        d_model=263,
        n_heads=8,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"测试集时间注意力处理完成，输出形状: {processed_X_test.shape}")
    print(f"注意力权重数量: {len(test_attention_weights)}")
    
    # 将处理后的数据转换回numpy数组
    X_train = processed_X_train.cpu().detach().numpy()
    X_test = processed_X_test.cpu().detach().numpy()
    y_train = None  # Assuming y_train is not provided in the original function
    y_test = None  # Assuming y_test is not provided in the original function

    print("\n=== 时频掩蔽处理完成 ===")
    print(f"处理后的训练集形状: {X_train.shape}")
    print(f"处理后的测试集形状: {X_test.shape}")
    
    # 保存最终处理后的数据
    np.save('data/processed_train.npy', X_train)
    np.save('data/processed_test.npy', X_test)
    print("已保存最终处理后的训练集和测试集")

    return X_train, X_test, (adj, dist, poi_sim), y_train, y_test

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalize', action='store_true', help='是否归一化数据')
    parser.add_argument('--n_day', type=int, default=14, help='训练集天数')
    parser.add_argument('--inject_anomaly', action='store_true', help='是否注入异常')
    parser.add_argument('--anomaly_ratio', type=float, default=0.1, help='异常比例')
    args = parser.parse_args()
    
    # 加载数据集
    X_train, X_test, (adj, dist, poi_sim), y_train, y_test = load_dataset(args)
    
    print("\n最终数据形状:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"adj shape: {adj.shape}")
    print(f"dist shape: {dist.shape}")
    print(f"poi_sim shape: {poi_sim.shape}")
    if y_train is not None:
        print(f"y_train shape: {y_train.shape}")
    if y_test is not None:
        print(f"y_test shape: {y_test.shape}") 