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

def inject_anomalies(data, anomaly_ratio=0.05, random_seed=42):
    """
    优化版：异常片段均匀分配到每个区域，异常标签数量=0.05*区域数*时间槽数，不乘以特征数。
    Args:
        data: 形状为 (n_slots, n_zones, n_features) 的三维数据
        anomaly_ratio: 异常比例，默认0.05
        random_seed: 随机种子
    Returns:
        data_with_anomalies: 注入异常后的数据
        anomaly_labels: 异常标签，形状为 (n_slots, n_zones)，1表示异常，0表示正常
    """
    np.random.seed(random_seed)
    total_slots, n_zones, n_features = data.shape
    data_with_anomalies = data.copy()
    # 修改：标签形状为 (n_slots, n_zones)，不考虑特征维度
    anomaly_labels = np.zeros((total_slots, n_zones), dtype=int)

    # 计算总数据点数（不乘以特征数）
    total_points = total_slots * n_zones
    # 计算目标异常数据点数
    target_anomaly_points = int(total_points * anomaly_ratio)
    # 计算每个区域分配的异常片段数（每片段3个点）
    n_anomaly_fragments = target_anomaly_points // 3
    fragments_per_zone = n_anomaly_fragments // n_zones
    extra_fragments = n_anomaly_fragments % n_zones

    print(f"\n=== 优化异常注入统计 ===")
    print(f"区域数量: {n_zones}")
    print(f"总时间槽数: {total_slots}")
    print(f"特征数量: {n_features}")
    print(f"总数据点数(不含特征): {total_points}")
    print(f"目标异常数据点数: {target_anomaly_points}")
    print(f"目标异常片段数: {n_anomaly_fragments}")
    print(f"每个区域分配的异常片段数: {fragments_per_zone}，有{extra_fragments}个区域多分1个")
    print(f"预期异常比例: {anomaly_ratio}")

    k = 200  # 候选片段数
    injected_fragments = 0
    zone_injected_count = np.zeros(n_zones, dtype=int)
    max_retry = 1000

    for zone in range(n_zones):
        n_frag = fragments_per_zone + (1 if zone < extra_fragments else 0)
        retry = 0
        while zone_injected_count[zone] < n_frag and retry < max_retry:
            target_slot = np.random.randint(0, total_slots - 2)
            if np.any(anomaly_labels[target_slot:target_slot+3, zone] == 1):
                retry += 1
                continue
            target_fragment = data[target_slot:target_slot+3, zone, :]
            candidate_zones = [z for z in range(n_zones) if z != zone]
            candidate_fragments = []
            for _ in range(k):
                candidate_zone = np.random.choice(candidate_zones)
                candidate_slot = np.random.randint(0, total_slots - 2)
                if candidate_zone == zone and abs(candidate_slot - target_slot) < 3:
                    continue
                candidate_fragment = data[candidate_slot:candidate_slot+3, candidate_zone, :]
                distance = np.linalg.norm(target_fragment - candidate_fragment)
                candidate_fragments.append((distance, candidate_slot, candidate_zone))
            if not candidate_fragments:
                retry += 1
                continue
            best_candidate = max(candidate_fragments, key=lambda x: x[0])
            _, best_slot, best_zone = best_candidate
            # 替换数据并打上标签（只标记区域-时间组合，不考虑特征维度）
            data_with_anomalies[target_slot:target_slot+3, zone, :] = data[best_slot:best_slot+3, best_zone, :]
            anomaly_labels[target_slot:target_slot+3, zone] = 1
            zone_injected_count[zone] += 1
            injected_fragments += 1
            if injected_fragments % 100 == 0:
                print(f"已注入异常片段数: {injected_fragments}/{n_anomaly_fragments}")
            retry = 0
        if retry >= max_retry:
            print(f"区域{zone}注入异常片段失败次数过多，已跳过。实际注入: {zone_injected_count[zone]}")

    # 统计异常点数（不乘以特征数）
    actual_anomaly_points = np.sum(anomaly_labels == 1)
    actual_ratio = actual_anomaly_points / total_points
    print(f"\n=== 优化异常注入完成 ===")
    print(f"实际注入的异常片段数: {injected_fragments}")
    print(f"实际异常点数(不含特征): {actual_anomaly_points}")
    print(f"实际异常比例: {actual_ratio:.4f}")
    print(f"每个区域实际注入片段数: {zone_injected_count}")

    return data_with_anomalies, anomaly_labels

# ========== 智能时空均值填补函数 ==========
def smart_fill_nan(values):
    filled = values.copy()
    T, Z = filled.shape
    for t in range(T):
        for z in range(Z):
            if np.isnan(filled[t, z]):
                # 1. 相邻时间均值
                time_neighbors = []
                for dt in [-2, -1, 1, 2]:
                    nt = t + dt
                    if 0 <= nt < T and not np.isnan(filled[nt, z]):
                        time_neighbors.append(filled[nt, z])
                if time_neighbors:
                    filled[t, z] = np.mean(time_neighbors)
                    continue
                # 2. 相邻空间均值
                space_neighbors = []
                for dz in [-2, -1, 1, 2]:
                    nz = z + dz
                    if 0 <= nz < Z and not np.isnan(filled[t, nz]):
                        space_neighbors.append(filled[t, nz])
                if space_neighbors:
                    filled[t, z] = np.mean(space_neighbors)
                    continue
                # 3. 本区域历史均值
                region_mean = np.nanmean(filled[:, z])
                if not np.isnan(region_mean):
                    filled[t, z] = region_mean
                    continue
                # 4. 全局均值
                global_mean = np.nanmean(filled)
                if not np.isnan(global_mean):
                    filled[t, z] = global_mean
                    continue
                # 5. 最后用0
                filled[t, z] = 0
    return filled
# ========== END ==========

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
    # 读取taxi zones数据，先得到曼哈顿区域ID
    zone_lookup = pd.read_csv("data/taxi _zone_lookup.csv")
    manhattan_ids = zone_lookup[zone_lookup['Borough'] == 'Manhattan']['LocationID'].tolist()
    print(f"曼哈顿区域ID数量: {len(manhattan_ids)}，ID列表: {manhattan_ids}")
    valid_zones = manhattan_ids  # 后续所有valid_zones都用曼哈顿ID

    # 加载邻接矩阵，并裁剪为曼哈顿子矩阵
    adj_data = np.load("data/static_adjacency.npz")
    adj = adj_data['connectivity']
    dist = adj_data['distance']
    poi_sim = adj_data['poi_similarity']
    manhattan_indices = [i-1 for i in manhattan_ids]
    adj = adj[np.ix_(manhattan_indices, manhattan_indices)]
    dist = dist[np.ix_(manhattan_indices, manhattan_indices)]
    poi_sim = poi_sim[np.ix_(manhattan_indices, manhattan_indices)]
    print(f"曼哈顿子矩阵形状: adj={adj.shape}, dist={dist.shape}, poi_sim={poi_sim.shape}")

    # 读取出租车数据
    df_jan = pd.read_parquet("data/yellow_tripdata_2023-01.parquet")
    df_feb = pd.read_parquet("data/yellow_tripdata_2023-02.parquet")
    df = pd.concat([df_jan, df_feb], ignore_index=True)
    print("原始数据形状:", df.shape)

    # 先转为datetime
    df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    # 只保留2023年1月和2月
    df = df[(df['pickup_datetime'].dt.year == 2023) & (df['pickup_datetime'].dt.month.isin([1, 2]))]
    # 只保留曼哈顿区域的出发和到达
    df = df[df['PULocationID'].isin(valid_zones) & df['DOLocationID'].isin(valid_zones)]
    print("只保留曼哈顿后数据形状:", df.shape)
    # 只保留每天7:00-23:50的数据
    df['slot_time'] = df['pickup_datetime'].dt.time
    from datetime import time
    df = df[(df['slot_time'] >= time(7,0)) & (df['slot_time'] <= time(23,50))]
    print("只保留7:00-23:50后数据形状:", df.shape)

    # 时间槽处理
    df['time_slot'] = df['pickup_datetime'].dt.floor('10min')
    df['day'] = df['pickup_datetime'].dt.date

    time_slots_per_day = 17 * 6  # 7:00~23:50 共17小时
    all_days = sorted(df['day'].unique())

    # 统计流入/流出
    inflow_data = df.groupby(['day', 'time_slot', 'DOLocationID']).size().unstack(fill_value=np.nan)
    outflow_data = df.groupby(['day', 'time_slot', 'PULocationID']).size().unstack(fill_value=np.nan)

    # 确保所有曼哈顿区域都存在
    for zone in valid_zones:
        if zone not in inflow_data.columns:
            inflow_data[zone] = np.nan
        if zone not in outflow_data.columns:
            outflow_data[zone] = np.nan
    inflow_data = inflow_data[sorted(valid_zones)]
    outflow_data = outflow_data[sorted(valid_zones)]

    # 补全时间槽
    inflow_data = inflow_data.reset_index()
    outflow_data = outflow_data.reset_index()
    inflow_data['day'] = pd.to_datetime(inflow_data['day']).dt.date
    outflow_data['day'] = pd.to_datetime(outflow_data['day']).dt.date
    inflow_data = inflow_data.sort_values(['day', 'time_slot'])
    outflow_data = outflow_data.sort_values(['day', 'time_slot'])

    complete_inflow_data = []
    complete_outflow_data = []
    for day in all_days:
        day_inflow = inflow_data[inflow_data['day'] == day].copy()
        if len(day_inflow) < time_slots_per_day:
            time_slots = pd.date_range(
                start=pd.Timestamp(day) + pd.Timedelta(hours=7),
                end=pd.Timestamp(day) + pd.Timedelta(hours=23, minutes=50),
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
            )
            complete_inflow_data.append(complete_day)
        else:
            complete_inflow_data.append(day_inflow)
        day_outflow = outflow_data[outflow_data['day'] == day].copy()
        if len(day_outflow) < time_slots_per_day:
            time_slots = pd.date_range(
                start=pd.Timestamp(day) + pd.Timedelta(hours=7),
                end=pd.Timestamp(day) + pd.Timedelta(hours=23, minutes=50),
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
            )
            complete_outflow_data.append(complete_day)
        else:
            complete_outflow_data.append(day_outflow)
    inflow_data = pd.concat(complete_inflow_data, ignore_index=True)
    outflow_data = pd.concat(complete_outflow_data, ignore_index=True)

    # 用历史均值填充缺失值（而不是0）
    inflow_values = inflow_data.iloc[:, 2:].values.astype(float)
    outflow_values = outflow_data.iloc[:, 2:].values.astype(float)
    # 智能时空均值填补
    inflow_values = smart_fill_nan(inflow_values)
    outflow_values = smart_fill_nan(outflow_values)
    # 归一化：对每个区域单独归一化
    scaler_in = MinMaxScaler()
    scaler_out = MinMaxScaler()
    inflow_values = scaler_in.fit_transform(inflow_values)
    outflow_values = scaler_out.fit_transform(outflow_values)
    # 归一化后再检查一遍
    if np.any(np.isnan(inflow_values)):
        print("归一化后inflow仍有NaN，全部用0填补")
        inflow_values[np.isnan(inflow_values)] = 0
    if np.any(np.isnan(outflow_values)):
        print("归一化后outflow仍有NaN，全部用0填补")
        outflow_values[np.isnan(outflow_values)] = 0

    # 合并流入和流出
    flow_values = np.stack([inflow_values, outflow_values], axis=-1)
    print(f"合并后数据形状 (三维): {flow_values.shape}")

    # 划分训练/测试集
    n_train_slots = 14 * time_slots_per_day
    n_test_slots = 14 * time_slots_per_day
    X_train = flow_values[:n_train_slots]
    X_test = flow_values[n_train_slots : n_train_slots + n_test_slots]
    # 标签初始化为二维 shape: (时间槽数, 区域数)
    y_train = np.zeros((X_train.shape[0], X_train.shape[1]), dtype=int)
    y_test = np.zeros((X_test.shape[0], X_test.shape[1]), dtype=int)
    print(f"\n数据集划分:")
    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")

    # 剔除0值比例大于0.7的区域（在异常注入之前）
    zero_ratio_per_zone = (X_train == 0).sum(axis=(0, 2)) / (X_train.shape[0] * X_train.shape[2])
    active_zones = np.where(zero_ratio_per_zone < 0.7)[0]
    print('活跃区域索引:', active_zones)
    print('活跃区域数量:', len(active_zones))
    # 只对数据做区域裁剪
    X_train = X_train[:, active_zones, :]
    X_test = X_test[:, active_zones, :]
    adj = adj[np.ix_(active_zones, active_zones)]
    dist = dist[np.ix_(active_zones, active_zones)]
    poi_sim = poi_sim[np.ix_(active_zones, active_zones)]
    print(f'最终活跃区域邻接矩阵形状: adj={adj.shape}, dist={dist.shape}, poi_sim={poi_sim.shape}')

    # 用裁剪后的shape初始化标签（二维）
    y_train = np.zeros((X_train.shape[0], X_train.shape[1]), dtype=int)
    y_test = np.zeros((X_test.shape[0], X_test.shape[1]), dtype=int)

    # 保存清洗好但未注入异常的训练数据
    np.save('data/datanew1/normalized_train_clean.npy', X_train)
    print('已保存清洗但未注入异常的训练数据到: data/datanew1/normalized_train_clean.npy')

    # 异常注入（对训练集和测试集都注入异常）
    if args.inject_anomaly:
        print("\n开始注入异常到训练集...")
        X_train, y_train = inject_anomalies(X_train, anomaly_ratio=args.anomaly_ratio)
        print("\n开始注入异常到测试集...")
        X_test, y_test = inject_anomalies(X_test, anomaly_ratio=args.anomaly_ratio)

    # 返回数据，并返回active_zones索引
    return X_train, X_test, (adj, dist, poi_sim), y_train, y_test, active_zones

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
    parser.add_argument('--anomaly_ratio', type=float, default=0.05, help='异常比例')
    args = parser.parse_args()
    
    # 加载数据集
    X_train, X_test, (adj, dist, poi_sim), y_train, y_test, active_zones = load_dataset(args)
    
    print("\n最终数据形状:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"adj shape: {adj.shape}")
    print(f"dist shape: {dist.shape}")
    print(f"poi_sim shape: {poi_sim.shape}")
    if y_train is not None:
        print(f"y_train shape: {y_train.shape}")  # 应该是 (时间槽数, 区域数)
    if y_test is not None:
        print(f"y_test shape: {y_test.shape}")  # 应该是 (时间槽数, 区域数)

    save_dir = 'data/datanew1'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'normalized_train_injected.npy')
    np.save(save_path, X_train)
    print(f'已保存归一化且注入异常后的训练集到: {save_path}')
    # 保存标签
    if y_train is not None:
        label_path = os.path.join(save_dir, 'anomaly_labels_train_injected.npy')
        np.save(label_path, y_train)
        print(f'已保存对应标签到: {label_path}') 
    # 保存active_zones索引
    az_path = os.path.join(save_dir, 'active_zones.npy')
    np.save(az_path, active_zones)
    print(f'已保存活跃区域索引到: {az_path}') 

    # ====== 新增：保存测试集和测试集标签 ======
    test_save_path = os.path.join(save_dir, 'normalized_test_injected.npy')
    np.save(test_save_path, X_test)
    print(f'已保存归一化后的测试集到: {test_save_path}')
    if y_test is not None:
        test_label_path = os.path.join(save_dir, 'anomaly_labels_test_injected.npy')
        np.save(test_label_path, y_test)
        print(f'已保存测试集标签到: {test_label_path}') 