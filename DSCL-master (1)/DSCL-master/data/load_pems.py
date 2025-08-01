import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import os
from scipy.sparse import csr_matrix, coo_matrix

def load_dataset(args):
    """
    加载PEMS数据集，支持多种邻接/相似度矩阵
    Args:
        args: argparse.Namespace对象，包含所有参数
    Returns:
        X: 训练数据，形状为 (n_nodes, time_slots_per_day, n_days)
        val_X: 验证数据，形状为 (n_nodes, time_slots_per_day, n_days)
        test_X: 测试数据，形状为 (n_nodes, time_slots_per_day, n_days)
        mats: (adj, distance, connectivity) 三种矩阵，均为 (n_nodes, n_nodes)
        y: 标签，形状为 (n_nodes,)
    """
    # 根据数据集编号选择对应的文件夹
    dataset_name = f"PEMS0{args.dataset}"
    # 获取当前文件所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 数据目录为上级目录下的 pems0X
    data_dir = os.path.join(base_dir, "..", f"pems0{args.dataset}")
    data_dir = os.path.abspath(data_dir)

    # 加载交通流量数据（已归一化/异常注入）
    X = np.load(os.path.join(data_dir, "X_anom.npy"))    # shape: (天数*时间步, 节点数)
    test_X_anom = np.load(os.path.join(data_dir, "test_X_anom.npy"))
    
    # 加载clean data
    X_clean = np.load(os.path.join(data_dir, "X_train_clean.npy"))
    test_X_clean = np.load(os.path.join(data_dir, "test_X_clean.npy"))

    # 标签
    y_train = np.load(os.path.join(data_dir, "anomaly_labels_train.npy"))  # shape: (天数*时间步, 节点数)
    y_test = np.load(os.path.join(data_dir, "anomaly_labels_test.npy"))    # shape: (天数*时间步, 节点数)

    # 加载邻接矩阵
    adj_path = os.path.join(data_dir, "adj.npz")
    adj_data = np.load(adj_path)
    row = adj_data['row']
    col = adj_data['col']
    shape = tuple(adj_data['shape'])
    adj = csr_matrix((adj_data['data'], (row, col)), shape=shape).toarray()

    # 兼容distance/connectivity
    distance = None
    connectivity = None
    pems_npz_path = os.path.join(data_dir, f"{dataset_name}.npz")
    if os.path.exists(pems_npz_path):
        pems_npz = np.load(pems_npz_path, allow_pickle=True)
        if 'distance' in pems_npz:
            # 用adj的row/col/shape还原distance方阵
            distance = coo_matrix((pems_npz['distance'], (row, col)), shape=shape).toarray()
        if 'connectivity' in pems_npz:
            # 用adj的row/col/shape还原connectivity方阵
            connectivity = coo_matrix((pems_npz['connectivity'], (row, col)), shape=shape).toarray()

    n_nodes = adj.shape[0]
    # 自动推断天数和时间步
    total_steps = X.shape[0]
    time_slots_per_day = 288
    n_days = total_steps // time_slots_per_day
    # 训练集
    X = X.reshape(-1, n_nodes).T[:, :, None]  # (节点数, 总时间步, 1)
    X_clean = X_clean.reshape(-1, n_nodes).T[:, :, None]  # clean data也做相同的reshape
    val_X = X.copy()  # 这里val_X直接用训练集（如需分割可自行调整）
    # 测试集
    test_X = test_X_anom.reshape(-1, n_nodes).T[:, :, None]
    test_X_clean = test_X_clean.reshape(-1, n_nodes).T[:, :, None]

    # 标签按区域聚合
    y = (y_test != 0).any(axis=0).astype(float)  # 只要某区域有任意时间戳为异常就为1

    print(f"X shape: {X.shape}")
    print(f"val_X shape: {val_X.shape}")
    print(f"test_X shape: {test_X.shape}")
    print(f"adj shape: {adj.shape}")
    if distance is not None:
        print(f"distance shape: {distance.shape}")
    if connectivity is not None:
        print(f"connectivity shape: {connectivity.shape}")
    print(f"y shape: {y.shape}")
    print(f"异常区域数: {int(y.sum())} / {len(y)}")

    X = torch.from_numpy(X).float()
    X_clean = torch.from_numpy(X_clean).float()
    val_X = torch.from_numpy(val_X).float()
    test_X = torch.from_numpy(test_X).float()
    test_X_clean = torch.from_numpy(test_X_clean).float()
    adj = torch.from_numpy(adj).float()
    y = torch.from_numpy(y).float()
    if distance is not None:
        distance = torch.from_numpy(distance).float()
    if connectivity is not None:
        connectivity = torch.from_numpy(connectivity).float()
    mats = (adj, distance, connectivity)
    return X, X_clean, val_X, test_X, test_X_clean, mats, y

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=int, choices=[3, 8])
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--n_day', type=int, default=14)
    args = parser.parse_args()
    X, val_X, test_X, mats, y = load_dataset(args)
    adj, distance, connectivity = mats
    print(f"\n最终数据形状:")
    print(f"X shape: {X.shape}")
    print(f"val_X shape: {val_X.shape}")
    print(f"test_X shape: {test_X.shape}")
    print(f"adj shape: {adj.shape}")
    print(f"distance shape: {distance.shape if distance is not None else 'None'}")
    print(f"connectivity shape: {connectivity.shape if connectivity is not None else 'None'}")
    print(f"y shape: {y.shape}") 