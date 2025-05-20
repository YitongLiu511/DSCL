import numpy as np
from STAnomalyFormer.interface.estimator import STAnomalyFormerDetector_v4, TemporalTSFMDetector
from STAnomalyFormer.interface.utils import recall_k
from data.load_nyc import load_dataset
import torch
from sklearn.metrics import roc_auc_score

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--n_day', type=int, default=14)  # 训练集和测试集都是14天
parser.add_argument('--normalize', action='store_true', default=True)  # 默认使用min-max归一化
parser.add_argument('--poi_normalize', action='store_true', default=True)  # 默认使用POI归一化

parser.add_argument('--delay', action='store_true')
parser.add_argument('--lag', type=int, default=6)
parser.add_argument('--vol', action='store_true')
parser.add_argument('--threshold', default=0.5, type=float)
parser.add_argument('--attr', action='store_true')
parser.add_argument('--k', default=None, type=int)

parser.add_argument('--mask_ratio', default=0.4, type=float)
parser.add_argument('--patch_len', default=12, type=int)  # patch大小设为12
parser.add_argument('--stride', default=6, type=int)
parser.add_argument('--d_model', default=128, type=int)  # 隐藏状态维度128
parser.add_argument('--n_heads', default=16, type=int)  # 时间注意力头数16
parser.add_argument('--s_heads', default=8, type=int)   # 空间注意力头数8
parser.add_argument('--t_half', action='store_true')
parser.add_argument('--s_half', action='store_true')
parser.add_argument('--n_gcn', default=3, type=int)  # GCN层数3
parser.add_argument('--lr', default=0.001, type=float)  # 学习率0.001
parser.add_argument('--epochs', default=100, type=int)  # 训练轮数100
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--repeat', default=1, type=int)  # 独立运行1次
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--no_recon', action='store_true')
parser.add_argument('--no_const', action='store_true')
parser.add_argument('--diff_const', action='store_true')
parser.add_argument('--dynamic_only', action='store_true')
parser.add_argument('--static_only', action='store_true')
parser.add_argument('--early_stopping', action='store_true')
parser.add_argument('--temperature', type=float, default=50.0)
parser.add_argument('--anormly_ratio', type=float, default=0.1)

parser.add_argument('--cuda', action='store_true', default=True)  # 默认使用GPU

args = parser.parse_args()
print("\n实验参数设置:")
print(f"数据集: NYC-Taxi")
print(f"训练/测试天数: {args.n_day}天")
print(f"数据归一化: {'启用' if args.normalize else '禁用'}")
print(f"POI归一化: {'启用' if args.poi_normalize else '禁用'}")
print(f"模型参数:")
print(f"- 隐藏状态维度: {args.d_model}")
print(f"- 时间注意力头数: {args.n_heads}")
print(f"- 空间注意力头数: {args.s_heads}")
print(f"- GCN层数: {args.n_gcn}")
print(f"- Patch大小: {args.patch_len}")
print(f"训练设置:")
print(f"- 学习率: {args.lr}")
print(f"- 训练轮数: {args.epochs}")
print(f"- 独立运行次数: {args.repeat}")
print(f"- 设备: {'GPU' if args.cuda else 'CPU'}\n")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True

score_list = []
X, val_X, test_X, (adj, dist, poi_sim), y = load_dataset(args)

# 在读取数据后，定义全局区域编号（例如，从X的shape中提取，假设X的shape为 (263, 144, 14)，即全局区域编号为0,1,…,262）
global_ids = list(range(X.shape[0]))
print("全局区域编号列表（示例）:", global_ids[: 5])

# 打印原始shape
print("原始数据shape:", X.shape)

# 从 (14, 144, 263) 变成 (263, 144, 14)
X = X.transpose(2, 1, 0)        # 让区域数在第一维，时间槽在第二维，天数在第三维
val_X = val_X.transpose(2, 1, 0)
test_X = test_X.transpose(2, 1, 0)

# 修改标签转置
# y的原始形状是 (1, n_time_slots, n_zones, 2)
# 需要转置为 (n_zones, n_time_slots, 1, 2)
y = y.transpose(2, 1, 0, 3)  # 转置为 (区域, 时间, 天数, 异常类型)

# 打印调整后的shape
print("调整后数据shape:", X.shape)
print("调整后标签shape:", y.shape)

# 打印标签信息
print("\n标签统计:")
print(f"总单元格数: {y.size}")
print(f"正常单元格数: {np.sum(y == 0)}")
print(f"异常单元格数: {np.sum(y == 1)}")
print(f"异常比例: {np.mean(y == 1) * 100:.2f}%")

# 计算每个区域的异常时间槽比例（考虑所有天数）
region_anomaly_ratio = np.zeros(263)
for i in range(263):
    region_anomaly_ratio[i] = np.sum(y[i, :, :, 0] == 1) / (y.shape[1] * y.shape[2])

# 计算每个时间槽的异常区域比例（考虑所有天数）
time_anomaly_ratio = np.zeros(144)
for t in range(144):
    time_anomaly_ratio[t] = np.sum(y[:, t, :, 0] == 1) / (y.shape[0] * y.shape[2])

print("\n原始异常比例统计:")
print(f"区域异常比例范围: [{np.min(region_anomaly_ratio):.4f}, {np.max(region_anomaly_ratio):.4f}]")
print(f"时间异常比例范围: [{np.min(time_anomaly_ratio):.4f}, {np.max(time_anomaly_ratio):.4f}]")

# 设置区域异常标签 - 确保异常区域数量符合比例
expected_region_anomalies = int(263 * args.anormly_ratio)  # 应该约26个异常区域
region_threshold = np.percentile(region_anomaly_ratio, 100 * (1 - args.anormly_ratio))
print(f"\n区域异常阈值计算:")
print(f"预期异常区域数: {expected_region_anomalies}")
print(f"区域异常阈值: {region_threshold:.4f}")
print(f"超过阈值的区域数: {np.sum(region_anomaly_ratio > region_threshold)}")

# 设置时间异常标签 - 确保异常时间槽数量符合比例
expected_time_anomalies = int(144 * args.anormly_ratio)  # 应该约14个异常时间槽
time_threshold = np.percentile(time_anomaly_ratio, 100 * (1 - args.anormly_ratio))
print(f"\n时间异常阈值计算:")
print(f"预期异常时间槽数: {expected_time_anomalies}")
print(f"时间异常阈值: {time_threshold:.4f}")
print(f"超过阈值的时间槽数: {np.sum(time_anomaly_ratio > time_threshold)}")

# 新建标签，shape为(区域, 时间, 天数, 2)
y_reshaped = np.zeros((263, 144, 14, 2))

print("\n开始注入异常...")

# 1. 时间异常注入
print("\n时间异常注入:")
# 计算每个区域的历史最大流量
max_flows = np.max(X, axis=(1, 2))  # shape: (263,)
mu = 0.5  # 设置阈值μ为0.5，即限制流量不超过历史最大值的50%

# 选择10%的区域注入时间异常
n_anomaly_regions = int(263 * args.anormly_ratio)  # 约26个区域
anomaly_region_indices = np.random.choice(263, size=n_anomaly_regions, replace=False)

for i in anomaly_region_indices:
    # 对该区域的所有时间点和天数进行限制
    X[i] = np.minimum(X[i], mu * max_flows[i])
    # 标记为时间异常
    y_reshaped[i, :, :, 0] = 1
    print(f"区域 {i} 被注入时间异常，历史最大流量: {max_flows[i]:.2f}，限制后最大流量: {mu * max_flows[i]:.2f}")

# 2. 空间异常注入
print("\n空间异常注入:")
# 选择10%的时间槽注入空间异常
n_anomaly_times = int(144 * args.anormly_ratio)  # 约14个时间槽
anomaly_time_indices = np.random.choice(144, size=n_anomaly_times, replace=False)

for t in anomaly_time_indices:
    # 对每个时间点，选择10%的区域注入空间异常
    n_anomaly_regions = int(263 * args.anormly_ratio)
    anomaly_regions = np.random.choice(263, size=n_anomaly_regions, replace=False)
    
    for i in anomaly_regions:
        # 随机选择k个其他区域
        k = int(263 * 0.1)  # k=10%的节点数
        other_regions = np.random.choice([j for j in range(263) if j != i], size=k, replace=False)
        
        # 计算与当前区域流量差异最大的区域
        current_flow = X[i, t]
        max_diff = 0
        max_diff_region = None
        
        for j in other_regions:
            diff = np.linalg.norm(current_flow - X[j, t])
            if diff > max_diff:
                max_diff = diff
                max_diff_region = j
        
        # 替换流量
        X[i, t] = X[max_diff_region, t]
        # 标记为空间异常
        y_reshaped[i, t, :, 1] = 1
        print(f"时间槽 {t} 区域 {i} 被注入空间异常，替换为区域 {max_diff_region} 的流量")

# 打印调整后的标签信息
print("\n调整后的标签分布:")
print(f"区域异常标签:")
print(f"- 异常区域数: {len(anomaly_region_indices)} (预期: {n_anomaly_regions})")
print(f"- 异常区域比例: {len(anomaly_region_indices) / 263 * 100:.2f}%")
print(f"时间异常标签:")
print(f"- 异常时间槽数: {len(anomaly_time_indices)} (预期: {n_anomaly_times})")
print(f"- 异常时间槽比例: {len(anomaly_time_indices) / 144 * 100:.2f}%")
print(f"标签形状: {y_reshaped.shape}")
print(f"总异常样本数: {np.sum(y_reshaped == 1)}")
print(f"总正常样本数: {np.sum(y_reshaped == 0)}")
print(f"总体异常比例: {np.mean(y_reshaped == 1) * 100:.2f}%")

# 打印每个维度的异常分布
print("\n异常分布详情:")
print("区域异常分布:")
print(f"- 异常区域数量: {len(anomaly_region_indices)}")
print(f"- 异常区域比例: {len(anomaly_region_indices)/263*100:.2f}%")
print("时间异常分布:")
print(f"- 异常时间槽数量: {len(anomaly_time_indices)}")
print(f"- 异常时间槽比例: {len(anomaly_time_indices)/144*100:.2f}%")

# 确保输入数据和标签的维度匹配
print("\n维度检查:")
print(f"输入数据X形状: {X.shape}")
print(f"标签y_reshaped形状: {y_reshaped.shape}")
print(f"验证集形状: {val_X.shape}")
print(f"测试集形状: {test_X.shape}")

# 在计算邻接矩阵后，增加映射字典，将全局区域编号映射到邻接矩阵索引（0到262）
# 假设全局区域编号（例如，pickup_location_id 或 dropoff_location_id）存储在全局变量 global_ids 中，且邻接矩阵 adj 的 shape 为 (263, 263)
# 如果 global_ids 未定义，请根据数据预处理部分（例如，读取数据时）进行定义，例如：
# global_ids = sorted(df['pickup_location_id'].unique())
global_to_adj_idx = {global_id: i for i, global_id in enumerate(sorted(global_ids))}
print("全局区域编号到邻接矩阵索引的映射字典（示例）:", dict(list(global_to_adj_idx.items())[: 5]))

for t in range(args.repeat):
    print(f"{t+1}-th experiment:")
    
    # 初始化模型
    model = STAnomalyFormerDetector_v4(
        d_in=X.shape[-1],
        d_model=args.d_model,
        dim_k=args.d_model // args.n_heads,
        dim_v=args.d_model // args.n_heads,
        n_heads=args.n_heads,
        n_gcn=args.n_gcn,
        device='cuda' if args.cuda else 'cpu',
        epoch=args.epochs,
        lr=args.lr,
        verbose=args.verbose,
        contamination=args.anormly_ratio,
    )
    
    # 训练模型
    model.fit(X, torch.stack([torch.tensor(1 - adj), torch.tensor(dist), torch.tensor(poi_sim)]), (val_X, y_reshaped))
    
    # 评估模型
    score = model.decision_function(test_X)
    if isinstance(score, dict):
        # 获取时间异常和空间异常的分数
        temporal_scores = score['time_scores'].detach().cpu().numpy()
        spatial_scores = score['region_scores'].detach().cpu().numpy()
        
        # 计算阈值
        temporal_thresh = np.percentile(temporal_scores, 100 * (1 - args.anormly_ratio))
        spatial_thresh = np.percentile(spatial_scores, 100 * (1 - args.anormly_ratio))
        
        # 生成预测
        temporal_pred = (temporal_scores >= temporal_thresh).astype(int)
        spatial_pred = (spatial_scores >= spatial_thresh).astype(int)
        
        # 确保标签维度匹配
        y_temporal = y_reshaped[:, :, :, 0].reshape(-1)  # 时间异常标签
        y_spatial = y_reshaped[:, :, :, 1].reshape(-1)   # 空间异常标签
        
        # 调整预测维度以匹配标签
        temporal_pred = temporal_pred.reshape(-1)
        spatial_pred = spatial_pred.reshape(-1)
        
        # 计算评估指标
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # 时间异常评估
        temporal_precision = precision_score(y_temporal, temporal_pred, zero_division=0)
        temporal_recall = recall_score(y_temporal, temporal_pred, zero_division=0)
        temporal_f1 = f1_score(y_temporal, temporal_pred, zero_division=0)
        
        # 空间异常评估
        spatial_precision = precision_score(y_spatial, spatial_pred, zero_division=0)
        spatial_recall = recall_score(y_spatial, spatial_pred, zero_division=0)
        spatial_f1 = f1_score(y_spatial, spatial_pred, zero_division=0)
        
        print("\n时间异常评估:")
        print(f"Precision: {temporal_precision:.4f}")
        print(f"Recall: {temporal_recall:.4f}")
        print(f"F1-score: {temporal_f1:.4f}")
        
        print("\n空间异常评估:")
        print(f"Precision: {spatial_precision:.4f}")
        print(f"Recall: {spatial_recall:.4f}")
        print(f"F1-score: {spatial_f1:.4f}")
        
        # 保存结果
        score_list.append([
            temporal_precision, temporal_recall, temporal_f1,
            spatial_precision, spatial_recall, spatial_f1
        ])
    else:
        print("警告：模型输出不是字典格式，无法进行分类型评估")

# 计算平均指标
if score_list:
    mean_scores = np.array(score_list).mean(0)
    print("\n平均评估指标:")
    print(f"时间异常 - Precision: {mean_scores[0]:.4f}, Recall: {mean_scores[1]:.4f}, F1: {mean_scores[2]:.4f}")
    print(f"空间异常 - Precision: {mean_scores[3]:.4f}, Recall: {mean_scores[4]:.4f}, F1: {mean_scores[5]:.4f}")
else:
    print("\n没有有效的评估结果")

with open('test.txt', 'a+') as f:
    if score_list:
        f.write(str(list(mean_scores)) + '\n')
    else:
        f.write("No valid results\n")