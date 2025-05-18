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
parser.add_argument('--epochs', default=500, type=int)  # 训练轮数500
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--repeat', default=10, type=int)  # 独立运行10次
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
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True

score_list = []
X, val_X, test_X, (adj, dist, poi_sim), y = load_dataset(args)

# 打印原始shape
print("原始数据shape:", X.shape)

# 从 (14, 144, 263) 变成 (263, 144, 14)
X = X.transpose(2, 1, 0)        # 让区域数在第一维，时间槽在第二维，天数在第三维
val_X = val_X.transpose(2, 1, 0)
test_X = test_X.transpose(2, 1, 0)
y = y.transpose(2, 1, 0)  # 将y也转置成 (区域, 时间, 天数)

# 打印调整后的shape
print("调整后数据shape:", X.shape)

# 打印标签信息
print("\n标签统计:")
print(f"总单元格数: {y.size}")
print(f"正常单元格数: {np.sum(y == 0)}")
print(f"异常单元格数: {np.sum(y == 1)}")
print(f"异常比例: {np.mean(y == 1) * 100:.2f}%")

# 计算每个区域的异常时间槽比例（考虑所有天数）
region_anomaly_ratio = np.zeros(263)
for i in range(263):
    region_anomaly_ratio[i] = np.sum(y[i, :, :] == 1) / (y.shape[1] * y.shape[2])

# 计算每个时间槽的异常区域比例（考虑所有天数）
time_anomaly_ratio = np.zeros(144)
for t in range(144):
    time_anomaly_ratio[t] = np.sum(y[:, t, :] == 1) / (y.shape[0] * y.shape[2])

# 设置区域异常标签 - 确保异常区域数量符合比例
expected_region_anomalies = int(263 * args.anormly_ratio)  # 应该约26个异常区域
region_threshold = np.percentile(region_anomaly_ratio, 100 * (1 - args.anormly_ratio))
# 设置时间异常标签 - 确保异常时间槽数量符合比例
expected_time_anomalies = int(144 * args.anormly_ratio)  # 应该约14个异常时间槽
time_threshold = np.percentile(time_anomaly_ratio, 100 * (1 - args.anormly_ratio))

# 新建标签，shape为(区域, 时间, 天数, 2)
y_reshaped = np.zeros((263, 144, 14, 2))
# 区域异常标签
y_reshaped[:, :, :, 0] = (region_anomaly_ratio > region_threshold)[:, np.newaxis, np.newaxis]
# 时间异常标签
y_reshaped[:, :, :, 1] = (time_anomaly_ratio > time_threshold)[np.newaxis, :, np.newaxis]

# 打印调整后的标签信息
print("\n调整后的标签分布:")
print(f"区域异常标签:")
print(f"- 异常区域数: {np.sum(y_reshaped[0, 0, :, 0] == 1)} (预期: {expected_region_anomalies})")
print(f"- 异常区域比例: {np.sum(y_reshaped[0, 0, :, 0] == 1) / 263 * 100:.2f}%")
print(f"时间异常标签:")
print(f"- 异常时间槽数: {np.sum(y_reshaped[0, :, 0, 1] == 1)} (预期: {expected_time_anomalies})")
print(f"- 异常时间槽比例: {np.sum(y_reshaped[0, :, 0, 1] == 1) / 144 * 100:.2f}%")
print(f"标签形状: {y_reshaped.shape}")
print(f"总异常样本数: {np.sum(y_reshaped == 1)}")
print(f"总正常样本数: {np.sum(y_reshaped == 0)}")
print(f"总体异常比例: {np.mean(y_reshaped == 1) * 100:.2f}%")

# 打印每个维度的异常分布
print("\n异常分布详情:")
print("区域异常分布:")
region_anomalies = np.sum(y_reshaped[0, 0, :, 0] == 1)
print(f"- 异常区域数量: {region_anomalies}")
print(f"- 异常区域比例: {region_anomalies/263*100:.2f}%")
print("时间异常分布:")
time_anomalies = np.sum(y_reshaped[0, :, 0, 1] == 1)
print(f"- 异常时间槽数量: {time_anomalies}")
print(f"- 异常时间槽比例: {time_anomalies/144*100:.2f}%")

# 确保输入数据和标签的维度匹配
print("\n维度检查:")
print(f"输入数据X形状: {X.shape}")
print(f"标签y_reshaped形状: {y_reshaped.shape}")
print(f"验证集形状: {val_X.shape}")
print(f"测试集形状: {test_X.shape}")

for t in range(args.repeat):
    print("{}-th experiment:".format(t + 1))

    model = STAnomalyFormerDetector_v4(
        d_in=X.shape[2],
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
    ).fit(X, torch.stack([torch.tensor(1 - adj), torch.tensor(dist), torch.tensor(poi_sim)]), (val_X, y_reshaped))
    
    # 打印 model 的 __dict__ 以及 model.decision_function 的 __dict__，以查看 model 内部状态
    print("model.__dict__:", model.__dict__)
    print("model.decision_function.__dict__:", model.decision_function.__dict__)
    
    # 打印 model.decision_function 的返回值，以确认返回字典中键名
    print("model.decision_function 返回字典的键名:", model.decision_function(test_X, return_dict=True).keys())
    
    # 获取异常检测结果
    output_dict = model.decision_function(test_X, return_dict=True)
    
    # 获取异常分数和预测
    anomaly_scores = output_dict['anomaly_scores']  # [N, T, 2]
    region_scores = anomaly_scores[:, :, 0]        # [N, T]
    time_scores = anomaly_scores[:, :, 1]          # [N, T]
    
    # 计算区域异常评估指标
    try:
        region_auc = roc_auc_score(y_reshaped[:, :, 0, 0].flatten(), region_scores.cpu().numpy().flatten())
    except ValueError as e:
        print(f"警告：区域异常评估出错 - {str(e)}")
        region_auc = 0.0
    
    # 计算时间异常评估指标
    try:
        time_auc = roc_auc_score(y_reshaped[:, :, 0, 1].flatten(), time_scores.cpu().numpy().flatten())
    except ValueError as e:
        print(f"警告：时间异常评估出错 - {str(e)}")
        time_auc = 0.0
    
    # 计算综合评估指标
    try:
        combined_auc = roc_auc_score(y_reshaped[:, :, 0, :].flatten(), anomaly_scores.cpu().numpy().flatten())
    except ValueError as e:
        print(f"警告：综合评估出错 - {str(e)}")
        combined_auc = 0.0
    
    # 计算区域异常的recall@k
    k1 = np.ceil(263 // 10).astype(int)
    k2 = np.ceil(263 // 5).astype(int)
    region_recall_k1 = recall_k(y_reshaped[:, 0, 0, 0], region_scores.cpu().numpy().flatten(), k1)
    region_recall_k2 = recall_k(y_reshaped[:, 0, 0, 0], region_scores.cpu().numpy().flatten(), k2)
    
    # 计算时间异常的recall@k
    k1 = np.ceil(144 // 10).astype(int)
    k2 = np.ceil(144 // 5).astype(int)
    time_recall_k1 = recall_k(y_reshaped[0, 0, 0, 1], time_scores.cpu().numpy().flatten(), k1)
    time_recall_k2 = recall_k(y_reshaped[0, 0, 0, 1], time_scores.cpu().numpy().flatten(), k2)
    
    score_list.append([
        region_recall_k1, region_recall_k2, region_auc,  # 区域异常指标
        time_recall_k1, time_recall_k2, time_auc,        # 时间异常指标
        combined_auc                                     # 综合指标
    ])
    
    # 打印异常检测结果
    print("\n异常检测结果:")
    print(f"区域异常分数: {region_scores}")
    print(f"时间异常分数: {time_scores}")
    print(f"区域异常标签: {region_scores.cpu().numpy().flatten() > 0}")
    print(f"时间异常标签: {time_scores.cpu().numpy().flatten() > 0}")
    print("\n评估指标:")
    print(f"区域异常 - Recall@k1: {region_recall_k1:.4f}, Recall@k2: {region_recall_k2:.4f}, AUC: {region_auc:.4f}")
    print(f"时间异常 - Recall@k1: {time_recall_k1:.4f}, Recall@k2: {time_recall_k2:.4f}, AUC: {time_auc:.4f}")
    print(f"综合指标 - AUC: {combined_auc:.4f}")

# 计算平均指标
mean_scores = np.array(score_list).mean(0)
print("\n平均评估指标:")
print(f"区域异常 - Recall@k1: {mean_scores[0]:.4f}, Recall@k2: {mean_scores[1]:.4f}, AUC: {mean_scores[2]:.4f}")
print(f"时间异常 - Recall@k1: {mean_scores[3]:.4f}, Recall@k2: {mean_scores[4]:.4f}, AUC: {mean_scores[5]:.4f}")
print(f"综合指标 - AUC: {mean_scores[6]:.4f}")

with open('test.txt', 'a+') as f:
    f.write(str(list(mean_scores)) + '\n')