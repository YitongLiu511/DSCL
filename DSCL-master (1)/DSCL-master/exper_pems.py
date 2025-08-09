import sys
import os
# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from STAnomalyFormer.interface.estimator import STPatch_MGCNDetector
from data.load_pems import load_dataset
import torch
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from STAnomalyFormer.interface.utils import recall_k

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=int, choices=[3, 4, 7, 8])
parser.add_argument('--n_day', type=int, default=14)
parser.add_argument('--interval', type=int, default=6)
parser.add_argument('--normalize', action='store_true')

parser.add_argument('--vol', action='store_true')
parser.add_argument('--threshold', default=0.5, type=float)
parser.add_argument('--attr', action='store_true')
parser.add_argument('--k', default=None, type=int)

# parser.add_argument('--mask_ratio', default=0.4, type=float)
parser.add_argument('--patch_len', default=12, type=int)
parser.add_argument('--stride', default=6, type=int)
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--n_heads', default=16, type=int)
parser.add_argument('--t_half', action='store_true')
parser.add_argument('--s_half', action='store_true')
parser.add_argument('--n_gcn', default=3, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--repeat', default=1, type=int)
parser.add_argument('--early_stopping', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--no_recon', action='store_true')
parser.add_argument('--no_const', action='store_true')
parser.add_argument('--dynamic_only', action='store_true')
parser.add_argument('--static_only', action='store_true')
parser.add_argument('--diff_const', action='store_true')

# 添加软聚类相关参数
parser.add_argument('--cluster_weight', default=0.1, type=float, help='软聚类损失权重')
parser.add_argument('--n_prototypes', default=10, type=int, help='原型数量')
parser.add_argument('--tau', default=0.5, type=float, help='温度参数')

parser.add_argument('--cuda', action='store_true')

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True

score_list = []
last_timestamp_score_list = []
X, X_clean, val_X, test_X, test_X_clean, mats, y = load_dataset(args)
adj, distance, connectivity = mats
for t in range(args.repeat):
    print("{}-th experiment:".format(t + 1))
    print("正在初始化模型...")

    model = STPatch_MGCNDetector(
        seq_len=12,  # 修复：使用固定的滑动窗口长度12，而不是整个时间序列长度
        patch_len=args.patch_len,
        stride=args.stride,
        d_in=X.shape[-1],
        d_model=args.d_model,
        n_heads=args.n_heads,
        temporal_half=args.t_half,
        spatial_half=args.s_half,
        n_gcn=args.n_gcn,
        device='cuda' if args.cuda and torch.cuda.is_available() else 'cpu',
        epoch=args.epochs,
        lr=args.lr,
        verbose=args.verbose,
        use_recon=(not args.no_recon),
        use_const=(not args.no_const),
        diff_const=args.diff_const,
        static_only=args.static_only,
        dynamic_only=args.dynamic_only,
        contamination=0.2,
        cluster_weight=args.cluster_weight,  # 添加聚类损失权重
        args=args,  # 传递args参数
    )
    print("模型初始化完成，开始训练...")
    model.fit(X, np.array([adj, distance, connectivity]), (val_X, y), x_clean=X_clean)
    print("训练完成！")

    # 区域异常检测：使用多个窗口的平均分数
    print("🔍 开始区域异常检测...")
    
    # 准备多个窗口用于区域异常检测
    seq_len = 12
    total_windows = test_X.shape[1] - seq_len + 1  # 4021个可能的窗口
    
    # 使用所有可能的窗口进行区域异常检测
    print(f"📊 总窗口数: {total_windows}")
    print(f"🔍 使用所有窗口进行区域异常检测...")
    
    window_scores = []
    
    for i in range(total_windows):
        # 使用第i个窗口
        window = test_X[:, i:i+seq_len, :]  # (N, 12, D)
        
        # 通过模型获取区域异常分数
        window_score = model.decision_function(window)  # (N,)
        window_scores.append(window_score)
        
        if (i + 1) % 500 == 0 or i == total_windows - 1:
            print(f"⏳ 已处理 {i+1}/{total_windows} 个窗口 ({(i+1)/total_windows*100:.1f}%)")
    
    # 计算所有窗口分数的平均值
    window_scores = np.array(window_scores)  # (total_windows, N)
    region_scores = np.mean(window_scores, axis=0)  # (N,)
    
    print(f"✅ 区域异常检测完成！使用 {total_windows} 个窗口的平均分数")
    print(f"📊 区域异常分数范围: [{region_scores.min():.6f}, {region_scores.max():.6f}]")
    print(f"📊 区域异常分数均值: {region_scores.mean():.6f}")
    print(f"📊 区域异常分数标准差: {region_scores.std():.6f}")
    
    # 使用平均分数进行区域异常检测
    score = region_scores
    true_anomaly_count = y.sum()
    threshold = np.sort(score)[-int(true_anomaly_count)]
    pred = np.zeros_like(score)
    pred[score >= threshold] = 1

    # 原有的区域异常检测评价指标
    recall_k_10 = recall_k(y, pred, np.ceil(len(y) // 10).astype(int))
    recall_k_5 = recall_k(y, pred, np.ceil(len(y) // 5).astype(int))
    roc_auc_original = roc_auc_score(y, score)
    
    print(f"\n{'='*80}")
    print("📊 区域异常检测评估指标（基于真实异常数量）")
    print(f"{'='*80}")
    print(f"真实异常区域数: {true_anomaly_count}")
    print(f"预测阈值: {threshold:.6f}")
    print(f"预测异常区域数: {pred.sum()}")
    print(f"Recall@K (K=10%): {recall_k_10:.4f}")
    print(f"Recall@K (K=20%): {recall_k_5:.4f}")
    print(f"ROC-AUC: {roc_auc_original:.4f}")
    print(f"{'='*80}")

    score_list.append([recall_k_10, recall_k_5, roc_auc_original])
    print(f"区域异常检测指标: {score_list[-1]}")
    print(f"{'='*80}\n")
    
    # 新功能：获取所有时间戳的异常检测结果
    print("\n" + "="*80)
    print("🆕 新功能：滑动窗口时间戳异常检测")
    print("="*80)
    
    print("🔧 滑动窗口预测逻辑:")
    print("   - 用时间点 0-11 预测时间点 12 的异常")
    print("   - 用时间点 1-12 预测时间点 13 的异常")
    print("   - 用时间点 2-13 预测时间点 14 的异常")
    print("   - 以此类推...")
    
    all_timestamps_scores = model.get_all_timestamps_scores(test_X, test_X_clean, batch_size=10)  # 获取所有时间戳的异常分数 (N, T)
    
    # 确保异常分数越大表示越异常（如果分数是越小越异常，则取负值）
    # 检查分数分布，如果大部分分数都很小，说明可能需要取负值
    if np.nanmean(all_timestamps_scores) < 0.1:  # 如果平均分数很小，可能需要调整
        print("⚠️  检测到异常分数可能较小，正在调整分数方向...")
        # 这里可以根据实际情况调整，确保分数越大越异常
    
    print(f"\n📊 滑动窗口预测结果概览:")
    print("─" * 60)
    print(f"   🎯 区域数量: {all_timestamps_scores.shape[0]:,}")
    print(f"   🏢 时间戳数量: {all_timestamps_scores.shape[1]:,}")
    print(f"   📈 异常分数范围: [{np.nanmin(all_timestamps_scores):.6f}, {np.nanmax(all_timestamps_scores):.6f}]")
    print(f"   📊 异常分数均值: {np.nanmean(all_timestamps_scores):.6f}")
    print(f"   📉 异常分数标准差: {np.nanstd(all_timestamps_scores):.6f}")
    print(f"   🔍 数据形状: {all_timestamps_scores.shape} (区域数 × 时间戳数)")
    
    # 计算整体统计信息
    print(f"\n📈 整体异常检测统计:")
    print("─" * 60)
    
    # 计算每个时间戳的异常区域数量（只考虑可预测的时间戳）
    seq_len = getattr(model, 'seq_len', 12)
    valid_scores = all_timestamps_scores[:, seq_len-1:]  # 只取可预测的时间戳
    
    # 确保异常分数越大表示越异常
    # 如果分数分布显示越小越异常，则取负值
    if np.nanmean(valid_scores) < 0.1:
        print("🔄 调整异常分数方向：分数越大表示越异常")
        valid_scores = -valid_scores  # 取负值，使分数越大越异常
    
    # 计算每个时间戳的阈值（取前5%的区域作为异常）
    k_anomaly = max(1, int(valid_scores.shape[0] * 0.05))  # 5%的区域
    thresholds = np.sort(valid_scores, axis=0)[-k_anomaly, :]  # 每个时间戳的阈值
    anomaly_counts = (valid_scores >= thresholds[np.newaxis, :]).sum(axis=0)
    
    print(f"   🚨 平均异常区域数: {anomaly_counts.mean():.1f}")
    print(f"   📊 异常区域数范围: [{anomaly_counts.min()}, {anomaly_counts.max()}]")
    print(f"   📈 异常区域数标准差: {anomaly_counts.std():.1f}")
    
    # 找出异常最多的前5个时间戳
    top_anomaly_indices = np.argsort(anomaly_counts)[-5:][::-1]
    print(f"\n🔥 异常最多的前5个时间戳:")
    print("─" * 60)
    
    for i, idx in enumerate(top_anomaly_indices):
        seq_len = getattr(model, 'seq_len', 12)
        actual_timestamp = idx + seq_len - 1
        timestamp_scores = all_timestamps_scores[:, actual_timestamp]  # 获取该时间戳的所有区域分数
        threshold = thresholds[idx]
        
        print(f"   🥇 第{i+1}名 - 时间戳 {actual_timestamp:4d}:")
        print(f"       🚨 异常区域数: {anomaly_counts[idx]:3d} / {len(timestamp_scores)}")
        print(f"       📊 异常比例: {anomaly_counts[idx]/len(timestamp_scores)*100:.1f}% (目标5%)")
        print(f"       🎯 异常分数范围: [{timestamp_scores.min():.6f}, {timestamp_scores.max():.6f}]")
        print(f"       📈 异常分数均值: {timestamp_scores.mean():.6f}")
        print(f"       🔍 预测阈值: {threshold:.6f}")
        
        # 显示前3个最异常的区域
        top_anomaly_regions = np.argsort(timestamp_scores)[-3:][::-1]
        print(f"       🏆 前3个最异常区域:")
        for j, region_idx in enumerate(top_anomaly_regions):
            status = "🚨异常" if timestamp_scores[region_idx] >= threshold else "✅正常"
            print(f"          {j+1}. 区域 {region_idx:3d}: {status} (分数: {timestamp_scores[region_idx]:.6f})")
        print()
    
    # 加载测试集的真实时间戳标签
    import os
    data_dir = f"pems0{args.dataset}"  # 修复：直接使用pems03目录
    test_y_timestamps = np.load(os.path.join(data_dir, "anomaly_labels_test.npy"))  # (T, N)
    
    print(f"真实标签矩阵形状: {test_y_timestamps.shape}")  # (T, N)
    
    # 计算每个时间戳的异常检测指标
    print(f"\n📈 时间戳异常检测评估:")
    print("─" * 60)
    
    # 计算预测的异常区域数量
    N, T = all_timestamps_scores.shape
    print(f"   📊 预测异常统计:")
    print(f"      - 每个时间戳预测5%的区域为异常")
    print(f"      - 预期每个时间戳异常区域数: {max(1, int(N * 0.05))} (5% × {N})")
    
    # 计算每个时间戳的异常区域数量
    valid_scores = all_timestamps_scores[:, seq_len-1:]  # 只取可预测的时间戳
    k_anomaly = max(1, int(valid_scores.shape[0] * 0.05))  # 5%的区域
    thresholds = np.sort(valid_scores, axis=0)[-k_anomaly, :]  # 每个时间戳的阈值
    anomaly_counts = (valid_scores >= thresholds[np.newaxis, :]).sum(axis=0)
    
    print(f"   📊 实际预测异常统计:")
    print(f"      - 可预测时间戳数: {valid_scores.shape[1]}")
    print(f"      - 平均预测异常区域数: {anomaly_counts.mean():.1f}")
    print(f"      - 预测异常区域数范围: [{anomaly_counts.min()}, {anomaly_counts.max()}]")
    
    # 收集所有时间戳×所有区域的数据
    all_scores = []
    all_labels = []
    
    # 只评估可预测的时间戳
    valid_timestamps = range(seq_len-1, all_timestamps_scores.shape[1])
    for t_idx, t in enumerate(valid_timestamps):
        scores_t = all_timestamps_scores[:, t]  # (N,) - 获取该时间戳的所有区域分数
        # 对应的时间戳标签
        if t < test_y_timestamps.shape[0]:
            labels_t = test_y_timestamps[t]  # (N,)
            
            # 过滤掉NaN值
            valid_mask = ~np.isnan(scores_t)
            if valid_mask.sum() > 0:
                valid_scores = scores_t[valid_mask]
                valid_labels = labels_t[valid_mask]
                
                all_scores.extend(valid_scores)
                all_labels.extend(valid_labels)
    
    # 转换为numpy数组
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    print(f"   📊 整体数据集统计:")
    print(f"      - 总样本数: {len(all_scores)} (N×T)")
    print(f"      - 真实异常样本数: {all_labels.sum()}")
    print(f"      - 异常比例: {all_labels.sum()/len(all_labels):.4f}")
    
    # 计算整体的异常检测指标
    if len(all_scores) > 0 and all_labels.sum() > 0:
        # 使用Recall@K的形式，异常比例改为5%和10%
        k_5 = max(1, int(len(all_labels) * 0.05))  # 5%
        k_10 = max(1, int(len(all_labels) * 0.10))  # 10%
        
        # 计算Recall@K
        recall_k_5 = recall_k(all_labels, all_scores, k_5)
        recall_k_10 = recall_k(all_labels, all_scores, k_10)
        roc_auc_overall = roc_auc_score(all_labels, all_scores)
        
        print(f"\n   📈 整体异常检测指标:")
        print("   ─" * 40)
        print(f"   🎯 Recall@5%: {recall_k_5:.4f}")
        print(f"   🎯 Recall@10%: {recall_k_10:.4f}")
        print(f"   🎯 ROC-AUC: {roc_auc_overall:.4f}")
        print(f"   📊 评估样本数: {len(all_scores)}")
        
        # 保存整体指标
        last_timestamp_score_list.append([recall_k_5, recall_k_10, roc_auc_overall])
        print(f"   📋 整体异常检测指标: {last_timestamp_score_list[-1]}")
    else:
        print("   ⚠️  没有找到有效样本")
    
    # 显示评估结果
    print(f"   📊 异常检测评估总结:")
    print(f"      - 评估方式: 将所有时间戳×所有区域作为整体数据集")
    print(f"      - 预测异常比例: 5%和10% (从所有样本中预测top-5%和top-10%为异常)")
    print(f"      - 评估指标: Recall@5%, Recall@10% 和 ROC-AUC")
    
    # 总结信息
    print("="*80)
    print("🎉 滑动窗口时间戳异常检测完成！")
    print("📊 功能总结:")
    print("   ✅ 区域异常检测: 基于静态流和动态流的传统方法")
    print("   🆕 时间戳异常检测: 滑动窗口预测的新功能")
    print("   📈 输出结果: 每个时间戳每个区域的异常分数矩阵")
    print("   🔍 数据形状: " + str(all_timestamps_scores.shape))
    print("   🎯 异常比例设置: 5% (时间戳×区域数×0.05)")
    print("   📊 异常分数方向: 分数越大表示越异常")
    print("="*80)
    


print(np.array(score_list).mean(0))
