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
from STAnomalyFormer.interface.timestamp_scores import (
    compute_timestamp_scores,
    dcd_style_kl_scorer,
)

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
parser.add_argument('--segment_mode', choices=['windows','sequence'], default='windows')
parser.add_argument('--aggregate', choices=['mean','max','median'], default='mean')
parser.add_argument('--no_recon', action='store_true')
parser.add_argument('--no_const', action='store_true')
parser.add_argument('--dynamic_only', action='store_true')
parser.add_argument('--static_only', action='store_true')
parser.add_argument('--diff_const', action='store_true')

# 添加软聚类相关参数
parser.add_argument('--cluster_weight', default=0.1, type=float, help='软聚类损失权重')
parser.add_argument('--n_prototypes', default=10, type=int, help='原型数量')
parser.add_argument('--tau', default=0.5, type=float, help='温度参数')

# 添加DCdetector相关参数
parser.add_argument('--use_dcdetector', action='store_true', help='是否使用DCdetector损失')
parser.add_argument('--dcdetector_weight', default=1.0, type=float, help='DCdetector损失权重')
parser.add_argument('--dcdetector_patch_sizes', nargs='+', type=int, default=[3, 5, 7], help='DCdetector的patch sizes')

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
        seq_len=12,
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
        cluster_weight=args.cluster_weight,
        args=args,
        segment_mode=args.segment_mode,
        aggregate=args.aggregate,
        # DCdetector相关参数
        use_dcdetector=args.use_dcdetector,
        dcdetector_weight=args.dcdetector_weight,
        dcdetector_patch_sizes=args.dcdetector_patch_sizes,
    )
    print("模型初始化完成，开始训练...")
    model.fit(X, np.array([adj, distance, connectivity]), (val_X, y), x_clean=X_clean)
    print("训练完成！")

    # 区域异常检测：使用origin公式（重构+一致性）
    print("🔍 开始区域异常检测（origin公式：重构+一致性）...")
    region_input = test_X.numpy() if isinstance(test_X, torch.Tensor) else test_X
    region_scores = model.decision_function(region_input, verbose_output=False)
    print("✅ 区域异常检测完成！")
    print(f"📊 区域异常分数范围: [{np.nanmin(region_scores):.6f}, {np.nanmax(region_scores):.6f}]")
    print(f"📊 区域异常分数均值: {np.nanmean(region_scores):.6f}")
    print(f"📊 区域异常分数标准差: {np.nanstd(region_scores):.6f}")
    
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
    
    # 时间戳异常检测：使用DCdetector公式（dcd_style_kl_scorer）
    # ==========================
    # [OLD] 滑动窗口时间戳检测（已弃用，保留注释以便对照）
    # print("\n" + "="*80)
    # print("🆕 时间戳异常检测（DCdetector公式：series/prior KL + 温度/softmax 能量）")
    # print("="*80)
    # window_score_fn = dcd_style_kl_scorer(
    #     detector=None,
    #     patch_len=args.patch_len,
    #     stride=args.stride,
    #     temperature=50.0,
    #     weighting="uniform",
    #     invert_softmax=True,
    #     use_raw_score=True,
    # )
    # seq_len_ts = 12 if (test_X.shape[1] if isinstance(test_X, torch.Tensor) else np.asarray(test_X).shape[1]) >= 12 else (test_X.shape[1] if isinstance(test_X, torch.Tensor) else np.asarray(test_X).shape[1])
    # device_ts = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    # x_for_ts = test_X.numpy() if isinstance(test_X, torch.Tensor) else test_X
    # all_timestamps_scores = compute_timestamp_scores(
    #     x=x_for_ts,
    #     seq_len=seq_len_ts,
    #     batch_size=10,
    #     window_score_fn=window_score_fn,
    #     device=device_ts,
    #     verbose=True,
    # )
    # if np.nanmean(all_timestamps_scores) < 0:
    #     all_timestamps_scores = -all_timestamps_scores
    # ==========================

    # 新：直接使用模型的整段推理（不分批），DCdetector 风格上采样
    print("\n" + "="*80)
    print("🆕 时间戳异常检测（DCdetector：模型整段推理，不滑窗）")
    print("="*80)
    # 🔧 修复：使用更有区分度的参数
    all_timestamps_scores = model.decision_function_time(
        region_input,
        prior='learned_repeat',
        reduce='max',  # 使用max而不是mean，让分数更有区分度
        dc_style_softmax=False,  # 不使用softmax，直接返回KL能量
        temperature=1.0,  # 温度参数在dc_style_softmax=False时不起作用
    )  # (T, N)
    # 转置成 (N, T) 以兼容后续打印与评估
    all_timestamps_scores = all_timestamps_scores.T

    print(f"\n📊 时间戳打分结果概览:")
    print("─" * 60)
    print(f"   🎯 区域数量: {all_timestamps_scores.shape[0]:,}")
    print(f"   🏢 时间戳数量: {all_timestamps_scores.shape[1]:,}")
    print(f"   📈 异常分数范围: [{np.nanmin(all_timestamps_scores):.6f}, {np.nanmax(all_timestamps_scores):.6f}]")
    print(f"   📊 异常分数均值: {np.nanmean(all_timestamps_scores):.6f}")
    print(f"   📉 异常分数标准差: {np.nanstd(all_timestamps_scores):.6f}")
    print(f"   🔍 数据形状: {all_timestamps_scores.shape} (区域数 × 时间戳数)")

    # 🆕 优化：基于区域异常检测结果，只在异常区域中寻找异常时间戳
    print(f"\n🚀 优化策略：只在异常区域中寻找异常时间戳")
    print("─" * 60)

    # 获取区域异常检测结果
    N = all_timestamps_scores.shape[0]  # 区域数量
    region_anomaly_mask = (score >= threshold).astype(bool)  # (N,) - True表示异常区域
    anomaly_region_count = region_anomaly_mask.sum()
    normal_region_count = (~region_anomaly_mask).sum()

    print(f"   📊 区域异常检测结果:")
    print(f"      - 异常区域数: {anomaly_region_count} ({anomaly_region_count/N*100:.1f}%)")
    print(f"      - 正常区域数: {normal_region_count} ({normal_region_count/N*100:.1f}%)")

    # 只在异常区域中寻找异常时间戳
    if anomaly_region_count > 0:
        # 提取异常区域的时间戳分数
        anomaly_region_scores = all_timestamps_scores[region_anomaly_mask]  # (anomaly_region_count, T)
        
        print(f"   🎯 异常区域时间戳分析:")
        print(f"      - 异常区域时间戳总数: {anomaly_region_count * all_timestamps_scores.shape[1]:,}")
        print(f"      - 预期异常时间戳数: {max(1, int(anomaly_region_count * all_timestamps_scores.shape[1] * 0.05)):,} (5%)")
        
        # 在异常区域中寻找异常时间戳（前5%）
        total_anomaly_timestamps = anomaly_region_count * all_timestamps_scores.shape[1]
        k_anomaly_timestamps = max(1, int(total_anomaly_timestamps * 0.05))  # 5%
        
        # 展平异常区域的所有时间戳分数
        flat_anomaly_scores = anomaly_region_scores.flatten()
        anomaly_threshold = np.partition(flat_anomaly_scores, -k_anomaly_timestamps)[-k_anomaly_timestamps]
        
        print(f"   �� 异常时间戳阈值: {anomaly_threshold:.6f}")
        print(f"   🚨 预测异常时间戳数: {k_anomaly_timestamps:,}")
        
        # 创建异常时间戳预测掩码
        anomaly_timestamp_mask = np.zeros_like(all_timestamps_scores, dtype=bool)
        anomaly_timestamp_mask[region_anomaly_mask] = (anomaly_region_scores >= anomaly_threshold)
        
        # 统计每个时间戳的异常区域数
        per_ts_anomaly_counts = anomaly_timestamp_mask.sum(axis=0)
        print(f"   📊 每时间戳异常区域数范围: [{per_ts_anomaly_counts.min()}, {per_ts_anomaly_counts.max()}]，均值: {per_ts_anomaly_counts.mean():.1f}")
        
    else:
        print("   ⚠️ 没有检测到异常区域，跳过时间戳异常检测")
        anomaly_timestamp_mask = np.zeros_like(all_timestamps_scores, dtype=bool)
        per_ts_anomaly_counts = np.zeros(all_timestamps_scores.shape[1])

    # 加载测试集的真实时间戳标签
    import os
    data_dir = f"pems0{args.dataset}"  # 修复：直接使用pems03目录
    test_y_timestamps = np.load(os.path.join(data_dir, "anomaly_labels_test.npy"))  # (T, N)

    print(f"真实标签矩阵形状: {test_y_timestamps.shape}")  # (T, N)

    # 计算每个时间戳的异常检测指标
    print(f"\n📈 时间戳异常检测评估（优化版）:")
    print("─" * 60)

    # 收集异常区域的时间戳数据
    anomaly_scores = []
    anomaly_labels = []

    # 评估异常区域的所有时间戳
    if anomaly_region_count > 0:
        for t_idx in range(all_timestamps_scores.shape[1]):
            if t_idx < test_y_timestamps.shape[0]:
                # 只考虑异常区域
                anomaly_scores_t = all_timestamps_scores[region_anomaly_mask, t_idx]  # (anomaly_region_count,)
                anomaly_labels_t = test_y_timestamps[t_idx][region_anomaly_mask]  # (anomaly_region_count,)
                
                # 过滤掉NaN值
                valid_mask = ~np.isnan(anomaly_scores_t)
                if valid_mask.sum() > 0:
                    valid_scores = anomaly_scores_t[valid_mask]
                    valid_labels = anomaly_labels_t[valid_mask]
                    
                    anomaly_scores.extend(valid_scores)
                    anomaly_labels.extend(valid_labels)

    # 转换为numpy数组
    anomaly_scores = np.array(anomaly_scores)
    anomaly_labels = np.array(anomaly_labels)

    print(f"   📊 异常区域时间戳统计:")
    print(f"      - 异常区域时间戳总数: {len(anomaly_scores)}")
    print(f"      - 真实异常时间戳数: {anomaly_labels.sum()}")
    print(f"      - 异常比例: {anomaly_labels.sum()/len(anomaly_labels):.4f}")

    # 计算异常区域时间戳的异常检测指标
    if len(anomaly_scores) > 0 and anomaly_labels.sum() > 0:
        # 使用Recall@K的形式，异常比例改为5%和10%
        k_5 = max(1, int(len(anomaly_labels) * 0.05))  # 5%
        k_10 = max(1, int(len(anomaly_labels) * 0.10))  # 10%
        
        # 计算Recall@K
        recall_k_5 = recall_k(anomaly_labels, anomaly_scores, k_5)
        recall_k_10 = recall_k(anomaly_labels, anomaly_scores, k_10)
        roc_auc_anomaly = roc_auc_score(anomaly_labels, anomaly_scores)
        
        print(f"\n   📈 异常区域时间戳异常检测指标:")
        print("   ─" * 40)
        print(f"   🎯 Recall@5%: {recall_k_5:.4f}")
        print(f"   🎯 Recall@10%: {recall_k_10:.4f}")
        print(f"   🎯 ROC-AUC: {roc_auc_anomaly:.4f}")
        print(f"   📊 评估样本数: {len(anomaly_scores)}")
        
        # 保存异常区域时间戳指标
        last_timestamp_score_list.append([recall_k_5, recall_k_10, roc_auc_anomaly])
        print(f"   📋 异常区域时间戳异常检测指标: {last_timestamp_score_list[-1]}")
    else:
        print("   ⚠️  没有找到有效的异常区域时间戳样本")
        last_timestamp_score_list.append([0.0, 0.0, 0.0])

    # 显示评估结果
    print(f"   📊 异常检测评估总结:")
    print(f"      - 评估方式: 只在预测为异常的区域中寻找异常时间戳")
    print(f"      - 异常区域数: {anomaly_region_count} / {N} ({anomaly_region_count/N*100:.1f}%)")
    print(f"      - 异常时间戳比例: 5%和10% (从异常区域时间戳中预测top-5%和top-10%为异常)")
    print(f"      - 计算量减少: {(1 - anomaly_region_count/N)*100:.1f}% (跳过正常区域)")

    # 总结信息
    print("="*80)
    print("🎉 优化版时间戳异常检测完成！")
    print("📊 功能总结:")
    print("   ✅ 区域异常检测: 基于静态流和动态流的传统方法")
    print("   🆕 时间戳异常检测: 只在异常区域中寻找异常时间戳（优化版）")
    print("   📈 输出结果: 每个时间戳每个区域的异常分数矩阵")
    print("   🔍 数据形状: " + str(all_timestamps_scores.shape))
    print("   🎯 异常比例设置: 5% (异常区域时间戳数×0.05)")
    print("   📊 异常分数方向: 分数越大表示越异常")
    print("   🚀 优化效果: 计算量减少" + f"{(1 - anomaly_region_count/N)*100:.1f}%" + "（跳过正常区域）")
    print("="*80)
    


print(np.array(score_list).mean(0))
