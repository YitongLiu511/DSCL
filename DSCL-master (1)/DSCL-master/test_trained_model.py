 

import os
import sys
import numpy as np
import torch
import argparse
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from STAnomalyFormer.interface.estimator import STPatch_MGCNDetector
from STAnomalyFormer.model.anomaly import STPatch_MGCNFormer
from STAnomalyFormer.model.patch import Patch
from STAnomalyFormer.model.module import SoftClusterLayer

def load_data(dataset_id):
    """加载数据"""
    print(f"📊 加载数据集 pems0{dataset_id}...")
    
    # 使用与训练相同的数据加载方式
    from data.load_pems import load_dataset
    import argparse
    
    # 创建args对象
    class Args:
        def __init__(self, dataset_id):
            self.dataset = dataset_id
    
    args = Args(dataset_id)
    X, X_clean, val_X, test_X, test_X_clean, mats, y = load_dataset(args)
    adj, distance, connectivity = mats
    
    # 转换为numpy数组
    X = X.numpy()
    adj = adj.numpy()
    if distance is not None:
        distance = distance.numpy()
    if connectivity is not None:
        connectivity = connectivity.numpy()
    y = y.numpy()
    
    # 加载测试集时间戳标签
    data_dir = f"pems0{dataset_id}"
    try:
        test_y_timestamps = np.load(os.path.join(data_dir, "anomaly_labels_test.npy"))
        print(f"✅ 成功加载测试集时间戳标签，形状: {test_y_timestamps.shape}")
    except:
        test_y_timestamps = None
        print("⚠️  未找到测试集时间戳标签文件")
    
    print(f"📈 数据形状: X={X.shape}, adj={adj.shape}")
    return X, adj, distance, connectivity, y, test_y_timestamps

def create_model(args, device):
    """创建模型"""
    print(f"🔧 创建模型...")
    
    # 模型参数
    model_args = {
        'seq_len': 12,
        'patch_len': args.patch_len,
        'stride': args.stride,
        'd_in': 1,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_gcn': args.n_gcn,
        'device': device,
        'epoch': 1,  # 测试时不需要训练
        'lr': args.lr,
        'early_stopping': False,
        'contamination': 0.1,
        'use_recon': True,
        'use_const': True,
        'diff_const': args.diff_const,
        'static_only': False,
        'dynamic_only': False,
        'verbose': False,
        'cluster_weight': 0.1,
        'args': args,
    }
    
    # 创建检测器
    detector = STPatch_MGCNDetector(**model_args)
    
    return detector

def load_trained_model(detector, model_path, X, mats):
    """加载训练好的模型"""
    print(f"📥 加载训练好的模型: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    try:
        # 先初始化模型（调用fit方法的一部分来创建self.model）
        print("🔧 初始化模型结构...")
        detector.fit(X, mats, evaluate=None, x_clean=None)
        
        # 加载模型状态
        checkpoint = torch.load(model_path, map_location=detector.device)
        
        # 检查checkpoint的结构
        print(f"🔍 Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")
        
        # 加载模型参数
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                detector.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                detector.model.load_state_dict(checkpoint['model'])
            else:
                # 假设整个checkpoint就是模型状态
                detector.model.load_state_dict(checkpoint)
            
            # 加载其他组件
            if 'last_linear_state_dict' in checkpoint:
                detector.last_linear.load_state_dict(checkpoint['last_linear_state_dict'])
            if 'time_proj_state_dict' in checkpoint:
                detector.time_proj.load_state_dict(checkpoint['time_proj_state_dict'])
            if 'loss_weight' in checkpoint:
                detector.loss_weight = checkpoint['loss_weight']
        else:
            # 如果checkpoint不是dict，直接加载
            detector.model.load_state_dict(checkpoint)
        
        detector.model.eval()
        print("✅ 模型加载成功！")
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def evaluate_regional_anomaly(detector, X, y_true=None):
    """评估区域异常检测"""
    print(f"\n{'='*60}")
    print("🏢 区域异常检测评估")
    print(f"{'='*60}")
    
    # 进行区域异常检测
    regional_scores = detector.decision_function(X, verbose_output=True)
    
    if y_true is not None:
        # 计算评估指标
        auc = roc_auc_score(y_true, regional_scores)
        
        # 使用20%作为阈值
        threshold = np.sort(regional_scores)[-int(len(regional_scores) * 0.2)]
        y_pred = (regional_scores >= threshold).astype(int)
        
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"\n📊 区域异常检测评估结果:")
        print(f"   🎯 AUC: {auc:.4f}")
        print(f"   📈 Recall: {recall:.4f}")
        print(f"   📉 Precision: {precision:.4f}")
        print(f"   📊 F1-Score: {f1:.4f}")
        print(f"   🚨 预测异常区域: {y_pred.sum()} / {len(y_pred)}")
        print(f"   ✅ 真实异常区域: {y_true.sum()} / {len(y_true)}")
    
    return regional_scores

def evaluate_timestamp_anomaly(detector, X, y_timestamps=None):
    """评估时间戳异常检测"""
    print(f"\n{'='*60}")
    print("⏰ 时间戳异常检测评估")
    print(f"{'='*60}")
    
    # 进行时间戳异常检测
    timestamp_scores = detector.get_last_timestamp_score(X)
    
    # 确保异常分数越大表示越异常
    if np.nanmean(timestamp_scores) < 0.1:
        print("🔄 调整异常分数方向：分数越大表示越异常")
        timestamp_scores = -timestamp_scores  # 取负值，使分数越大越异常
    
    if y_timestamps is not None:
        print(f"\n📊 时间戳异常检测评估结果:")
        
        # 计算每个时间戳的评估指标
        auc_list = []
        recall_5_list = []
        recall_10_list = []
        
        for t_idx in range(timestamp_scores.shape[0]):
            scores_t = timestamp_scores[t_idx]  # (N,)
            
            # 对应的时间戳标签
            seq_len = getattr(detector, 'seq_len', 12)
            actual_timestamp = t_idx + seq_len - 1
            
            if actual_timestamp < y_timestamps.shape[0]:
                labels_t = y_timestamps[actual_timestamp]  # (N,)
                
                if labels_t.sum() > 0:  # 如果有异常
                    # 计算AUC
                    auc = roc_auc_score(labels_t, scores_t)
                    auc_list.append(auc)
                    
                    # 计算Recall@K，异常比例改为5%
                    k_5 = max(1, int(len(labels_t) * 0.05))  # 5%
                    k_10 = max(1, int(len(labels_t) * 0.10))  # 10%
                    
                    # 获取top-k预测
                    top_k_5_indices = np.argsort(scores_t)[-k_5:]
                    top_k_10_indices = np.argsort(scores_t)[-k_10:]
                    
                    # 计算Recall@K
                    recall_5 = labels_t[top_k_5_indices].sum() / labels_t.sum()
                    recall_10 = labels_t[top_k_10_indices].sum() / labels_t.sum()
                    
                    recall_5_list.append(recall_5)
                    recall_10_list.append(recall_10)
        
        if auc_list:
            avg_auc = np.mean(auc_list)
            avg_recall_5 = np.mean(recall_5_list)
            avg_recall_10 = np.mean(recall_10_list)
            
            print(f"   🎯 平均AUC: {avg_auc:.4f}")
            print(f"   📈 平均Recall@5%: {avg_recall_5:.4f} (异常比例5%)")
            print(f"   📈 平均Recall@10%: {avg_recall_10:.4f}")
            print(f"   📊 评估时间戳数: {len(auc_list)}")
        else:
            print("   ⚠️  没有找到异常时间戳")
    
    return timestamp_scores

def main():
    parser = argparse.ArgumentParser(description='测试训练好的异常检测模型')
    parser.add_argument('--dataset', type=int, default=3, help='数据集ID')
    parser.add_argument('--model_path', type=str, default='checkpoint.pt', help='模型文件路径')
    parser.add_argument('--cuda', action='store_true', help='使用GPU')
    parser.add_argument('--d_model', type=int, default=32, help='模型维度')
    parser.add_argument('--n_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--patch_len', type=int, default=4, help='patch长度')
    parser.add_argument('--stride', type=int, default=6, help='stride')
    parser.add_argument('--n_gcn', type=int, default=3, help='GCN层数')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--diff_const', action='store_true', help='使用差分一致性损失')
    
    args = parser.parse_args()
    
    # 设置设备
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    print(f"🔧 使用设备: {device}")
    
    # 加载数据
    X, adj, distance, connectivity, y, test_y_timestamps = load_data(args.dataset)
    
    # 创建模型
    detector = create_model(args, device)
    
    # 准备数据
    mats = np.array([adj, distance, connectivity])
    
    # 加载训练好的模型
    if not load_trained_model(detector, args.model_path, X, mats):
        print("❌ 无法加载模型，退出")
        return
    
    # 评估区域异常检测
    regional_scores = evaluate_regional_anomaly(detector, X, y)
    
    # 评估时间戳异常检测
    timestamp_scores = evaluate_timestamp_anomaly(detector, X, test_y_timestamps)
    
    print(f"\n{'='*60}")
    print("✅ 测试完成！")
    print(f"{'='*60}")
    
    # 保存结果
    np.save('regional_scores.npy', regional_scores)
    np.save('timestamp_scores.npy', timestamp_scores)
    print("💾 结果已保存到 regional_scores.npy 和 timestamp_scores.npy")

if __name__ == "__main__":
    main()