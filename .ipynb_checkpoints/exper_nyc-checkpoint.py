import torch
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from STAnomalyFormer.interface.estimator import STPatch_MGCNDetector
from data.load_nyc import load_dataset
from model.solver import Solver

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--n_day', type=int, default=14)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--patch_len', type=int, default=12)
    parser.add_argument('--stride', type=int, default=6)
    parser.add_argument('--n_gcn', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    train_data, val_data, test_data, dist_mat, y = load_dataset(args)
    
    # 训练MTFA模型
    mtfa_config = {
        'data_path': 'data/nyc',
        'dataset': 'nyc',
        'win_size': args.n_day * 24,  # 使用天数作为窗口大小
        'seq_size': args.patch_len,
        'input_c': 1,
        'output_c': 1,
        'd_model': args.d_model,
        'e_layers': 3,
        'fr': 0.4,
        'tr': 0.5,
        'batch_size': 32,
        'num_epochs': 100,
        'lr': args.lr,
        'gpu': 0 if args.cuda else -1,
        'model_save_path': 'checkpoints',
        'anormly_ratio': 0.1
    }
    
    mtfa_solver = Solver(mtfa_config)
    mtfa_solver.train()
    mtfa_accuracy, mtfa_precision, mtfa_recall, mtfa_f_score = mtfa_solver.test()
    
    # 训练STPatch_MGCNDetector
    score_list = []
    for i in range(args.repeat):
        detector = STPatch_MGCNDetector(
            seq_len=args.n_day * 24,
            patch_len=args.patch_len,
            stride=args.stride,
            d_in=1,
            d_model=args.d_model,
            n_heads=args.n_heads,
            dist_mats=dist_mat,
            n_gcn=args.n_gcn,
        )
        
        detector.fit(train_data, dist_mat, (val_data, None))
        score = detector.decision_function(test_data)
        score_list.append(score)
    
    # 计算平均分数
    mean_score = np.mean(score_list, axis=0)
    
    # 保存结果
    with open('test.txt', 'a') as f:
        f.write(f'MTFA Results:\n')
        f.write(f'Accuracy: {mtfa_accuracy:.4f}, Precision: {mtfa_precision:.4f}, Recall: {mtfa_recall:.4f}, F-score: {mtfa_f_score:.4f}\n')
        f.write(f'STPatch_MGCNDetector Results:\n')
        f.write(f'Mean Score: {np.mean(mean_score):.4f}\n\n')

if __name__ == '__main__':
    main()
