import numpy as np
from STAnomalyFormer.interface.estimator import STPatch_MGCNDetector, TemporalTSFMDetector
from STAnomalyFormer.interface.utils import recall_k
from data.load_nyc import load_dataset
import torch
from sklearn.metrics import roc_auc_score

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--n_day', type=int, default=14)
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--poi_normalize', action='store_true')

parser.add_argument('--delay', action='store_true')
parser.add_argument('--lag', type=int, default=6)
parser.add_argument('--vol', action='store_true')
parser.add_argument('--threshold', default=0.5, type=float)
parser.add_argument('--attr', action='store_true')
parser.add_argument('--k', default=None, type=int)

parser.add_argument('--mask_ratio', default=0.4, type=float)
parser.add_argument('--patch_len', default=12, type=int)
parser.add_argument('--stride', default=6, type=int)
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--n_heads', default=16, type=int)
parser.add_argument('--t_half', action='store_true')
parser.add_argument('--s_half', action='store_true')
parser.add_argument('--n_gcn', default=3, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--repeat', default=10, type=int)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--no_recon', action='store_true')
parser.add_argument('--no_const', action='store_true')
parser.add_argument('--diff_const', action='store_true')
parser.add_argument('--dynamic_only', action='store_true')
parser.add_argument('--static_only', action='store_true')
parser.add_argument('--early_stopping', action='store_true')

parser.add_argument('--cuda', action='store_true')

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
# torch.autograd.set_detect_anomaly(True)

score_list = []
X, val_X, test_X, (adj, dist, poi_sim), y = load_dataset(args)

# 打印原始shape
print("原始数据shape:", X.shape)

# 从 (14, 144, 263) 变成 (263, 144, 14)
X = X.transpose(2, 1, 0)        # 让区域数在第一维，时间槽在第二维，天数在第三维
val_X = val_X.transpose(2, 1, 0)
test_X = test_X.transpose(2, 1, 0)

# 打印调整后的shape
print("调整后数据shape:", X.shape)

for t in range(args.repeat):
    print("{}-th experiment:".format(t + 1))

    model = STPatch_MGCNDetector(
        seq_len=X.shape[1],
        patch_len=args.patch_len,
        stride=args.stride,
        d_in=X.shape[2],
        d_model=args.d_model,
        n_heads=args.n_heads,
        temporal_half=args.t_half,
        spatial_half=args.s_half,
        n_gcn=args.n_gcn,
        device='cuda' if args.cuda else 'cpu',
        epoch=args.epochs,
        lr=args.lr,
        verbose=args.verbose,
        use_recon=(not args.no_recon),
        use_const=(not args.no_const),
        diff_const=args.diff_const,
        static_only=args.static_only,
        dynamic_only=args.dynamic_only,
    ).fit(X, torch.stack([torch.tensor(1 - adj), torch.tensor(dist), torch.tensor(poi_sim)]), (val_X, y))
    score = model.decision_function(test_X)
    threshold = np.sort(score)[-np.ceil(len(score) // 5).astype(int)]
    pred = np.zeros_like(score)
    pred[score >= threshold] = 1
    score_list.append([
        recall_k(y, pred,
                 np.ceil(len(y) // 10).astype(int)),
        recall_k(y, pred,
                 np.ceil(len(y) // 5).astype(int)),
        roc_auc_score(y, score)
    ])
    print(score_list[-1])

with open('test.txt', 'a+') as f:
    f.write(str(list(np.array(score_list).mean(0))) + '\n')
