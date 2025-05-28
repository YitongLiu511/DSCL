import numpy as np
from STAnomalyFormer.interface.utils import recall_k
from data.load_nyc import load_dataset
import torch
from sklearn.metrics import roc_auc_score
from data.anomaly_injection import AnomalyInjector
from STAnomalyFormer.model.patch import STPatchMaskFormer
import torch.nn as nn

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
parser.add_argument('--epochs', default=1, type=int)  # 训练轮数改为1
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
parser.add_argument('--time_threshold', type=float, default=0.7, help='时间异常注入的阈值μ')
parser.add_argument('--k_neighbors', type=int, default=None, help='空间异常注入时采样的邻居节点数')
parser.add_argument('--time_window', type=int, default=120, help='时间异常检测窗口（分钟）')

parser.add_argument('--cuda', action='store_true', default=False)  # 默认使用CPU

# 掩码相关参数
parser.add_argument('--time_ratio', type=float, default=0.5, help='时间掩码比例')
parser.add_argument('--freq_ratio', type=float, default=0.4, help='频域掩码比例')

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
print(f"- 异常区域数: {np.sum(y_reshaped[:, 0, 0, 0] == 1)} (预期: {expected_region_anomalies})")
print(f"- 异常区域比例: {np.sum(y_reshaped[:, 0, 0, 0] == 1) / 263 * 100:.2f}%")
print(f"时间异常标签:")
print(f"- 异常时间槽数: {len(anomaly_time_indices)} (预期: {n_anomaly_times})")
print(f"- 异常时间槽比例: {len(anomaly_time_indices) / 144 * 100:.2f}%")
print(f"标签形状: {y_reshaped.shape}")
print(f"总异常样本数: {np.sum(y_reshaped == 1)}")
print(f"总正常样本数: {np.sum(y_reshaped == 0)}")
print(f"总体异常比例: {np.mean(y_reshaped == 1) * 100:.2f}%")

# 确保输入数据和标签的维度匹配
print("\n维度检查:")
print(f"输入数据X形状: {X.shape}")
print(f"标签y_reshaped形状: {y_reshaped.shape}")
print(f"验证集形状: {val_X.shape}")
print(f"测试集形状: {test_X.shape}")

# 计算分片数量
n_patches = (test_X.shape[1] - args.patch_len) // args.stride + 1
print(f"\n分片信息:")
print(f"时间序列长度: {test_X.shape[1]}")
print(f"Patch大小: {args.patch_len}")
print(f"步长: {args.stride}")
print(f"分片数量: {n_patches}")

# 计算时间分片大小
time_interval = 10  # 每个时间点间隔10分钟
time_patch_len = args.time_window // time_interval  # 根据时间窗口计算分片大小
print(f"\n时间异常检测设置:")
print(f"时间窗口: {args.time_window}分钟")
print(f"时间点间隔: {time_interval}分钟")
print(f"分片大小: {time_patch_len}个时间点")

# 创建异常注入器
injector = AnomalyInjector(
    n_nodes=X.shape[0],  # 263个节点
    n_timesteps=n_patches,  # 使用实际的分片数量
    time_anomaly_ratio=args.anormly_ratio * 0.5,  # 降低时间异常比例
    space_anomaly_ratio=args.anormly_ratio * 0.5,  # 降低空间异常比例
    time_threshold=args.time_threshold,
    k_neighbors=args.k_neighbors,
    seed=args.seed
)

for t in range(args.repeat):
    print("{}-th experiment:".format(t + 1))

    # 确保test_X是NumPy数组
    if isinstance(test_X, torch.Tensor):
        test_X = test_X.cpu().numpy()
    
    # 对测试数据进行时间分片
    test_X_time_patched = np.zeros((test_X.shape[0], n_patches, time_patch_len, test_X.shape[2]))
    for i in range(n_patches):
        start_idx = i * args.stride
        test_X_time_patched[:, i] = test_X[:, start_idx:start_idx + time_patch_len]

    # 区域异常处理（使用因果卷积后的特征）
    test_X_region_patched = np.zeros((test_X.shape[0], n_patches, args.patch_len, test_X.shape[2]))
    for i in range(n_patches):
        start_idx = i * args.stride
        # 这里应该使用经过因果卷积处理后的特征
        test_X_region_patched[:, i] = test_X[:, start_idx:start_idx + args.patch_len]

    # 注入异常并获取异常掩码
    test_X_patched_injected, anomaly_mask = injector.inject_anomalies(
        test_X_region_patched,
        inject_time=True,
        inject_space=True,
        return_mask=True
    )

    # 将分片后的数据还原回原始时间序列
    test_X_injected = np.zeros_like(test_X)
    for i in range(n_patches):
        start_idx = i * args.stride
        test_X_injected[:, start_idx:start_idx + args.patch_len] = test_X_patched_injected[:, i]

    # 创建标签
    y_reshaped = np.zeros((263, 144, 14, 2), dtype=bool)
    
    # 将异常掩码还原到原始时间序列
    anomaly_mask_full = np.zeros((263, 144, 14), dtype=bool)
    for i in range(n_patches):
        start_idx = i * args.stride
        # 打印每一步的维度信息
        print(f"\n处理第 {i} 个patch:")
        print(f"1. anomaly_mask[:, i] shape: {anomaly_mask[:, i].shape}")
        
        # 将每个patch的异常掩码压缩到节点级别
        patch_mask = np.any(anomaly_mask[:, i], axis=(1, 2))  # [263]
        print(f"2. patch_mask shape: {patch_mask.shape}")
        
        # 扩展到正确的维度
        expanded_mask = np.tile(patch_mask[:, np.newaxis], (1, 14))  # [263, 14]
        print(f"3. expanded_mask shape: {expanded_mask.shape}")
        
        # 对每个patch的时间段进行赋值
        for j in range(args.patch_len):
            anomaly_mask_full[:, start_idx + j] = expanded_mask
    
    # 打印最终的维度信息
    print("\n最终维度信息:")
    print(f"原始异常掩码形状: {anomaly_mask.shape}")
    print(f"还原后的掩码形状: {anomaly_mask_full.shape}")
    print(f"目标标签形状: {y_reshaped.shape}")
    
    # 时间异常标签（第一维）
    y_reshaped[:, :, :, 0] = anomaly_mask_full
    # 空间异常标签（第二维）
    y_reshaped[:, :, :, 1] = anomaly_mask_full

    # 打印异常注入统计信息
    print("\n异常注入统计:")
    time_anomaly_nodes = np.sum(np.any(anomaly_mask_full, axis=(1, 2)))
    space_anomaly_nodes = np.sum(np.any(anomaly_mask_full, axis=(0, 2)))
    total_anomalies = np.sum(anomaly_mask_full)
    anomaly_ratio = np.mean(anomaly_mask_full) * 100
    
    print(f"时间异常节点数: {time_anomaly_nodes}")
    print(f"空间异常节点数: {space_anomaly_nodes}")
    print(f"总异常样本数: {total_anomalies}")
    print(f"异常比例: {anomaly_ratio:.2f}%")
    print(f"预期异常比例: {args.anormly_ratio * 100:.2f}%")

    # 确保输入数据和标签的维度匹配
    print("\n维度检查:")
    print(f"原始数据X形状: {X.shape}")
    print(f"注入异常后的测试数据形状: {test_X_injected.shape}")
    print(f"标签y_reshaped形状: {y_reshaped.shape}")
    print(f"验证集形状: {val_X.shape}")
    print(f"测试集形状: {test_X.shape}")

    # 将numpy数组转换为PyTorch张量
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(val_X, torch.Tensor):
        val_X = val_X.cpu().numpy()
    if isinstance(test_X, torch.Tensor):
        test_X = test_X.cpu().numpy()
    if isinstance(test_X_injected, torch.Tensor):
        test_X_injected = test_X_injected.cpu().numpy()
    if isinstance(y_reshaped, torch.Tensor):
        y_reshaped = y_reshaped.cpu().numpy()

    X = torch.FloatTensor(X)
    val_X = torch.FloatTensor(val_X)
    test_X = torch.FloatTensor(test_X)
    test_X_injected = torch.FloatTensor(test_X_injected)
    y_reshaped = torch.FloatTensor(y_reshaped)

    # 初始化模型
    model = STPatchMaskFormer(
        c_in=X.shape[2],
        seq_len=X.shape[1],
        patch_len=args.patch_len,
        stride=args.stride,
        max_seq_len=X.shape[1],
        n_layers=args.n_gcn,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_model,
        shared_embedding=True,
        attn_dropout=0.1,
        dropout=0.1,
        act='gelu',
        mask_ratio=args.mask_ratio,
        time_ratio=args.time_ratio,
        freq_ratio=args.freq_ratio,
        patch_size=args.patch_len,
        poi_sim=torch.FloatTensor(poi_sim),
        dist_mat=torch.FloatTensor(dist),
    )

    # 训练模型
    model = model.to('cuda' if args.cuda else 'cpu')
    X = X.to('cuda' if args.cuda else 'cpu')
    val_X = val_X.to('cuda' if args.cuda else 'cpu')
    test_X = test_X.to('cuda' if args.cuda else 'cpu')
    test_X_injected = test_X_injected.to('cuda' if args.cuda else 'cpu')
    y_reshaped = y_reshaped.to('cuda' if args.cuda else 'cpu')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        output = model(X)
        
        # 计算重建损失
        loss = criterion(output, X)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        if args.verbose and epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

    # 使用注入异常后的测试数据进行评估
    model.eval()
    with torch.no_grad():
        output_dict = model(test_X_injected, return_dict=True)
        
        # 获取异常分数和预测
        anomaly_scores = output_dict['anomaly_scores']  # [N, T, 2]
        region_scores = anomaly_scores[:, :, 0]        # [N, T]
        time_scores = anomaly_scores[:, :, 1]          # [N, T]
        
        print("\n评估阶段维度追踪:")
        print(f"1. anomaly_scores维度: {anomaly_scores.shape}")
        print(f"2. region_scores维度: {region_scores.shape}")
        print(f"3. time_scores维度: {time_scores.shape}")
        
        # 将分数转换为与标签相同的维度
        region_scores_full = np.zeros((263, 144))
        time_scores_full = np.zeros((263, 144))
        
        # 确保分数是2D数组
        region_scores = region_scores.cpu().numpy()  # [263, T]
        time_scores = time_scores.cpu().numpy()      # [263, T]
        
        print(f"4. region_scores转换后维度: {region_scores.shape}")
        print(f"5. time_scores转换后维度: {time_scores.shape}")
        
        # 使用最近邻插值将分数扩展到144个时间步
        for i in range(263):
            try:
                # 获取当前区域的时间序列
                region_series = region_scores[i]  # [T]
                time_series = time_scores[i]      # [T]
                
                # 确保是一维数组
                if len(region_series.shape) > 1:
                    region_series = region_series[:, 0]  # 取第一个通道
                if len(time_series.shape) > 1:
                    time_series = time_series[:, 0]  # 取第一个通道
                
                # 创建插值点
                x_old = np.linspace(0, 143, len(region_series))
                x_new = np.arange(144)
                
                # 执行插值
                region_scores_full[i] = np.interp(x_new, x_old, region_series)
                time_scores_full[i] = np.interp(x_new, x_old, time_series)
                
            except Exception as e:
                print(f"处理第{i}个样本时出错:")
                print(f"region_series形状: {region_series.shape}")
                print(f"time_series形状: {time_series.shape}")
                print(f"x_old形状: {x_old.shape}")
                print(f"x_new形状: {x_new.shape}")
                raise e
        
        print(f"6. region_scores_full维度: {region_scores_full.shape}")
        print(f"7. time_scores_full维度: {time_scores_full.shape}")
        
        # 计算评估指标
        metrics = model.evaluate(test_X_injected, y_reshaped)

    # 保存结果
    results = {
        'region_auc': metrics['region_auc'],
        'time_auc': metrics['time_auc'],
        'region_scores': region_scores_full,
        'time_scores': time_scores_full,
        'anomaly_mask': anomaly_mask_full
    }

    # 保存结果到文件
    np.save(f'results/experiment_{t+1}.npy', results)

    # 打印评估结果
    print(f"\n评估结果:")
    print(f"区域异常AUC: {metrics['region_auc']:.4f}")
    print(f"时间异常AUC: {metrics['time_auc']:.4f}")

    # 更新score_list
    score_list.append([
        metrics['region_auc'],
        metrics['time_auc']
    ])

# 计算平均指标
mean_scores = np.array(score_list).mean(0)
print("\n平均评估指标:")
print(f"区域异常 AUC: {mean_scores[0]:.4f}")
print(f"时间异常 AUC: {mean_scores[1]:.4f}")

with open('test.txt', 'a+') as f:
    if score_list:
        f.write(str(list(mean_scores)) + '\n')
    else:
        f.write("No valid results\n")