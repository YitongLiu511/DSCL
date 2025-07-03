import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

def kl_loss(p, q):
    """
    计算KL散度损失，与DSCL-master完全一致
    Args:
        p: 第一个分布
        q: 第二个分布
    Returns:
        KL散度损失
    """
    res = p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))
    # 根据DSCL-master的实现，保持维度信息
    return res

def sym_kl_loss(p, q):
    """
    计算对称KL散度损失，与DSCL-master完全一致
    Args:
        p: 第一个分布
        q: 第二个分布
    Returns:
        对称KL散度损失
    """
    return (kl_loss(p, q) + kl_loss(q, p)) / 2

class DualStreamContrastiveLoss(nn.Module):
    """
    双流对比损失模块，与DSCL-master完全一致
    """
    def __init__(self, use_diff_const=False):
        super(DualStreamContrastiveLoss, self).__init__()
        self.use_diff_const = use_diff_const
        
    def forward(self, score_dy, score_st):
        """
        计算双流对比损失，与DSCL-master完全一致
        Args:
            score_dy: 动态流注意力分数 [B, n_heads, N, N] 或 [N, N]
            score_st: 静态流注意力分数 [N, N]
        Returns:
            discrepancy: 对称KL散度损失
        """
        if self.use_diff_const:
            # 与DSCL-master中的diff_const=True逻辑一致
            discrepancy = sym_kl_loss(score_dy, score_st.detach()) - sym_kl_loss(score_dy.detach(), score_st)
        else:
            # 与DSCL-master中的diff_const=False逻辑一致
            discrepancy = sym_kl_loss(score_dy, score_st)
        
        return discrepancy

def compute_anomaly_score(score_dy, score_st):
    """
    计算异常分数，与DSCL-master完全一致
    Args:
        score_dy: 动态流注意力分数
        score_st: 静态流注意力分数
    Returns:
        anomaly_score: 异常分数
    """
    # 直接使用对称KL散度作为异常分数，与DSCL-master一致
    discrepancy = sym_kl_loss(score_dy, score_st)
    return discrepancy

def compute_dual_stream_loss(score_dy, score_st, use_recon=True, use_const=True, diff_const=False, 
                           recon_loss=None, loss_weight=None):
    """
    计算完整的双流对比损失，与DSCL-master完全一致
    Args:
        score_dy: 动态流注意力分数
        score_st: 静态流注意力分数
        use_recon: 是否使用重构损失
        use_const: 是否使用对比损失
        diff_const: 是否使用差分对比损失
        recon_loss: 重构损失
        loss_weight: 损失权重
    Returns:
        total_loss: 总损失
        loss1: 重构损失
        loss2: 对比损失
    """
    loss1 = None
    loss2 = None
    
    if use_recon and recon_loss is not None:
        loss1 = loss_weight[0] * recon_loss.mean()
    
    if use_const:
        if diff_const:
            discrepancy = sym_kl_loss(score_dy, score_st.detach()) - sym_kl_loss(score_dy.detach(), score_st)
        else:
            discrepancy = sym_kl_loss(score_dy, score_st)
        loss2 = loss_weight[1] * discrepancy.mean()
    
    # 计算总损失
    if use_recon and not use_const:
        total_loss = loss1
    elif use_const and not use_recon:
        total_loss = loss2
    else:
        total_loss = loss1 + loss2
    
    return total_loss, loss1, loss2

def update_loss_weight(loss1, loss2, loss_weight):
    """
    更新损失权重，与DSCL-master完全一致
    Args:
        loss1: 重构损失
        loss2: 对比损失
        loss_weight: 当前损失权重
    Returns:
        new_loss_weight: 更新后的损失权重
    """
    exp = torch.tensor([-loss1, -loss2]).exp()
    new_weight = loss_weight * exp
    new_loss_weight = new_weight / torch.sum(new_weight)
    return new_loss_weight 

class HardNegativeMiner:
    """
    困难负样本筛选与真假负样本判别模块
    """
    def __init__(self, dist_mat, poi_sim_mat, top_k=10, true_neg_thresh=0.7):
        self.dist_mat = dist_mat
        self.poi_sim_mat = poi_sim_mat
        self.top_k = top_k
        self.true_neg_thresh = true_neg_thresh

    def select_hard_negatives(self, anchor_idx):
        # 功能近空间远
        func_close = np.argsort(-self.poi_sim_mat[anchor_idx])[:self.top_k*2]
        space_far = np.argsort(self.dist_mat[anchor_idx])[::-1][:self.top_k*2]
        hard_neg1 = list(set(func_close) & set(space_far))
        # 空间近功能远
        space_close = np.argsort(self.dist_mat[anchor_idx])[:self.top_k*2]
        func_far = np.argsort(self.poi_sim_mat[anchor_idx])[:self.top_k*2]
        hard_neg2 = list(set(space_close) & set(func_far))
        return list(set(hard_neg1 + hard_neg2))

    def estimate_true_negative_probs(self, sim_scores):
        from sklearn.mixture import GaussianMixture
        sim_scores = np.array(sim_scores).reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, random_state=0).fit(sim_scores)
        probs = gmm.predict_proba(sim_scores)
        true_neg_comp = np.argmin(gmm.means_)
        true_neg_probs = probs[:, true_neg_comp]
        return true_neg_probs

    def mine(self, features):
        """
        features: [N, ...] numpy array, 每个节点的特征
        返回: 每个anchor的困难真负样本索引列表
        """
        N = features.shape[0]
        all_hard_negs = []
        all_sim_scores = []
        anchor_neg_pairs = []
        for i in range(N):
            hard_negs = self.select_hard_negatives(i)
            anchor_feat = features[i].flatten()
            for j in hard_negs:
                neg_feat = features[j].flatten()
                sim = np.dot(anchor_feat, neg_feat) / (np.linalg.norm(anchor_feat) * np.linalg.norm(neg_feat) + 1e-8)
                all_sim_scores.append(sim)
                anchor_neg_pairs.append((i, j))
        true_neg_probs = self.estimate_true_negative_probs(all_sim_scores)
        # 选取高概率真负样本
        selected_pairs = [pair for pair, prob in zip(anchor_neg_pairs, true_neg_probs) if prob > self.true_neg_thresh]
        # 按anchor分组
        from collections import defaultdict
        anchor2negs = defaultdict(list)
        for i, j in selected_pairs:
            anchor2negs[i].append(j)
        return anchor2negs

class ClusterContrastiveTrainer:
    """
    聚类+困难负样本+对比损失一体化训练器
    """
    def __init__(self, node_features, dist_mat, poi_sim_mat, n_clusters=10, top_k=10, true_neg_thresh=0.7, temperature=0.2):
        """
        node_features: [N, F] numpy array
        dist_mat, poi_sim_mat: [N, N] numpy array
        """
        self.node_features = node_features
        self.dist_mat = dist_mat
        self.poi_sim_mat = poi_sim_mat
        self.n_clusters = n_clusters
        self.top_k = top_k
        self.true_neg_thresh = true_neg_thresh
        self.temperature = temperature
        self._cluster()
        self.miner = HardNegativeMiner(dist_mat, poi_sim_mat, top_k=top_k, true_neg_thresh=true_neg_thresh)

    def _cluster(self):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(self.node_features)
        self.cluster_labels = kmeans.labels_
        self.cluster_centers = kmeans.cluster_centers_
        # 构建正样本索引
        cluster2nodes = defaultdict(list)
        for idx, label in enumerate(self.cluster_labels):
            cluster2nodes[label].append(idx)
        self.anchor2pos = {}
        for idx, label in enumerate(self.cluster_labels):
            self.anchor2pos[idx] = [i for i in cluster2nodes[label] if i != idx]

    def mine_hard_negatives(self):
        # 用当前node_features动态挖掘困难负样本
        self.anchor2negs = self.miner.mine(self.node_features)
        return self.anchor2negs

    def contrastive_loss(self, device='cpu'):
        # node_features转torch
        node_features = torch.tensor(self.node_features, dtype=torch.float32, device=device)
        loss = 0
        N = node_features.shape[0]
        valid_count = 0
        for anchor in range(N):
            anchor_feat = node_features[anchor]
            pos_indices = self.anchor2pos[anchor]
            neg_indices = self.anchor2negs.get(anchor, [])
            if not pos_indices or not neg_indices:
                continue
            pos_feats = node_features[pos_indices]
            pos_sim = F.cosine_similarity(anchor_feat.unsqueeze(0), pos_feats).mean()
            neg_feats = node_features[neg_indices]
            neg_sim = F.cosine_similarity(anchor_feat.unsqueeze(0), neg_feats)
            logits = torch.cat([pos_sim.unsqueeze(0), neg_sim])  # [1+K]
            logits = logits.unsqueeze(0)  # [1, 1+K]
            labels = torch.zeros(1, dtype=torch.long, device=logits.device)  # [1]
            logits = logits / self.temperature
            loss += F.cross_entropy(logits, labels)
            valid_count += 1
        return loss / (valid_count if valid_count > 0 else 1)

    def step(self, device='cpu'):
        """
        一步式调用：动态挖掘困难负样本并计算对比损失
        """
        self.mine_hard_negatives()
        return self.contrastive_loss(device=device)