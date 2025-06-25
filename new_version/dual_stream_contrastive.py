import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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