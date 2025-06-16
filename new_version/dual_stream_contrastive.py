import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def kl_loss(p, q):
    """
    计算KL散度损失
    Args:
        p: 第一个分布
        q: 第二个分布
    Returns:
        KL散度损失
    """
    res = p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))
    return torch.mean(torch.mean(res, dim=(0, 1)), dim=1)

def sym_kl_loss(p, q):
    """
    计算对称KL散度损失
    Args:
        p: 第一个分布
        q: 第二个分布
    Returns:
        对称KL散度损失
    """
    return (kl_loss(p, q) + kl_loss(q, p)) / 2

class DualStreamContrastiveLoss(nn.Module):
    """
    双流对比损失模块
    """
    def __init__(self, temperature=0.07):
        super(DualStreamContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, dynamic_features, static_features, dynamic_attn=None, static_attn=None):
        """
        计算双流对比损失
        Args:
            dynamic_features: 动态流特征 [batch_size, feature_dim]
            static_features: 静态流特征 [batch_size, feature_dim]
            dynamic_attn: 动态流注意力分数 [batch_size, n_nodes, n_nodes]
            static_attn: 静态流注意力分数 [batch_size, n_nodes, n_nodes]
        Returns:
            total_loss: 总损失
            adv_loss: 对抗损失
            con_loss: 对比损失
        """
        # 计算时间异常分数
        max_flow = torch.max(dynamic_features, dim=1)[0]
        flow_ratio = dynamic_features / (max_flow.unsqueeze(1) + 1e-6)
        time_scores = torch.mean(flow_ratio, dim=-1)
        
        # 计算空间异常分数
        region_flow = torch.mean(dynamic_features, dim=1)
        region_diff = torch.cdist(region_flow, region_flow)
        
        # 计算特征级别的对比损失
        feature_loss = sym_kl_loss(dynamic_features, static_features)
        
        # 如果提供了注意力分数，计算注意力级别的对比损失
        attn_loss = 0
        if dynamic_attn is not None and static_attn is not None:
            # 确保注意力分数维度匹配
            if dynamic_attn.size() != static_attn.size():
                min_size = min(dynamic_attn.size(0), static_attn.size(0))
                dynamic_attn = dynamic_attn[:min_size, :min_size]
                static_attn = static_attn[:min_size, :min_size]
            
            # 计算注意力对比损失
            attn_loss = sym_kl_loss(dynamic_attn, static_attn)
        
        # 计算总损失
        total_loss = feature_loss + attn_loss
        
        # 打印损失信息
        print("\n" + "="*80)
        print("【双流对比损失计算】")
        print(f"特征对比损失: {feature_loss.item():.4f}")
        if dynamic_attn is not None and static_attn is not None:
            print(f"注意力对比损失: {attn_loss.item():.4f}")
        print(f"总损失: {total_loss.item():.4f}")
        print("="*80 + "\n")
        
        return total_loss, feature_loss, attn_loss

def compute_anomaly_score(dynamic_features, static_features, dynamic_attn=None, static_attn=None, temperature=0.07):
    """
    计算异常分数
    Args:
        dynamic_features: 动态流特征
        static_features: 静态流特征
        dynamic_attn: 动态流注意力分数
        static_attn: 静态流注意力分数
        temperature: 温度参数
    Returns:
        anomaly_score: 异常分数
    """
    # 计算时间异常分数
    max_flow = torch.max(dynamic_features, dim=1)[0]
    flow_ratio = dynamic_features / (max_flow.unsqueeze(1) + 1e-6)
    time_scores = torch.mean(flow_ratio, dim=-1)
    
    # 计算空间异常分数
    region_flow = torch.mean(dynamic_features, dim=1)
    region_diff = torch.cdist(region_flow, region_flow)
    
    # 计算特征级别的KL散度
    feature_kl_1 = kl_loss(dynamic_features, static_features.detach())
    feature_kl_2 = kl_loss(static_features, dynamic_features.detach())
    feature_sym_kl = (feature_kl_1 + feature_kl_2) / 2
    
    # 如果提供了注意力分数，计算注意力级别的KL散度
    attn_sym_kl = 0
    if dynamic_attn is not None and static_attn is not None:
        # 确保注意力分数维度匹配
        if dynamic_attn.size() != static_attn.size():
            min_size = min(dynamic_attn.size(0), static_attn.size(0))
            dynamic_attn = dynamic_attn[:min_size, :min_size]
            static_attn = static_attn[:min_size, :min_size]
        
        # 计算注意力级别的KL散度
        attn_kl_1 = kl_loss(dynamic_attn, static_attn.detach())
        attn_kl_2 = kl_loss(static_attn, dynamic_attn.detach())
        attn_sym_kl = (attn_kl_1 + attn_kl_2) / 2
    
    # 计算最终的异常分数
    anomaly_score = (feature_sym_kl + attn_sym_kl + time_scores + region_diff) * temperature
    
    return anomaly_score 