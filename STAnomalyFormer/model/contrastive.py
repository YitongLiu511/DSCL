import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Beta

def my_kl_loss(p, q):
    """
    计算KL散度损失
    Args:
        p: 第一个分布
        q: 第二个分布
    Returns:
        KL散度损失
    """
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.sum(res, dim=-1)

class BetaMixtureModel(nn.Module):
    """基于最大期望的贝塔混合分布模型"""
    def __init__(self, n_components=2, max_iter=100, tol=1e-4):
        super().__init__()
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        
        # 初始化混合权重
        self.mixture_weights = nn.Parameter(torch.ones(n_components) / n_components)
        
        # 初始化每个贝塔分布的参数
        self.alphas = nn.Parameter(torch.ones(n_components))
        self.betas = nn.Parameter(torch.ones(n_components))
        
    def forward(self, x):
        """
        计算每个样本属于各个贝塔分布的概率
        Args:
            x: 输入数据 [B, N]
        Returns:
            responsibilities: 每个样本属于各个分布的概率 [B, N, n_components]
            log_likelihood: 对数似然
        """
        # 确保输入在[0,1]范围内
        x = torch.clamp(x, 0, 1)
        
        # 创建贝塔分布
        beta_dist = Beta(self.alphas, self.betas)
        
        # 计算每个样本在每个分布下的概率密度
        log_probs = beta_dist.log_prob(x.unsqueeze(-1))  # [B, N, n_components]
        
        # 添加混合权重的对数
        log_probs = log_probs + torch.log(self.mixture_weights)
        
        # 计算对数似然
        log_likelihood = torch.logsumexp(log_probs, dim=-1)
        
        # 计算每个样本属于各个分布的概率（responsibilities）
        responsibilities = torch.exp(log_probs - log_likelihood.unsqueeze(-1))
        
        return responsibilities, log_likelihood
    
    def fit(self, x):
        """
        使用EM算法拟合贝塔混合分布
        Args:
            x: 输入数据 [B, N]
        """
        for _ in range(self.max_iter):
            # E步：计算responsibilities
            responsibilities, log_likelihood = self.forward(x)
            
            # M步：更新参数
            # 更新混合权重
            self.mixture_weights.data = responsibilities.mean(dim=(0, 1))
            
            # 更新贝塔分布参数
            for k in range(self.n_components):
                # 计算加权统计量
                weighted_x = x * responsibilities[..., k]
                weighted_log_x = torch.log(x + 1e-8) * responsibilities[..., k]
                weighted_log_1_x = torch.log(1 - x + 1e-8) * responsibilities[..., k]
                
                # 更新alpha和beta
                self.alphas.data[k] = self._update_alpha(
                    weighted_x, weighted_log_x, weighted_log_1_x, responsibilities[..., k]
                )
                self.betas.data[k] = self._update_beta(
                    weighted_x, weighted_log_x, weighted_log_1_x, responsibilities[..., k]
                )
    
    def _update_alpha(self, weighted_x, weighted_log_x, weighted_log_1_x, responsibilities):
        """更新alpha参数"""
        # 使用牛顿法求解
        alpha = self.alphas.data.clone()
        for _ in range(10):  # 最多迭代10次
            psi_alpha = torch.digamma(alpha)
            psi_alpha_beta = torch.digamma(alpha + self.betas.data)
            
            # 计算梯度
            grad = responsibilities.sum() * (psi_alpha - psi_alpha_beta) + weighted_log_x.sum()
            
            # 计算Hessian
            hessian = responsibilities.sum() * (torch.polygamma(1, alpha) - torch.polygamma(1, alpha + self.betas.data))
            
            # 更新alpha
            alpha = alpha - grad / (hessian + 1e-8)
            alpha = torch.clamp(alpha, min=1e-8)
            
        return alpha
    
    def _update_beta(self, weighted_x, weighted_log_x, weighted_log_1_x, responsibilities):
        """更新beta参数"""
        # 使用牛顿法求解
        beta = self.betas.data.clone()
        for _ in range(10):  # 最多迭代10次
            psi_beta = torch.digamma(beta)
            psi_alpha_beta = torch.digamma(self.alphas.data + beta)
            
            # 计算梯度
            grad = responsibilities.sum() * (psi_beta - psi_alpha_beta) + weighted_log_1_x.sum()
            
            # 计算Hessian
            hessian = responsibilities.sum() * (torch.polygamma(1, beta) - torch.polygamma(1, self.alphas.data + beta))
            
            # 更新beta
            beta = beta - grad / (hessian + 1e-8)
            beta = torch.clamp(beta, min=1e-8)
            
        return beta

def compute_contrastive_loss(tematt, freatt, temperature=50.0, weekday_info=None):
    """
    计算时间掩码和频率掩码分支输出的特征表示之间的对比损失
    Args:
        tematt: 时间掩码分支的输出特征列表
        freatt: 频率掩码分支的输出特征列表
        temperature: 温度参数，用于调整损失的强度
        weekday_info: 星期几信息 [B, T]，值为0-6表示周一到周日
    Returns:
        total_loss: 总损失
        adv_loss: 对抗损失
        con_loss: 对比损失
    """
    adv_loss = 0.0
    con_loss = 0.0
    
    # 创建贝塔混合分布模型
    beta_mixture = BetaMixtureModel(n_components=2)
    
    for u in range(len(freatt)):
        # 计算时间特征和频率特征
        time_feat = tematt[u] / torch.unsqueeze(torch.sum(tematt[u], dim=-1), dim=-1)
        freq_feat = freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(time_feat, freq_feat, dim=-1)
        
        # 使用贝塔混合分布拟合相似度分布
        beta_mixture.fit(similarity)
        
        # 获取每个样本属于各个分布的概率
        responsibilities, _ = beta_mixture(similarity)
        
        # 使用概率作为权重来调整损失
        weights = responsibilities[:, :, 1]  # 使用第二个分布的概率作为权重
        
        # 计算频域特征的差异
        freq_diff = torch.cdist(freq_feat, freq_feat)
        
        # 使用贝塔混合分布拟合频域差异分布
        beta_mixture.fit(freq_diff)
        
        # 获取每个样本属于各个分布的概率
        freq_responsibilities, _ = beta_mixture(freq_diff)
        
        # 使用频域差异的概率作为额外的权重
        freq_weights = freq_responsibilities[:, :, 1]  # 使用第二个分布的概率作为权重
        
        # 将频域权重与原有权重结合
        combined_weights = weights * freq_weights
        
        # 如果有星期几信息，进一步优化权重
        if weekday_info is not None:
            # 创建星期几的掩码矩阵
            weekday_mask = (weekday_info.unsqueeze(1) == weekday_info.unsqueeze(2))  # [B, T, T]
            
            # 计算时间特征差异
            time_diff = torch.cdist(time_feat, time_feat)
            
            # 使用贝塔混合分布拟合时间特征差异分布
            beta_mixture.fit(time_diff)
            time_responsibilities, _ = beta_mixture(time_diff)
            time_weights = time_responsibilities[:, :, 1]
            
            # 对于相同星期几的样本：
            # 1. 如果时间特征相似（可能是正常样本），降低权重
            # 2. 如果时间特征相似但频域特征不同（可能是困难负样本），增加权重
            same_weekday_weights = torch.where(
                weekday_mask,
                time_weights * (1 - freq_weights),  # 时间相似但频域不同
                torch.ones_like(combined_weights)   # 不同星期几的样本保持原有权重
            )
            
            # 更新组合权重
            combined_weights = combined_weights * same_weekday_weights
        
        # 计算对抗损失
        # 1. 时间特征到频率特征的KL散度
        adv_loss += (torch.mean(combined_weights * my_kl_loss(tematt[u], 
            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach())) + 
            # 2. 频率特征到时间特征的KL散度
            torch.mean(combined_weights * my_kl_loss(
                (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach(),
                tematt[u])))
        
        # 计算对比损失
        # 1. 频率特征到时间特征的KL散度
        con_loss += (torch.mean(combined_weights * my_kl_loss(
            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
            tematt[u].detach())) + 
            # 2. 时间特征到频率特征的KL散度
            torch.mean(combined_weights * my_kl_loss(tematt[u].detach(), 
                (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)))))
    
    # 对损失进行归一化
    adv_loss = adv_loss / len(freatt)
    con_loss = con_loss / len(freatt)
    
    # 应用温度参数
    adv_loss = adv_loss * temperature
    con_loss = con_loss * temperature
    
    # 总损失 = 对比损失 - 对抗损失
    total_loss = con_loss - adv_loss
    
    return total_loss, adv_loss, con_loss

def compute_anomaly_score(tematt, freatt, temperature=50.0):
    """
    计算异常分数
    Args:
        tematt: 时间掩码分支的输出特征列表
        freatt: 频率掩码分支的输出特征列表
        temperature: 温度参数
    Returns:
        anomaly_score: 异常分数
    """
    adv_loss = 0.0
    con_loss = 0.0
    
    for u in range(len(freatt)):
        if u == 0:
            # 第一个时间步的损失
            adv_loss = my_kl_loss(tematt[u], 
                (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach()) * temperature
            con_loss = my_kl_loss(
                (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                tematt[u].detach()) * temperature
        else:
            # 累加后续时间步的损失
            adv_loss += my_kl_loss(tematt[u], 
                (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach()) * temperature
            con_loss += my_kl_loss(
                (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                tematt[u].detach()) * temperature
    
    # 计算最终的异常分数
    metric = torch.softmax((adv_loss + con_loss), dim=-1)
    
    return metric

def compute_adversarial_contrastive_loss(tematt, freatt, temperature=50.0):
    """
    计算对抗对比损失
    Args:
        tematt: 时间掩码分支的输出特征列表
        freatt: 频率掩码分支的输出特征列表
        temperature: 温度参数
    Returns:
        total_loss: 总损失
        normal_loss: 正常样本的对比损失
        anomaly_loss: 异常样本的对比损失
    """
    # 1. 计算正常样本的对比损失（最小化）
    normal_loss = compute_contrastive_loss(tematt, freatt, temperature)[0]
    
    # 2. 计算异常样本的对比损失（最大化）
    anomaly_loss = -compute_contrastive_loss(tematt, freatt, temperature)[0]
    
    # 3. 总损失
    total_loss = normal_loss + anomaly_loss
    
    return total_loss, normal_loss, anomaly_loss

def generate_anomaly_features(x, model):
    """
    生成异常样本特征
    Args:
        x: 输入数据 [bs, seq_len, n_vars]
        model: 模型实例
    Returns:
        anomaly_features: 异常样本特征
    """
    # 1. 添加随机噪声
    noise = torch.randn_like(x) * 0.1
    x_noisy = x + noise
    
    # 2. 时间维度上的扰动
    x_shifted = torch.roll(x, shifts=1, dims=1)
    
    # 3. 获取特征
    noisy_features = model.get_features(x_noisy)
    shifted_features = model.get_features(x_shifted)
    
    return [noisy_features, shifted_features] 