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
    # 检查输入是否包含NaN或Inf
    if torch.isnan(p).any() or torch.isinf(p).any():
        print("警告：p包含NaN或Inf值")
        p = torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=-1.0)
    
    if torch.isnan(q).any() or torch.isinf(q).any():
        print("警告：q包含NaN或Inf值")
        q = torch.nan_to_num(q, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 确保输入在[0,1]范围内
    p = torch.clamp(p, 0, 1)
    q = torch.clamp(q, 0, 1)
    
    # 添加小的常数以避免数值不稳定
    epsilon = 1e-8
    p = p + epsilon
    q = q + epsilon
    
    # 计算KL散度
    res = p * (torch.log(p) - torch.log(q))
    
    # 检查结果是否包含NaN或Inf
    if torch.isnan(res).any() or torch.isinf(res).any():
        print("警告：KL散度计算结果包含NaN或Inf值")
        res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
    
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
        log_probs = log_probs + torch.log(self.mixture_weights + 1e-8)
        
        # 使用更稳定的方法计算对数似然
        max_log_probs = torch.max(log_probs, dim=-1, keepdim=True)[0]
        log_probs_shifted = log_probs - max_log_probs
        exp_log_probs = torch.exp(log_probs_shifted)
        log_likelihood = max_log_probs.squeeze(-1) + torch.log(torch.sum(exp_log_probs, dim=-1) + 1e-8)
        
        # 计算每个样本属于各个分布的概率（responsibilities）
        responsibilities = exp_log_probs / (torch.sum(exp_log_probs, dim=-1, keepdim=True) + 1e-8)
        
        return responsibilities, log_likelihood
    
    def fit(self, x):
        """
        使用EM算法拟合贝塔混合分布
        Args:
            x: 输入数据 [B, N]
        """
        # 输入保护，避免log(0)或log(1)
        x = torch.clamp(x, 1e-5, 1-1e-5)
        prev_log_likelihood = float('-inf')
        
        # 确保输入是二维的
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        for _ in range(self.max_iter):
            # E步：计算responsibilities
            responsibilities, log_likelihood = self.forward(x)
            
            # 检查收敛性
            current_log_likelihood = log_likelihood.mean().item()
            if abs(current_log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = current_log_likelihood
            
            # M步：更新参数
            # 更新混合权重
            self.mixture_weights.data = responsibilities.mean(dim=(0, 1)).clamp(1e-5, 1-1e-5)
            
            # 更新贝塔分布参数
            for k in range(self.n_components):
                # 计算加权统计量
                weighted_x = x * responsibilities[..., k]
                weighted_log_x = torch.log(x + 1e-8) * responsibilities[..., k]
                weighted_log_1_x = torch.log(1 - x + 1e-8) * responsibilities[..., k]
                
                # 确保所有输入都是二维的
                if weighted_x.dim() == 1:
                    weighted_x = weighted_x.unsqueeze(0)
                if weighted_log_x.dim() == 1:
                    weighted_log_x = weighted_log_x.unsqueeze(0)
                if weighted_log_1_x.dim() == 1:
                    weighted_log_1_x = weighted_log_1_x.unsqueeze(0)
                if responsibilities[..., k].dim() == 1:
                    responsibilities_k = responsibilities[..., k].unsqueeze(0)
                else:
                    responsibilities_k = responsibilities[..., k]
                
                # 更新alpha和beta，带数值保护
                alpha = self._update_alpha(
                    weighted_x, weighted_log_x, weighted_log_1_x, responsibilities_k
                )
                beta = self._update_beta(
                    weighted_x, weighted_log_x, weighted_log_1_x, responsibilities_k
                )
                # 检查数值有效性
                if torch.isnan(alpha) or torch.isinf(alpha) or alpha <= 0:
                    alpha = torch.tensor(1.0, device=alpha.device if isinstance(alpha, torch.Tensor) else None)
                if torch.isnan(beta) or torch.isinf(beta) or beta <= 0:
                    beta = torch.tensor(1.0, device=beta.device if isinstance(beta, torch.Tensor) else None)
                self.alphas.data[k] = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
                self.betas.data[k] = beta.item() if isinstance(beta, torch.Tensor) else beta
        # 最终再clamp一次
        self.alphas.data.clamp_(1e-3, 1e3)
        self.betas.data.clamp_(1e-3, 1e3)
    
    def _update_alpha(self, weighted_x, weighted_log_x, weighted_log_1_x, responsibilities):
        """更新alpha参数，带数值保护"""
        alpha = self.alphas.data.mean().clamp(1e-3, 1e3)
        beta = self.betas.data.mean().clamp(1e-3, 1e3)
        for _ in range(10):
            psi_alpha = torch.digamma(alpha)
            psi_alpha_beta = torch.digamma(alpha + beta)
            grad = responsibilities.sum() * (psi_alpha - psi_alpha_beta) + weighted_log_x.sum()
            hessian = responsibilities.sum() * (torch.polygamma(1, alpha) - torch.polygamma(1, alpha + beta))
            if torch.abs(hessian) < 1e-6:
                break
            alpha_new = alpha - grad / (hessian + 1e-8)
            if torch.isnan(alpha_new) or torch.isinf(alpha_new):
                break
            alpha = alpha_new.clamp(1e-3, 1e3)
        return alpha
    
    def _update_beta(self, weighted_x, weighted_log_x, weighted_log_1_x, responsibilities):
        """更新beta参数，带数值保护"""
        alpha = self.alphas.data.mean().clamp(1e-3, 1e3)
        beta = self.betas.data.mean().clamp(1e-3, 1e3)
        for _ in range(10):
            psi_beta = torch.digamma(beta)
            psi_alpha_beta = torch.digamma(alpha + beta)
            grad = responsibilities.sum() * (psi_beta - psi_alpha_beta) + weighted_log_1_x.sum()
            hessian = responsibilities.sum() * (torch.polygamma(1, beta) - torch.polygamma(1, alpha + beta))
            if torch.abs(hessian) < 1e-6:
                break
            beta_new = beta - grad / (hessian + 1e-8)
            if torch.isnan(beta_new) or torch.isinf(beta_new):
                break
            beta = beta_new.clamp(1e-3, 1e3)
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
    print("\n" + "="*80)
    print("【困难负样本构建 - 开始】")
    print("="*80)
    
    adv_loss = 0.0
    con_loss = 0.0
    
    # 创建贝塔混合分布模型
    beta_mixture = BetaMixtureModel(n_components=2)
    
    for u in range(len(freatt)):
        print("\n" + "-"*80)
        print(f"【第{u+1}层特征处理】")
        print("-"*80)
        
        # 获取特征
        time_feat = tematt[u]  # [bs*n_vars, num_patch, d_model]
        freq_feat = freatt[u]  # [bs*n_vars, num_patch, d_model]
        
        print(f"特征维度:")
        print(f"时间特征: {time_feat.shape}")
        print(f"频率特征: {freq_feat.shape}")
        
        # 检查特征是否包含NaN或Inf
        if torch.isnan(time_feat).any() or torch.isinf(time_feat).any():
            print("警告：时间特征包含NaN或Inf值")
            time_feat = torch.nan_to_num(time_feat, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(freq_feat).any() or torch.isinf(freq_feat).any():
            print("警告：频率特征包含NaN或Inf值")
            freq_feat = torch.nan_to_num(freq_feat, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # L2归一化
        time_feat = F.normalize(time_feat, p=2, dim=-1)
        freq_feat = F.normalize(freq_feat, p=2, dim=-1)
        
        print(f"\n归一化后的特征:")
        print(f"时间特征范数: {torch.norm(time_feat, p=2, dim=-1).mean().item():.4f}")
        print(f"频率特征范数: {torch.norm(freq_feat, p=2, dim=-1).mean().item():.4f}")
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(time_feat, freq_feat, dim=-1)  # [bs*n_vars, num_patch]
        print(f"\n【时间-频率特征相似度】")
        print(f"- 最小值: {similarity.min().item():.4f}")
        print(f"- 最大值: {similarity.max().item():.4f}")
        print(f"- 平均值: {similarity.mean().item():.4f}")
        print(f"- 标准差: {similarity.std().item():.4f}")
        
        # 使用贝塔混合分布拟合相似度分布
        beta_mixture.fit(similarity)
        
        # 获取每个样本属于各个分布的概率
        responsibilities, _ = beta_mixture(similarity)
        
        # 使用概率作为权重来调整损失
        weights = responsibilities[:, :, 1]  # 使用第二个分布的概率作为权重
        print(f"\n【时间-频率相似度权重】")
        print(f"- 最小值: {weights.min().item():.4f}")
        print(f"- 最大值: {weights.max().item():.4f}")
        print(f"- 平均值: {weights.mean().item():.4f}")
        print(f"- 标准差: {weights.std().item():.4f}")
        
        # 计算频域特征的差异
        freq_diff = torch.cdist(freq_feat, freq_feat)  # [bs*n_vars, num_patch, num_patch]
        print(f"\n【频域特征差异】")
        print(f"- 最小值: {freq_diff.min().item():.4f}")
        print(f"- 最大值: {freq_diff.max().item():.4f}")
        print(f"- 平均值: {freq_diff.mean().item():.4f}")
        print(f"- 标准差: {freq_diff.std().item():.4f}")
        
        # 使用贝塔混合分布拟合频域差异分布
        beta_mixture.fit(freq_diff)
        
        # 获取每个样本属于各个分布的概率
        freq_responsibilities, _ = beta_mixture(freq_diff)
        
        # 使用频域差异的概率作为额外的权重
        freq_weights = freq_responsibilities[:, :, 1]  # 使用第二个分布的概率作为权重
        print(f"\n【频域差异权重】")
        print(f"- 最小值: {freq_weights.min().item():.4f}")
        print(f"- 最大值: {freq_weights.max().item():.4f}")
        print(f"- 平均值: {freq_weights.mean().item():.4f}")
        print(f"- 标准差: {freq_weights.std().item():.4f}")
        
        # 将频域权重与原有权重结合
        combined_weights = weights * freq_weights
        print(f"\n【组合权重】")
        print(f"- 最小值: {combined_weights.min().item():.4f}")
        print(f"- 最大值: {combined_weights.max().item():.4f}")
        print(f"- 平均值: {combined_weights.mean().item():.4f}")
        print(f"- 标准差: {combined_weights.std().item():.4f}")
        
        # 计算对抗损失
        # 1. 时间特征到频率特征的KL散度
        adv_loss += (torch.mean(combined_weights * my_kl_loss(time_feat, freq_feat.detach())) + 
            # 2. 频率特征到时间特征的KL散度
            torch.mean(combined_weights * my_kl_loss(freq_feat.detach(), time_feat)))
        
        # 计算对比损失
        # 1. 频率特征到时间特征的KL散度
        con_loss += (torch.mean(combined_weights * my_kl_loss(freq_feat, time_feat.detach())) + 
            # 2. 时间特征到频率特征的KL散度
            torch.mean(combined_weights * my_kl_loss(time_feat.detach(), freq_feat)))
    
    # 对损失进行归一化
    adv_loss = adv_loss / len(freatt)
    con_loss = con_loss / len(freatt)
    
    # 应用温度参数
    adv_loss = adv_loss * temperature
    con_loss = con_loss * temperature
    
    # 总损失 = 对比损失 - 对抗损失
    total_loss = con_loss - adv_loss
    
    print("\n" + "="*80)
    print("【最终损失值】")
    print("="*80)
    print(f"[困难负样本构建] 对抗损失: {adv_loss.item():.4f}")
    print(f"[困难负样本构建] 对比损失: {con_loss.item():.4f}")
    print(f"[困难负样本构建] 总损失: {total_loss.item():.4f}")
    print("="*80 + "\n")
    
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

def compute_adversarial_contrastive_loss(time_features_list, freq_features_list, temperature=0.07):
    """
    计算对抗对比损失
    Args:
        time_features_list: 时间特征列表，每个元素形状为 [batch_size, num_patch, n_vars, d_model]
        freq_features_list: 频率特征列表，每个元素形状为 [batch_size, num_patch, n_vars, d_model]
        temperature: 温度参数
    Returns:
        total_loss: 总损失
        adv_loss: 对抗损失
        con_loss: 对比损失
    """
    print("\n" + "="*80)
    print("【对抗对比损失计算 - 开始】")
    print("="*80 + "\n")
    
    # 检查输入特征
    print("输入特征检查:")
    print(f"时间特征列表长度: {len(time_features_list)}")
    print(f"频率特征列表长度: {len(freq_features_list)}")
    
    total_loss = 0
    total_adv_loss = 0
    total_con_loss = 0
    
    for i, (time_feat, freq_feat) in enumerate(zip(time_features_list, freq_features_list)):
        print(f"\n第{i+1}层特征:")
        print(f"时间特征维度: {time_feat.shape}")
        print(f"频率特征维度: {freq_feat.shape}")
        
        # 调整维度顺序
        time_feat = time_feat.permute(0, 2, 1, 3)  # [batch_size, n_vars, num_patch, d_model]
        freq_feat = freq_feat.permute(0, 2, 1, 3)  # [batch_size, n_vars, num_patch, d_model]
        
        # 重塑为正确的维度
        bs, n_vars, num_patch, d_model = time_feat.shape
        time_feat = time_feat.reshape(bs * n_vars, num_patch, d_model)  # [bs*n_vars, num_patch, d_model]
        freq_feat = freq_feat.reshape(bs * n_vars, num_patch, d_model)  # [bs*n_vars, num_patch, d_model]
        
        # L2归一化
        time_feat = F.normalize(time_feat, p=2, dim=-1)
        freq_feat = F.normalize(freq_feat, p=2, dim=-1)
        
        print("\n归一化后的特征:")
        print(f"时间特征范数: {torch.norm(time_feat, p=2, dim=-1).mean().item():.4f}")
        print(f"频率特征范数: {torch.norm(freq_feat, p=2, dim=-1).mean().item():.4f}")
        
        # 计算相似度矩阵
        similarity = torch.matmul(time_feat, freq_feat.transpose(-2, -1)) / temperature
        
        # 计算时间-频率特征相似度
        time_freq_sim = torch.diagonal(similarity, dim1=-2, dim2=-1)
        
        print("\n【时间-频率特征相似度】")
        print(f"- 最小值: {time_freq_sim.min().item():.4f}")
        print(f"- 最大值: {time_freq_sim.max().item():.4f}")
        print(f"- 平均值: {time_freq_sim.mean().item():.4f}")
        print(f"- 标准差: {time_freq_sim.std().item():.4f}")
        
        # 计算频域特征差异
        freq_diff = torch.norm(time_feat - freq_feat, p=2, dim=-1)
        
        print("\n【频域特征差异】")
        print(f"- 最小值: {freq_diff.min().item():.4f}")
        print(f"- 最大值: {freq_diff.max().item():.4f}")
        print(f"- 平均值: {freq_diff.mean().item():.4f}")
        print(f"- 标准差: {freq_diff.std().item():.4f}")
        
        # 计算权重
        weights = torch.softmax(time_freq_sim, dim=-1)
        freq_weights = torch.softmax(-freq_diff, dim=-1)
        
        # 确保权重维度匹配
        weights = weights.unsqueeze(-1)  # [bs*n_vars, num_patch, 1]
        freq_weights = freq_weights.unsqueeze(-1)  # [bs*n_vars, num_patch, 1]
        
        # 计算加权损失
        adv_loss = -torch.mean(weights * time_freq_sim.unsqueeze(-1))
        con_loss = torch.mean(freq_weights * freq_diff.unsqueeze(-1))
        
        # 累加损失
        total_loss += adv_loss + con_loss
        total_adv_loss += adv_loss
        total_con_loss += con_loss
    
    # 计算平均损失
    num_layers = len(time_features_list)
    total_loss = total_loss / num_layers
    total_adv_loss = total_adv_loss / num_layers
    total_con_loss = total_con_loss / num_layers
    
    print("\n" + "="*80)
    print("【对抗对比损失计算 - 完成】")
    print(f"总损失: {total_loss.item():.4f}")
    print(f"对抗损失: {total_adv_loss.item():.4f}")
    print(f"对比损失: {total_con_loss.item():.4f}")
    print("="*80 + "\n")
    
    return total_loss, total_adv_loss, total_con_loss

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