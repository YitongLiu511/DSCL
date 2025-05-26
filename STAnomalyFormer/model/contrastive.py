import torch
import torch.nn.functional as F

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

def compute_contrastive_loss(tematt, freatt, temperature=50.0):
    """
    计算时间掩码和频率掩码分支输出的特征表示之间的对比损失
    Args:
        tematt: 时间掩码分支的输出特征列表
        freatt: 频率掩码分支的输出特征列表
        temperature: 温度参数，用于调整损失的强度
    Returns:
        total_loss: 总损失
        adv_loss: 对抗损失
        con_loss: 对比损失
    """
    adv_loss = 0.0
    con_loss = 0.0
    
    for u in range(len(freatt)):
        # 计算对抗损失
        # 1. 时间特征到频率特征的KL散度
        adv_loss += (torch.mean(my_kl_loss(tematt[u], 
            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach())) + 
            # 2. 频率特征到时间特征的KL散度
            torch.mean(my_kl_loss(
                (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach(),
                tematt[u])))
        
        # 计算对比损失
        # 1. 频率特征到时间特征的KL散度
        con_loss += (torch.mean(my_kl_loss(
            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
            tematt[u].detach())) + 
            # 2. 时间特征到频率特征的KL散度
            torch.mean(my_kl_loss(tematt[u].detach(), 
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