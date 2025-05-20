import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z1, z2):
        # 归一化
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        # 计算相似度矩阵
        sim = torch.matmul(z1, z2.t()) / self.temperature
        
        # 正样本对
        pos_sim = torch.diag(sim)
        
        # 负样本对
        neg_sim = sim - torch.eye(sim.size(0)).to(sim.device) * 1e9
        
        # 计算对比损失
        loss = -pos_sim + torch.logsumexp(neg_sim, dim=1)
        return loss.mean()

class AdversarialTraining(nn.Module):
    def __init__(self, model, epsilon=0.1):
        super().__init__()
        self.model = model
        self.epsilon = epsilon
        
    def forward(self, x, y):
        # 生成对抗样本
        x.requires_grad_(True)
        output = self.model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        
        # 添加扰动
        x_adv = x + self.epsilon * x.grad.sign()
        x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv 