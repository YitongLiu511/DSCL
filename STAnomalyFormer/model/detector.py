def _compute_adversarial_contrastive_loss(self, time_features, freq_features):
    """计算对抗对比损失"""
    print("\n" + "="*80)
    print("="*80)
    print("【对抗对比损失计算 - 开始】")
    print("="*80)
    print("="*80 + "\n")
    
    # 检查输入特征
    print("输入特征检查:")
    print(f"时间特征列表长度: {len(time_features)}")
    print(f"频率特征列表长度: {len(freq_features)}")
    
    total_loss = 0
    total_adversarial_loss = 0
    total_contrastive_loss = 0
    
    # 对每一层特征计算损失
    for i, (time_feat, freq_feat) in enumerate(zip(time_features, freq_features)):
        print(f"\n第{i+1}层特征:")
        print(f"时间特征维度: {time_feat.shape}")
        print(f"频率特征维度: {freq_feat.shape}")
        
        # 特征归一化
        time_feat_norm = F.normalize(time_feat, p=2, dim=-1)
        freq_feat_norm = F.normalize(freq_feat, p=2, dim=-1)
        
        print("\n归一化后的特征:")
        print(f"时间特征范数: {time_feat_norm.norm(p=2, dim=-1).mean().item():.4f}")
        print(f"频率特征范数: {freq_feat_norm.norm(p=2, dim=-1).mean().item():.4f}")
        
        # 计算相似度矩阵
        similarity = torch.matmul(time_feat_norm, freq_feat_norm.transpose(-2, -1))
        
        print("\n【时间-频率特征相似度】")
        print(f"- 最小值: {similarity.min().item():.4f}")
        print(f"- 最大值: {similarity.max().item():.4f}")
        print(f"- 平均值: {similarity.mean().item():.4f}")
        print(f"- 标准差: {similarity.std().item():.4f}")
        
        # 计算频域特征差异
        freq_diff = torch.abs(time_feat_norm - freq_feat_norm)
        
        print("\n【频域特征差异】")
        print(f"- 最小值: {freq_diff.min().item():.4f}")
        print(f"- 最大值: {freq_diff.max().item():.4f}")
        print(f"- 平均值: {freq_diff.mean().item():.4f}")
        print(f"- 标准差: {freq_diff.std().item():.4f}")
        
        # 计算困难负样本
        batch_size = time_feat.size(0)
        num_patches = time_feat.size(1)
        num_features = time_feat.size(2)
        
        # 重塑特征以便计算
        time_feat_flat = time_feat_norm.view(batch_size * num_patches, num_features)
        freq_feat_flat = freq_feat_norm.view(batch_size * num_patches, num_features)
        
        # 计算所有样本对之间的相似度
        similarity_matrix = torch.matmul(time_feat_flat, freq_feat_flat.t())
        
        # 获取每个样本的困难负样本
        mask = ~torch.eye(batch_size * num_patches, device=similarity_matrix.device)
        hard_neg_similarities = similarity_matrix.masked_fill(mask == 0, float('-inf'))
        hard_neg_indices = hard_neg_similarities.argmax(dim=1)
        
        print("\n【困难负样本统计】")
        print(f"- 总样本数: {batch_size * num_patches}")
        print(f"- 困难负样本相似度范围: [{hard_neg_similarities.max(dim=1)[0].min().item():.4f}, {hard_neg_similarities.max(dim=1)[0].max().item():.4f}]")
        print(f"- 平均困难负样本相似度: {hard_neg_similarities.max(dim=1)[0].mean().item():.4f}")
        
        # 使用困难负样本计算对抗损失
        adversarial_loss = -torch.mean(hard_neg_similarities.max(dim=1)[0])
        
        # 使用困难负样本计算对比损失
        positive_similarities = torch.diagonal(similarity_matrix)
        contrastive_loss = torch.mean(positive_similarities) - torch.mean(hard_neg_similarities.max(dim=1)[0])
        
        # 更新总损失
        total_loss += adversarial_loss + contrastive_loss
        total_adversarial_loss += adversarial_loss
        total_contrastive_loss += contrastive_loss
    
    print("\n" + "="*80)
    print("="*80)
    print("【对抗对比损失计算 - 完成】")
    print(f"总损失: {total_loss.item():.4f}")
    print(f"对抗损失: {total_adversarial_loss.item():.4f}")
    print(f"对比损失: {total_contrastive_loss.item():.4f}")
    print("="*80)
    print("="*80 + "\n")
    
    return total_loss 