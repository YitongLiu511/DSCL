import torch
import numpy as np

def my_kl_loss(p, q):
    """
    计算KL散度，与 models/model/solver.py 中的实现一致。
    返回每个样本的损失值，而不是批次的平均值。
    """
    res = p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))
    return torch.sum(res, dim=-1)

def calculate_contrastive_loss(tematt_list, freatt_list):
    """
    计算时频对比损失，完全复刻 models/model/solver.py 的逻辑。
    
    Args:
        tematt_list (list of torch.Tensor): 时域特征输出列表。
        freatt_list (list of torch.Tensor): 频域特征输出列表。
        
    Returns:
        tuple: (总损失, 对抗损失, 对比损失)
    """
    adv_loss = 0.0
    con_loss = 0.0

    # 遍历每一层编码器的输出特征
    for u in range(len(freatt_list)):
        tematt = tematt_list[u]
        freatt = freatt_list[u]

        # 确保特征已经归一化（这里假设输入前已归一化，或在模型内部归一化）
        # 在 solver.py 中，它对频域特征在计算损失时动态归一化
        freatt_norm = freatt / (torch.sum(freatt, dim=-1, keepdim=True) + 1e-8)
        
        # 对抗损失 (Adversarial Loss)
        adv_loss_u = (torch.mean(my_kl_loss(tematt, freatt_norm.detach())) +
                      torch.mean(my_kl_loss(freatt_norm.detach(), tematt)))
        
        # 对比损失 (Contrastive Loss)
        con_loss_u = (torch.mean(my_kl_loss(freatt_norm, tematt.detach())) +
                      torch.mean(my_kl_loss(tematt.detach(), freatt_norm)))
        
        adv_loss += adv_loss_u
        con_loss += con_loss_u

    # 对层数求平均
    adv_loss = adv_loss / len(freatt_list)
    con_loss = con_loss / len(freatt_list)

    # 总损失
    total_loss = con_loss - adv_loss
    
    return total_loss, adv_loss, con_loss

if __name__ == "__main__":
    print("--- 开始测试时频对比损失 ---")

    # --- 1. 加载预先计算好的时域和频域特征 ---
    # !!! 请确保这两个文件存在，并且是对应的输出 !!!
    # 我们假设这两个文件是单个tensor，代表某一层encoder的输出
    try:
        # 假设这是由 spatial_attention1 -> encoder 处理后的时域特征
        temporal_feature_path = 'C:/Users/86155/Downloads/DSCLBEW/data/processed/temporal_feature_output.npy'
        
        # 假设这是由 frequency_decoder 处理后的频域特征
        frequency_feature_path = 'C:/Users/86155/Downloads/DSCLBEW/data/processed/frequency_feature_output.npy'
        
        tematt_np = np.load(temporal_feature_path)
        freatt_np = np.load(frequency_feature_path)

        print(f"成功加载时域特征，形状: {tematt_np.shape}")
        print(f"成功加载频域特征，形状: {freatt_np.shape}")

    except FileNotFoundError as e:
        print(f"❌ 错误: 无法加载数据文件。 {e}")
        print("   - 请确保 'temporal_feature_output.npy' 和 'frequency_feature_output.npy' 文件已生成在 'data/processed' 目录下。")
        exit()

    # --- 2. 准备数据 ---
    # 转换为 PyTorch Tensors
    tematt_tensor = torch.from_numpy(tematt_np).float()
    freatt_tensor = torch.from_numpy(freatt_np).float()
    
    # 修正：在计算KL散度前，必须确保输入是有效的概率分布
    # 使用Softmax进行归一化，确保所有值为正且和为1
    print("\n应用Softmax归一化以确保输入为有效概率分布...")
    tematt_tensor = torch.softmax(tematt_tensor, dim=-1)
    freatt_tensor = torch.softmax(freatt_tensor, dim=-1)

    # 因为函数期望的是列表，我们将单个tensor放入列表中
    # 这模拟了只使用编码器最后一层特征进行对比的情况
    tematt_list_mock = [tematt_tensor]
    freatt_list_mock = [freatt_tensor]

    # --- 3. 计算损失 ---
    print("\n正在计算时频对比损失...")
    total_loss, adv_loss, con_loss = calculate_contrastive_loss(tematt_list_mock, freatt_list_mock)

    # --- 4. 打印结果 ---
    print("\n--- 计算结果 ---")
    print(f"对抗损失 (adv_loss): {adv_loss.item():.6f}")
    print(f"对比损失 (con_loss): {con_loss.item():.6f}")
    print(f"最终总损失 (con_loss - adv_loss): {total_loss.item():.6f}")
    print("\n--- 测试完成 ---") 