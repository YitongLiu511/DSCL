import numpy as np
import torch
import torch.nn as nn
from frequency_decoder import FrequencyDecoder
import matplotlib.pyplot as plt

def load_and_preprocess_data():
    """加载并预处理数据"""
    print("正在加载数据...")
    # 加载掩码后的数据
    masked_data = np.load('../data/frequency_masked_train.npy')  # (14, 144, 263, 2)
    print(f"掩码数据形状: {masked_data.shape}")
    
    # 加载掩码索引
    mask_indices = np.load('../data/frequency_mask_indices.npy')
    print(f"掩码索引形状: {mask_indices.shape}")
    
    # 重塑数据为 [263, 2016, 2]
    masked_data = masked_data.reshape(14*144, 263, 2)
    masked_data = np.transpose(masked_data, (1, 0, 2))
    masked_data = torch.from_numpy(masked_data).float()
    print(f"处理后的掩码数据形状: {masked_data.shape}")
    
    return masked_data, mask_indices

def test_decoder(masked_data, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """测试解码器"""
    print(f"\n使用设备: {device}")
    
    # 初始化模型
    model = FrequencyDecoder(
        d_model=256,
        n_heads=8,
        dim_fc=128,
        dropout=0.1,
        n_layers=3,
        device=device
    ).to(device)
    
    # 将数据移到设备上
    masked_data = masked_data.to(device)
    
    # 前向传播
    print("\n开始前向传播...")
    with torch.no_grad():
        output, attention_weights = model(masked_data, masked_data)
    
    print(f"输出形状: {output.shape}")
    print(f"注意力权重数量: {len(attention_weights)}")
    
    return output, attention_weights

def evaluate_masked_reconstruction(original_data, output_data, mask_indices):
    """评估掩码重建效果"""
    # 将掩码索引转换为张量
    mask_indices = torch.from_numpy(mask_indices).bool()
    
    # 计算掩码位置的MSE
    masked_mse = torch.mean((output_data[mask_indices] - original_data[mask_indices]) ** 2)
    
    # 计算非掩码位置的MSE
    unmasked_mse = torch.mean((output_data[~mask_indices] - original_data[~mask_indices]) ** 2)
    
    print("\n掩码重建评估:")
    print(f"掩码位置MSE: {masked_mse.item():.6f}")
    print(f"非掩码位置MSE: {unmasked_mse.item():.6f}")
    
    return masked_mse.item(), unmasked_mse.item()

def visualize_masked_reconstruction(original_data, output_data, mask_indices, save_path='../results/masked_reconstruction.png'):
    """可视化掩码重建效果"""
    # 选择第一个空间点进行可视化
    spatial_point = 0
    time_steps = range(100)  # 只显示前100个时间步
    
    plt.figure(figsize=(15, 5))
    
    # 绘制第一个特征
    plt.subplot(1, 2, 1)
    plt.plot(time_steps, original_data[spatial_point, time_steps, 0].cpu().numpy(), label='原始数据', alpha=0.5)
    plt.plot(time_steps, output_data[spatial_point, time_steps, 0].cpu().numpy(), label='重建数据')
    
    # 标记掩码位置
    mask_points = mask_indices[spatial_point, time_steps, 0]
    plt.scatter(time_steps[mask_points], original_data[spatial_point, time_steps, 0][mask_points].cpu().numpy(), 
               color='red', label='掩码位置', alpha=0.5)
    
    plt.title('特征1重建效果')
    plt.legend()
    
    # 绘制第二个特征
    plt.subplot(1, 2, 2)
    plt.plot(time_steps, original_data[spatial_point, time_steps, 1].cpu().numpy(), label='原始数据', alpha=0.5)
    plt.plot(time_steps, output_data[spatial_point, time_steps, 1].cpu().numpy(), label='重建数据')
    
    # 标记掩码位置
    mask_points = mask_indices[spatial_point, time_steps, 1]
    plt.scatter(time_steps[mask_points], original_data[spatial_point, time_steps, 1][mask_points].cpu().numpy(), 
               color='red', label='掩码位置', alpha=0.5)
    
    plt.title('特征2重建效果')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\n可视化结果已保存到: {save_path}")

def main():
    # 加载数据
    masked_data, mask_indices = load_and_preprocess_data()
    
    # 测试解码器
    output, attention_weights = test_decoder(masked_data)
    
    # 评估掩码重建效果
    masked_mse, unmasked_mse = evaluate_masked_reconstruction(masked_data, output, mask_indices)
    
    # 可视化掩码重建效果
    visualize_masked_reconstruction(masked_data, output, mask_indices)
    
    # 保存结果
    results = {
        'output': output.cpu(),
        'attention_weights': attention_weights,
        'metrics': {
            'masked_mse': masked_mse,
            'unmasked_mse': unmasked_mse
        }
    }
    torch.save(results, '../results/masked_reconstruction_results.pt')
    print("\n测试结果已保存到: ../results/masked_reconstruction_results.pt")

if __name__ == "__main__":
    main() 