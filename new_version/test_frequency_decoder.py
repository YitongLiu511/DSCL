import numpy as np
import torch
from frequency_decoder import FrequencyDecoder
import os

# 设置内存分配器
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def load_and_preprocess_data(batch_size=32, seq_len=144):
    """加载并预处理数据，使用批处理"""
    print("正在加载数据...")
    # 加载掩码后的数据
    masked_data = np.load('data/frequency_masked_train.npy')  # (14, 144, 263, 2)
    print(f"掩码数据形状: {masked_data.shape}")
    
    # 重塑数据为 [263, 2016, 2]
    masked_data = masked_data.reshape(14*144, 263, 2)
    masked_data = np.transpose(masked_data, (1, 0, 2))
    
    # 转换为PyTorch张量
    masked_data = torch.from_numpy(masked_data).float()
    print(f"处理后的掩码数据形状: {masked_data.shape}")
    
    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(masked_data)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False
    )
    
    return dataloader

def test_decoder(dataloader, device='cpu'):
    """测试解码器"""
    print(f"\n使用设备: {device}")
    
    # 初始化模型，减小模型大小
    model = FrequencyDecoder(
        d_model=64,   # 进一步减小模型维度
        n_heads=4,    # 保持注意力头数
        dim_fc=32,    # 减小前馈网络维度
        dropout=0.1,
        n_layers=2,   # 保持层数
        device=device
    ).to(device)
    
    # 清理GPU缓存
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # 前向传播
    print("\n开始前向传播...")
    all_outputs = []
    all_attention_weights = []
    
    with torch.no_grad():
        for batch_idx, (data,) in enumerate(dataloader):
            print(f"处理批次 {batch_idx + 1}/{len(dataloader)}")
            data = data.to(device)
            output, attention_weights = model(data, data)
            all_outputs.append(output)
            all_attention_weights.append(attention_weights)
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f"已完成 {batch_idx + 1} 个批次")
    
    # 合并所有批次的输出
    output = torch.cat(all_outputs, dim=0)
    print(f"最终输出形状: {output.shape}")
    
    return output, all_attention_weights

def main():
    # 加载数据
    dataloader = load_and_preprocess_data(batch_size=32, seq_len=144)
    
    # 测试解码器
    output, attention_weights = test_decoder(dataloader)
    
    print("\n测试完成！")

if __name__ == "__main__":
    main() 