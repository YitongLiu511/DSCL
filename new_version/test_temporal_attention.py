import torch
import numpy as np
from temporal_attention import process_temporal_masked_data
from tqdm import tqdm
import traceback
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # 转换为MB

def test_temporal_attention():
    try:
        # 加载时间掩码后的训练集
        print("加载数据...")
        masked_data = np.load('data/temporal_masked_train.npy')
        print(f"数据形状: {masked_data.shape}")
        print(f"初始内存使用: {get_memory_usage():.2f} MB")
        
        # 转换为PyTorch张量
        masked_data = torch.from_numpy(masked_data).float()
        
        # 强制使用CPU
        device = torch.device('cpu')
        
        # 创建输出目录
        output_dir = 'data/temporal_attention_processed'
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理每个节点
        num_nodes = masked_data.shape[2]
        
        print("\n开始处理...")
        for i in range(num_nodes):
            try:
                # 获取当前节点的所有时间步
                node_data = masked_data[:, :, i:i+1, :]  # [14, 144, 1, 2]
                
                # 处理当前节点
                processed_data, _ = process_temporal_masked_data(
                    node_data,
                    d_model=256,  # 保持原始模型维度
                    dim_k=32,     # 保持原始键维度
                    dim_v=32,     # 保持原始值维度
                    n_heads=4,    # 保持原始注意力头数
                    dim_fc=64,    # 保持原始前馈网络维度
                    device=device
                )
                
                # 立即保存当前节点的处理结果
                node_output = processed_data.detach().cpu().numpy()  # 先detach再转numpy
                np.save(f'{output_dir}/node_{i:03d}.npy', node_output)
                
                # 清理内存
                del processed_data
                del node_output
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # 每50轮打印一次进度
                if (i + 1) % 50 == 0:
                    print(f"进度: {i + 1}/{num_nodes} 节点处理成功")
                    print(f"当前内存使用: {get_memory_usage():.2f} MB")
                    
            except Exception as e:
                print(f"\n处理节点 {i} 时出错:")
                print(f"错误类型: {type(e).__name__}")
                print(f"错误信息: {str(e)}")
                print("错误详情:")
                print(traceback.format_exc())
                raise e
        
        print("\n所有节点处理完成，开始合并...")
        print(f"合并前内存使用: {get_memory_usage():.2f} MB")
        
        # 合并所有节点的输出
        all_data = []
        for i in range(num_nodes):
            node_data = np.load(f'{output_dir}/node_{i:03d}.npy')
            all_data.append(node_data)
        
        processed_data = np.concatenate(all_data, axis=0)
        print(f"合并后内存使用: {get_memory_usage():.2f} MB")
        
        # 保存最终结果
        print("保存最终数据...")
        np.save('data/temporal_attention_processed_train.npy', processed_data)
        
        # 清理临时文件
        print("清理临时文件...")
        for i in range(num_nodes):
            os.remove(f'{output_dir}/node_{i:03d}.npy')
        os.rmdir(output_dir)
        
        print(f"\n全部完成！共处理 {num_nodes} 个节点")
        print(f"输出数据形状: {processed_data.shape}")
        print(f"最终内存使用: {get_memory_usage():.2f} MB")
        
    except Exception as e:
        print("\n程序执行出错:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("错误详情:")
        print(traceback.format_exc())

if __name__ == '__main__':
    test_temporal_attention() 