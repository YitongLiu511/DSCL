import torch
import numpy as np
from process_normal_data import process_normal_data

def test_process_normal():
    print("=== 开始测试正常数据处理 ===")
    
    try:
        # 1. 加载归一化后的训练数据
        print("\n1. 加载归一化后的训练数据...")
        data = np.load('data/normalized_train.npy')
        print(f"加载数据形状: {data.shape}")
        
        # 2. 确保processed文件夹存在
        import os
        if not os.path.exists('data/processed'):
            os.makedirs('data/processed')
        
        # 3. 保存为X_normal.npy
        np.save('data/processed/X_normal.npy', data)
        print("已保存X_normal.npy")
        
        # 4. 处理数据
        print("\n2. 开始处理数据...")
        process_normal_data()
        
        print("\n=== 处理完成 ===")
        print("最终结果保存在 data/processed 目录下：")
        print("1. normal_processed_X.npy - 时间注意力处理后的数据")
        print("2. normal_spatial_features.npy - 空间特征")
        print("3. normal_attention_weights_*.npy - 时间注意力权重")
        print("4. normal_spatial_weights.npy - 空间权重")
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        return

if __name__ == "__main__":
    test_process_normal() 