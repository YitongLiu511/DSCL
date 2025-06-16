import torch
import numpy as np
from process_normal_data import process_normal_data

def test_process_normal():
    print("=== 开始测试正常数据处理 ===")
    
    try:
        # 1. 加载时间掩码处理后的训练数据
        print("\n1. 加载时间掩码处理后的训练数据...")
        data = np.load('data/temporal_masked_train.npy')
        print(f"加载数据形状: {data.shape}")
        
        # 2. 处理数据
        print("\n2. 开始处理数据...")
        process_normal_data()
        
        print("\n=== 处理完成 ===")
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        return

if __name__ == "__main__":
    test_process_normal() 