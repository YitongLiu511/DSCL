import torch
import numpy as np
from process_normal_data import process_normal_data
import os

def test_process_normal():
    print("=== 开始测试正常数据处理 ===")
    
    try:
        # 1. 加载异常数据
        print("\n1. 加载异常数据...")
        data = np.load('data/processed/train_data_with_anomalies_3d.npy')
        print(f"加载数据形状: {data.shape}")
        
        # 2. 检查时间注意力缓存
        print("\n2. 检查时间注意力缓存...")
        temporal_cache = 'data/processed/temporal_attention_processed.npy'
        if not os.path.exists(temporal_cache):
            print("未检测到时间注意力缓存，开始运行 temporal_attention.py ...")
            os.system('python new_version/temporal_attention.py')
        else:
            print("检测到时间注意力缓存，跳过时间注意力处理。")
        
        # 3. 处理数据
        print("\n3. 开始处理数据...")
        process_normal_data()
        
        print("\n=== 处理完成 ===")
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        return

if __name__ == "__main__":
    test_process_normal() 