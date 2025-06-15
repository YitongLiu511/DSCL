import torch
import numpy as np
from spatial_attention import SpatialAttentionProcessor

def test_spatial_attention():
    print("开始测试空间注意力处理...")
    
    # 加载时间注意力处理后的数据
    try:
        data = np.load('data/temporal_attention_processed_train.npy')
        print(f"成功加载数据，形状: {data.shape}")
    except Exception as e:
        print(f"加载数据失败: {str(e)}")
        return
    
    # 创建处理器
    processor = SpatialAttentionProcessor()
    print(f"使用设备: {processor.device}")
    
    try:
        # 处理数据
        print("开始处理数据...")
        output = processor.forward(data)
        print(f"处理完成，输出形状: {output.shape}")
        
        # 保存结果
        np.save('data/spatial_attention_processed_train.npy', output)
        print("结果已保存到 data/spatial_attention_processed_train.npy")
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        return
    
    print("测试完成！")

if __name__ == "__main__":
    test_spatial_attention() 