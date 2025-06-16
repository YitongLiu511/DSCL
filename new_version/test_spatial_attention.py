import torch
import numpy as np
from spatial_attention import SpatialAttentionBlock

def test_spatial_attention():
    print("开始测试空间注意力处理...")
    
    # 加载时间注意力处理后的数据
    try:
        data = np.load('data/temporal_attention_processed_train.npy')
        print(f"成功加载数据，形状: {data.shape}")
    except Exception as e:
        print(f"加载数据失败: {str(e)}")
        return
    
    # 加载邻接矩阵
    try:
        adj = np.load('data/processed/adj.npy')
        adj_list = [torch.FloatTensor(adj)]  # 单层GCN，只需要一个邻接矩阵
        print(f"成功加载邻接矩阵，形状: {adj.shape}")
    except Exception as e:
        print(f"加载邻接矩阵失败: {str(e)}")
        return
    
    try:
        # 处理数据
        print("开始处理数据...")
        
        # 创建处理器
        processor = SpatialAttentionBlock(
            in_channels=1,
            out_channels=2,
            n_views=1,
            n_heads=2,
            dropout=0.1,
            gcn_type='gcn'
        )
        
        # 转换数据格式
        data = torch.FloatTensor(data)  # [263, 2016, 2]
        data = data.permute(1, 0, 2)  # [2016, 263, 2]
        
        # 处理每个时间步
        outputs = []
        for t in range(data.size(0)):
            x_t = data[t].unsqueeze(-1)  # [263, 2, 1]
            x_s = processor(x_t, adj_list)  # [263, 2, 2]
            x_s = x_s.mean(dim=-1)  # [263, 2]
            outputs.append(x_s)
        
        # 堆叠所有时间步的输出
        output = torch.stack(outputs, dim=0)  # [2016, 263, 2]
        output = output.permute(1, 0, 2)  # [263, 2016, 2]
        
        print(f"处理完成，输出形状: {output.shape}")
        
        # 保存结果
        np.save('data/spatial_attention_processed_train.npy', output.detach().numpy())
        print("结果已保存到 data/spatial_attention_processed_train.npy")
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        return
    
    print("测试完成！")

if __name__ == "__main__":
    test_spatial_attention() 