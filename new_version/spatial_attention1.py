import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SpatialSelfAttentionBlock(nn.Module):
    """
    空间自注意力模块，用于计算节点间的依赖关系。
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(SpatialSelfAttentionBlock, self).__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [B, N, D_model]
        B, N, _ = x.shape
        
        q = self.q_linear(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_linear(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_linear(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, N, -1)
        
        output = self.out_proj(context)
        return output, attn_weights

def process_dynamic_stream_data(temporal_data, d_model=64, n_heads=8, batch_size=128):
    """
    高效、内存安全地处理动态空间流数据。
    核心思想：将时间步作为批次，分批送入GPU进行空间注意力计算。
    """
    print("=== 开始动态流处理 (高效分批版) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if isinstance(temporal_data, np.ndarray):
        temporal_data = torch.from_numpy(temporal_data).float()

    # 1. 修正维度：将 [N, T, C] -> [T, N, C]
    x = temporal_data.permute(1, 0, 2)
    T, N, C = x.shape
    print(f"输入数据已修正为 [时间, 节点, 特征] 格式: [{T}, {N}, {C}]")

    # 2. 初始化模型和投影层
    input_projection = nn.Linear(C, d_model).to(device)
    spatial_model = SpatialSelfAttentionBlock(d_model=d_model, n_heads=n_heads).to(device)
    spatial_model.eval()

    all_outputs = []
    all_attention_weights = []

    # 3. 分批处理
    print(f"开始分批处理，每批大小: {batch_size}")
    num_batches = math.ceil(T / batch_size)
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, T)
        
        batch_x = x[start_idx:end_idx].to(device)
        
        with torch.no_grad():
            batch_x_projected = input_projection(batch_x)
            batch_output, batch_attn_weights = spatial_model(batch_x_projected)
        
        all_outputs.append(batch_output.cpu())
        all_attention_weights.append(batch_attn_weights.cpu())
        print(f"  - 已处理批次 {i+1}/{num_batches}")

    # 4. 拼接结果
    attention_output = torch.cat(all_outputs, dim=0)
    attention_weights = torch.cat(all_attention_weights, dim=0)
    print("所有批次处理完毕，已拼接结果。")

    # 5. 计算最终分数
    dynamic_scores = attention_weights.mean(0)

    print(f"\n最终特征输出形状: {attention_output.shape}")       # [T, N, d_model] e.g., [2016, 263, 64]
    print(f"最终动态流分数形状: {dynamic_scores.shape}") # [n_heads, N, N] e.g., [8, 263, 263]
    print("=== 动态流处理完成 ===")
    
    return dynamic_scores, attention_output

# 用法示例
if __name__ == '__main__':
    # 模拟时间注意力输出
    temporal_output = torch.randn(263, 2016, 2)  # [num_nodes, num_timesteps, num_features]
    
    # 测试动态流处理
    dynamic_scores, attention_output = process_dynamic_stream_data(temporal_output)
    print('动态流分数形状:', dynamic_scores.shape)
    print('注意力权重形状:', attention_output.shape) 