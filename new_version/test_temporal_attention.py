import numpy as np
import torch
from temporal_attention import unmasked_data, TemporalAttentionProcessor

def test_temporal_attention():
    print("开始测试时域掩码+位置编码+attention流程...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = TemporalAttentionProcessor(device=device).to(device)
    batch_size = 15
    total_nodes = len(unmasked_data)
    all_outputs = []
    all_lengths = []
    for start in range(0, total_nodes, batch_size):
        batch = []
        batch_lens = []
        for arr in unmasked_data[start:start+batch_size]:
            if arr.shape[0] > 0:
                batch.append(torch.FloatTensor(arr))
                batch_lens.append(arr.shape[0])
        if not batch:
            continue
        max_len = max(batch_lens)
        padded = torch.zeros(len(batch), max_len, batch[0].shape[1], device=device)
        for i, arr in enumerate(batch):
            padded[i, :arr.shape[0], :] = arr.to(device)
        output, attention_weights = processor(padded)
        all_outputs.append(output.detach().cpu().numpy())
        all_lengths.extend(batch_lens)
        print(f"节点{start}~{start+len(batch)-1} 输出shape: {output.shape}")
        torch.cuda.empty_cache()  # 释放显存
    final_features = np.concatenate(all_outputs, axis=0)  # [节点总数, max_len, 特征数]
    np.save('temporal_attention_features.npy', final_features)
    print(f"最终特征shape: {final_features.shape}")
    print("测试完成！")

if __name__ == '__main__':
    test_temporal_attention() 