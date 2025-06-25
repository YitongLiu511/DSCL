import numpy as np

# 加载原始分数
scores = np.load('data/spatial_attention_scores_train.npy')  # (2016, 263, 2, 2, 2)

# 尝试直接reshape为(2016, 2, 263, 263)
try:
    scores_fixed = scores.reshape(2016, 2, 263, 263)
    print('reshape成功:', scores_fixed.shape)
except Exception as e:
    print('reshape失败，尝试transpose+reshape:', str(e))
    # 如果reshape失败，先transpose再reshape
    scores_trans = np.transpose(scores, (0, 2, 1, 3, 4))  # (2016, 2, 263, 2, 2)
    scores_fixed = scores_trans.reshape(2016, 2, 263, 263)
    print('transpose+reshape成功:', scores_fixed.shape)

# 保存为新文件
np.save('data/spatial_attention_scores_train_fixed.npy', scores_fixed)
print('已保存为 data/spatial_attention_scores_train_fixed.npy') 