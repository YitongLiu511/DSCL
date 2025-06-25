import numpy as np

scores = np.load('data/spatial_attention_scores_train.npy')
print('总shape:', scores.shape)
print('第一帧shape:', scores[0].shape)
print('第一帧第一个元素shape:', scores[0][0].shape)
print('第一帧第一个元素第一个元素shape:', scores[0][0][0].shape) 