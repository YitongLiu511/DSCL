import torch
from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score

def kl_loss(p, q):
    res = p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))
    return res  # 不取平均，保持维度信息

def recall_k(y_true, y_score, k):
    """自定义recall@k函数，计算y_true中为1的样本在y_score排序后前k个样本中出现的比例。"""
    if k <= 0:
        raise ValueError("k必须为正整数")
    # 将y_true、y_score转为numpy数组，并确保一维
    y_true = np.asarray(y_true).flatten()
    y_score = np.asarray(y_score).flatten()
    if len(y_true) != len(y_score):
        raise ValueError("y_true与y_score长度不一致")
    # 按y_score降序排序，取前k个索引
    sort_index = np.argsort(y_score)[::-1][:k]
    # 计算前k个样本中，y_true为1的样本数，并除以y_true中1的总数（若总数为0则返回0）
    n_anomaly = np.sum(y_true == 1)
    if n_anomaly == 0:
        return 0.0
    return np.sum(y_true[sort_index] == 1) / n_anomaly

from .embed import TemporalEmbedding
from .module import (
    SingleGCN,
    MultipleGCN,
    MultiheadAttention,
)
from .revin import RevIN


class STAnomalyFormer_v1(nn.Module):

    def __init__(
        self,
        dist_mat,
        d_in: int,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        n_gcn: int,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.embed = TemporalEmbedding(d_in, d_model)
        self.temporal_attn = MultiheadAttention(
            d_model,
            dim_k,
            dim_v,
            n_heads,
            batch_size=batch_size,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

        self.spatial_attn = MultiheadAttention(d_model, dim_k, dim_v, n_heads)
        self.norm2 = nn.LayerNorm(d_model)

        self.gcn = SingleGCN(
            d_model,
            d_model,
            dist_mat=dist_mat,
            n_layers=n_gcn,
        )

        self.proj1 = nn.Linear(d_model, d_in)
        self.proj2 = nn.Linear(d_model, d_in)

    def forward(self, x, return_dict=False):
        # temporal part
        x = self.embed(x)
        # x = self.auto_corr(x, x)[0]
        x, attn1 = self.temporal_attn(x, x)
        x = self.norm1(self.dropout(x))

        x = x.swapaxes(0, 1)
        # dynamic spatial part
        output_dy, score_dy = self.spatial_attn(x, x)
        output_dy = self.norm2(self.dropout(x + output_dy))

        # static spatial part
        output_st, score_st = self.gcn(x)

        # fusion
        output_dy, output_st = self.proj1(output_dy), self.proj2(output_st)
        g = torch.sigmoid(output_dy + output_st)
        output = g * output_dy + (1 - g) * output_st
        output = output.swapaxes(0, 1)
        score_dy = score_dy.mean(0)
        score_st = score_st.mean(0)

        # 计算区域异常分数（空间维度）
        region_scores = score_st.mean(dim=1)  # 对每个区域的所有时间点取平均
        # 计算时间异常分数（时间维度）
        time_scores = score_dy.mean(dim=0)  # 对每个时间点的所有区域取平均

        if return_dict:
            return {
                'reconstruction': (x, output),
                'attention': (attn1.mean(0), score_st),
                'region_scores': region_scores,
                'time_scores': time_scores,
                'score_dy': score_dy,
                'score_st': score_st
            }
        else:
            return output, score_dy, score_st


class STAnomalyFormer_v2(STAnomalyFormer_v1):

    def __init__(
        self,
        dist_mat,
        d_in: int,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        n_gcn: int,
        batch_size: int,
    ) -> None:
        super().__init__(dist_mat, d_in, d_model, dim_k, dim_v, n_heads, n_gcn,
                         batch_size)
        self.embed_gcn = SingleGCN(
            d_in,
            d_model,
            dist_mat,
            n_layers=1,
            activation=lambda x: x,
        )
        self.revin = RevIN(d_in)

    def forward(self, x, return_dict=False):
        x = self.revin(x, 'norm')
        x1 = self.embed(x)
        branch1 = x1
        branch2 = x1 + self.embed_gcn(x.swapaxes(0, 1))[0].swapaxes(0, 1)

        branch1_, attn1 = self.temporal_attn(branch1, branch1)
        branch2_, attn2 = self.temporal_attn(branch2, branch2)

        branch1_ = self.norm1(self.dropout(branch1 + branch1_))
        branch2_ = self.norm1(self.dropout(branch2 + branch2_))

        branch1_ = branch1_.swapaxes(0, 1)
        branch2_ = branch2_.swapaxes(0, 1)

        output_dy, score_dy = self.spatial_attn(branch1_, branch1_)
        output_dy = self.norm2(self.dropout(branch1_ + output_dy))

        output_st, score_st = self.gcn(branch2_)

        output_dy, output_st = self.proj1(output_dy), self.proj2(output_st)
        g = torch.sigmoid(output_dy + output_st)
        output = g * output_dy + (1 - g) * output_st
        output = output.swapaxes(0, 1)
        score_dy = score_dy.mean(0)

        if return_dict:
            # 计算区域异常分数（空间维度）
            region_scores = score_st.mean(dim=1)  # 对每个区域的所有时间点取平均
            
            # 计算时间异常分数（时间维度）
            time_scores = score_dy.mean(dim=0)  # 对每个时间点的所有区域取平均
            
            return {
                'reconstruction': self.revin(output, 'denorm'),
                'attention': (attn1.mean(0), attn2.mean(0)),
                'region_scores': region_scores,
                'time_scores': time_scores,
                'score_dy': score_dy,
                'score_st': score_st
            }
        else:
            return (
                self.revin(output, 'denorm'),
                attn1.mean(0),
                attn2.mean(0),
                score_dy,
                score_st,
            )


from .patch import Patch, PatchEncoder, RandomMasking
from .tsfm import TemporalTransformer


class STPatchFormer(nn.Module):

    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        stride: int,
        d_in: int,
        d_model: int,
        n_heads: int,
        dist_mat,
        n_gcn: int = 3,
        temporal_half: bool = False,
        spatial_half: bool = False,
        static_only: bool = False,
        dynamic_only: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.temporal_half = temporal_half
        self.spatial_half = spatial_half
        self.n_gcn = n_gcn
        self.static_only = static_only
        self.dynamic_only = dynamic_only

        self.revin = RevIN(d_in)
        self.patch = Patch(seq_len, patch_len, stride)
        self.patch_tsfm = PatchEncoder(
            d_in,
            self.patch.num_patch,
            patch_len,
            d_model,
            n_heads,
            256,
            False,
            0.1,
            0.1,
            temporal_half,
        )
        self.spatial_tsfm = TemporalTransformer(
            d_model,
            d_model // n_heads,
            d_model // n_heads,
            n_heads,
            d_model,
            0.1,
            spatial_half,
            True,
        )
        self.da_gcn = SingleGCN(
            d_model,
            d_model,
            dist_mat,
            n_layers=n_gcn,
        )
        self.proj_dy = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, patch_len),
        )
        self.proj_st = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, patch_len),
        )

    def forward(self, x, return_recon=True):
        # x : (N, T, d)
        patch_x = self.patch(x)  # (N, NP, VAR, PL)
        x = self.revin(patch_x.transpose(2, 3), 'norm').transpose(2, 3)
        z = self.patch_tsfm(x)  # (N, VAR, NP, D)
        z = z.permute(1, 2, 0, 3)  # (VAR, NP, N, D)
        z = z.reshape(-1, z.shape[2], z.shape[3])  # (VAR * NP, N, D)

        dy_z, attn = self.spatial_tsfm(z)  # (VAR * NP, N, D)
        st_z, graph = self.da_gcn(z)  # (VAR * NP, N, D)
        if not return_recon:
            return attn, graph

        if self.dynamic_only:
            z = self.proj_dy(dy_z)  # (VAR * NP, N, PL)
        elif self.static_only:
            z = self.proj_st(st_z)
        else:
            dy_out = self.proj_dy(dy_z)  # (VAR * NP, N, PL)
            st_out = self.proj_st(st_z)  # (VAR * NP, N, PL)
            g = torch.sigmoid(dy_out + st_out)
            z = g * dy_out + (1 - g) * st_out
        z = z.reshape(
            self.d_in,
            -1,
            x.shape[0],
            self.patch_len,  # (VAR, NP, N, PL)
        ).permute(2, 1, 3, 0)  # (N, NP, PL, VAR)
        z = self.revin(z, 'denorm')
        return (patch_x.transpose(2, 3), z), (attn, graph)


class STPatchMaskFormer(STPatchFormer):

    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        stride: int,
        d_in: int,
        d_model: int,
        n_heads: int,
        dist_mat,
        n_gcn: int = 3,
        temporal_half: bool = False,
        spatial_half: bool = False,
        static_only: bool = False,
        dynamic_only: bool = False,
        mask_ratio: float = 0.4,
        temperature: float = 50.0,
        anormly_ratio: float = 0.1,
    ) -> None:
        super().__init__(seq_len, patch_len, stride, d_in, d_model, n_heads,
                         dist_mat, n_gcn, temporal_half, spatial_half,
                         static_only, dynamic_only)
        self.mask_ratio = mask_ratio
        self.random_mask = RandomMasking(self.mask_ratio)
        self.temperature = temperature
        self.anormly_ratio = anormly_ratio

    def forward(self, x):
        patch_x = self.patch(x)  # (N, NP, VAR, PL)
        x = self.revin(patch_x.transpose(2, 3), 'norm').transpose(2, 3)
        mask_x = self.random_mask(x)
        z = self.patch_tsfm(mask_x)  # (N, VAR, NP, D)
        z = z.permute(1, 2, 0, 3)  # (VAR, NP, N, D)
        z = z.reshape(-1, z.shape[2], z.shape[3])  # (VAR * NP, N, D)

        dy_z, attn = self.spatial_tsfm(z)  # (VAR * NP, N, D)
        st_z, graph = self.da_gcn(z)  # (VAR * NP, N, D)

        dy_out = self.proj_dy(dy_z)  # (VAR * NP, N, PL)
        st_out = self.proj_st(st_z)  # (VAR * NP, N, PL)
        g = torch.sigmoid(dy_out + st_out)
        z = g * dy_out + (1 - g) * st_out
        z = z.reshape(
            self.d_in,
            -1,
            x.shape[0],
            self.patch_len,  # (VAR, NP, N, PL)
        ).permute(2, 1, 3, 0)  # (N, NP, PL, VAR)
        z = self.revin(z, 'denorm')

        # Calculate anomaly scores
        adv_loss = 0.0
        con_loss = 0.0
        
        # 计算KL散度损失
        for i in range(attn.shape[0]):
            if i == 0:
                adv_loss = my_kl_loss(attn[i], 
                    (graph[i] / torch.unsqueeze(torch.sum(graph[i], dim=-1), dim=-1)).detach()) * self.temperature
                con_loss = my_kl_loss(
                    (graph[i] / torch.unsqueeze(torch.sum(graph[i], dim=-1), dim=-1)),
                    attn[i].detach()) * self.temperature
            else:
                adv_loss += my_kl_loss(attn[i], 
                    (graph[i] / torch.unsqueeze(torch.sum(graph[i], dim=-1), dim=-1)).detach()) * self.temperature
                con_loss += my_kl_loss(
                    (graph[i] / torch.unsqueeze(torch.sum(graph[i], dim=-1), dim=-1)),
                    attn[i].detach()) * self.temperature
        
        # 计算异常分数
        metric = torch.softmax((adv_loss + con_loss), dim=-1)
        anomaly_scores = metric.detach().cpu().numpy()
        
        # 计算阈值
        thresh = np.percentile(anomaly_scores, 100 - self.anormly_ratio)
        
        # 检测异常
        pred = (anomaly_scores > thresh).astype(int)
        
        # 获取异常区域和时间戳
        anomaly_regions = []
        anomaly_timestamps = []
        
        anomaly_state = False
        for i in range(len(pred)):
            if pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                start_idx = i
            elif pred[i] == 0 and anomaly_state:
                anomaly_state = False
                end_idx = i - 1
                anomaly_regions.append((start_idx, end_idx))
                # 转换索引为时间戳
                start_time = start_idx * self.stride
                end_time = end_idx * self.stride + self.patch_len
                anomaly_timestamps.append((start_time, end_time))
        
        if anomaly_state:
            end_idx = len(pred) - 1
            anomaly_regions.append((start_idx, end_idx))
            start_time = start_idx * self.stride
            end_time = end_idx * self.stride + self.patch_len
            anomaly_timestamps.append((start_time, end_time))

        return {
            'reconstruction': (patch_x.transpose(2, 3), z),
            'attention': (attn, graph),
            'anomaly_scores': anomaly_scores,
            'anomaly_regions': anomaly_regions,
            'anomaly_timestamps': anomaly_timestamps,
            'threshold': thresh
        }


class STPatch_MGCNFormer(nn.Module):

    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        stride: int,
        d_in: int,
        d_model: int,
        n_heads: int,
        dist_mats,
        n_gcn: int = 3,
        temporal_half: bool = False,
        spatial_half: bool = False,
        static_only: bool = False,
        dynamic_only: bool = False,
        temperature: float = 50.0,
        anormly_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.temporal_half = temporal_half
        self.spatial_half = spatial_half
        self.n_gcn = n_gcn
        self.static_only = static_only
        self.dynamic_only = dynamic_only
        self.temperature = temperature
        self.anormly_ratio = anormly_ratio

        self.revin = RevIN(d_in)
        self.patch = Patch(seq_len, patch_len, stride)
        self.patch_tsfm = PatchEncoder(
            d_in,
            self.patch.num_patch,
            patch_len,
            d_model,
            n_heads,
            256,
            False,
            0.1,
            0.1,
            temporal_half,
        )
        self.spatial_tsfm = TemporalTransformer(
            d_model,
            d_model // n_heads,
            d_model // n_heads,
            n_heads,
            d_model,
            0.1,
            spatial_half,
            True,
        )
        self.da_gcn = MultipleGCN(
            d_model,
            d_model,
            dist_mats,
            n_layers=n_gcn,
        )
        self.proj_dy = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, patch_len),
        )
        self.proj_st = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, patch_len),
        )

    def forward(self, x, return_dict=False):
        # x : (N, T, d)
        patch_x = self.patch(x)  # (N, NP, VAR, PL)
        x = self.revin(patch_x.transpose(2, 3), 'norm').transpose(2, 3)
        z = self.patch_tsfm(x)  # (N, VAR, NP, D)
        z = z.permute(1, 2, 0, 3)  # (VAR, NP, N, D)
        z = z.reshape(-1, z.shape[2], z.shape[3])  # (VAR * NP, N, D)

        dy_z, attn = self.spatial_tsfm(z)  # (VAR * NP, N, D)
        st_z, graph = self.da_gcn(z)  # (VAR * NP, N, D)

        if self.dynamic_only:
            z = self.proj_dy(dy_z)  # (VAR * NP, N, PL)
        elif self.static_only:
            z = self.proj_st(st_z)
        else:
            dy_out = self.proj_dy(dy_z)  # (VAR * NP, N, PL)
            st_out = self.proj_st(st_z)  # (VAR * NP, N, PL)
            g = torch.sigmoid(dy_out + st_out)
            z = g * dy_out + (1 - g) * st_out

        z = z.reshape(
            self.d_in,
            -1,
            x.shape[0],
            self.patch_len,  # (VAR, NP, N, PL)
        ).permute(2, 1, 3, 0)  # (N, NP, PL, VAR)
        z = self.revin(z, 'denorm')

        if return_dict:
            # 计算KL散度损失
            adv_loss = 0.0
            con_loss = 0.0
            
            # 确保graph和attn的维度匹配
            for i in range(min(len(attn), len(graph))):
                if i == 0:
                    adv_loss = kl_loss(attn[i], 
                        (graph[i] / torch.unsqueeze(torch.sum(graph[i], dim=-1), dim=-1)).detach()) * self.temperature
                    con_loss = kl_loss(
                        (graph[i] / torch.unsqueeze(torch.sum(graph[i], dim=-1), dim=-1)),
                        attn[i].detach()) * self.temperature
                else:
                    adv_loss += kl_loss(attn[i], 
                        (graph[i] / torch.unsqueeze(torch.sum(graph[i], dim=-1), dim=-1)).detach()) * self.temperature
                    con_loss += kl_loss(
                        (graph[i] / torch.unsqueeze(torch.sum(graph[i], dim=-1), dim=-1)),
                        attn[i].detach()) * self.temperature
            
            # 计算动态和静态异常分数
            dy_scores = torch.mean(dy_out, dim=0)  # (N, T)
            st_scores = torch.mean(st_out, dim=0)  # (N, T)
            
            # 计算区域异常分数（空间维度）- 对每个区域的所有时间点取平均
            region_scores = torch.mean(st_scores, dim=1)  # (N,)
            
            # 计算时间异常分数（时间维度）- 对每个时间点的所有区域取平均
            time_scores = torch.mean(dy_scores, dim=0)  # (T,)
            
            # 计算综合异常分数
            anomaly_scores = (dy_scores + st_scores) / 2
            
            # 计算阈值（使用numpy进行百分位数计算）
            region_thresh = torch.tensor(np.percentile(region_scores.detach().cpu().numpy(), 100 - self.anormly_ratio)).to(region_scores.device)
            time_thresh = torch.tensor(np.percentile(time_scores.detach().cpu().numpy(), 100 - self.anormly_ratio)).to(time_scores.device)
            
            # 检测区域异常
            region_pred = (region_scores > region_thresh).int()
            
            # 检测时间异常
            time_pred = (time_scores > time_thresh).int()
            
            # 获取异常区域
            anomaly_regions = torch.where(region_pred == 1)[0].cpu().numpy().tolist()
            
            # 获取异常时间戳
            anomaly_timestamps = []
            time_indices = torch.where(time_pred == 1)[0].cpu().numpy()
            
            if len(time_indices) > 0:
                # 合并连续的异常时间戳
                start_idx = time_indices[0]
                prev_idx = start_idx
                
                for i in range(1, len(time_indices)):
                    if time_indices[i] != prev_idx + 1:
                        # 添加当前连续段
                        start_time = start_idx * self.stride
                        end_time = prev_idx * self.stride + self.patch_len
                        anomaly_timestamps.append((start_time, end_time))
                        # 开始新的连续段
                        start_idx = time_indices[i]
                    prev_idx = time_indices[i]
                
                # 添加最后一个连续段
                start_time = start_idx * self.stride
                end_time = prev_idx * self.stride + self.patch_len
                anomaly_timestamps.append((start_time, end_time))

            return {
                'reconstruction': (patch_x.transpose(2, 3), z),
                'attention': (attn, graph),
                'anomaly_scores': anomaly_scores,
                'region_scores': region_scores,  # 保持tensor格式
                'time_scores': time_scores,      # 保持tensor格式
                'region_pred': region_pred,
                'time_pred': time_pred,
                'anomaly_regions': anomaly_regions,
                'anomaly_timestamps': anomaly_timestamps,
                'region_threshold': region_thresh,
                'time_threshold': time_thresh
            }
        else:
            return (patch_x.transpose(2, 3), z), (attn, graph)