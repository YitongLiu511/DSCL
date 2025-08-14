import torch
from torch import nn

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

    def forward(self, x):
        # temporal part
        x = self.embed(x)
        # x = self.auto_corr(x, x)[0]
        x = self.norm1(self.dropout(x + self.temporal_attn(x, x)[0]))

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
        # output = output_dy + output_st
        output = output.swapaxes(0, 1)
        score_dy = score_dy.mean(0)

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

    def forward(self, x):
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

        return (
            self.revin(output, 'denorm'),
            # output,
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
        # 开启 patch 内时间戳注意力
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
            pe='zeros',
            learn_pe=True,
            use_inpatch_attn=True,
            inpatch_pe='sincos',
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
        # 移除self.last_linear，因为线性层已经移到estimator中了

    def forward(self, x, return_recon=True):
        # x : (N, T, d)
        patch_x = self.patch(x)  # (N, NP, VAR, PL)
        x = self.revin(patch_x.transpose(2, 3), 'norm').transpose(2, 3)
        z = self.patch_tsfm(x)  # (N, VAR, NP, D)
        z = z.permute(1, 2, 0, 3)  # (VAR, NP, N, D)
        z = z.reshape(-1, z.shape[2], z.shape[3])  # (VAR * NP, N, D)

        # 在这里添加软聚类
        # 重塑维度以便于聚类
        region_embeddings = z.transpose(0, 1)  # (N, VAR*NP, D)
        
        # 应用软聚类并获取损失
        clustered_embeddings, cluster_loss = self.soft_cluster(region_embeddings)
        
        # 恢复原始维度
        z = clustered_embeddings.transpose(0, 1)  # (VAR*NP, N, D)

        dy_z, attn = self.spatial_tsfm(z)  # dy_z: (VAR * NP, N, D); attn: (VAR*NP, H, N, N)
        st_z, graph = self.da_gcn(z)  # (VAR * NP, N, D)
        
        # 🔧 修复：确保返回正确的注意力分数形状，回归Origin设计
        if isinstance(attn, torch.Tensor) and attn.dim() == 4:
            # attn: (VAR*NP, H, N, N) -> score_dy: (N, N)
            score_dy = attn.mean(dim=(0, 1))  # 只对tokens和heads维度平均
        else:
            score_dy = attn
            
        if isinstance(graph, torch.Tensor) and graph.dim() > 2:
            # graph: (VAR*NP, N, N) -> score_st: (N, N)
            score_st = graph.mean(dim=0)  # 只对tokens维度平均
        else:
            score_st = graph
        
        if not return_recon:
            return score_dy, score_st

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
        z_flat = z.reshape(z.shape[0], -1, z.shape[-1])  # (N, NP*PL, VAR) - 只是reshape
        return (patch_x.transpose(2, 3), z), (score_dy, score_st), z_flat, cluster_loss


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
    ) -> None:
        super().__init__(seq_len, patch_len, stride, d_in, d_model, n_heads,
                         dist_mat, n_gcn, temporal_half, spatial_half,
                         static_only, dynamic_only)
        self.mask_ratio = mask_ratio
        self.random_mask = RandomMasking(self.mask_ratio)

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
        return (patch_x.transpose(2, 3), z), (attn, graph)


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
        n_prototypes: int = 10,  # 添加原型数量参数
        tau: float = 0.5,        # 添加温度参数
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
            pe='zeros',
            learn_pe=True,
            use_inpatch_attn=True,
            inpatch_pe='sincos',
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
            bias=True,
        )
        self.proj_dy = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, patch_len),
        )
        self.proj_st = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, patch_len),
        )
        
        # 添加软聚类层
        from .module import SoftClusterLayer
        self.soft_cluster = SoftClusterLayer(
            c_in=d_model,           # 使用模型的隐藏维度
            nmb_prototype=n_prototypes,
            tau=tau
        )

        # 🆕 新增：独立的片间注意力分支（用于DCdetector的prior attention）
        # 这个分支专门学习patch与patch之间的关系
        # 参数共享：片间注意力与patch内注意力共享同一Transformer实例
        # 直接复用 PatchEncoder 中的 inpatch_encoder（返回注意力）
        if hasattr(self.patch_tsfm, 'inpatch_encoder') and self.patch_tsfm.inpatch_encoder is not None:
            self.patchwise_attention = self.patch_tsfm.inpatch_encoder
        else:
            # 回退：若未构建inpatch_encoder，则创建一个同构Transformer供片间注意力使用
            self.patchwise_attention = TemporalTransformer(
                d_model,
                d_model // n_heads,
                d_model // n_heads,
                n_heads,
                d_model,
                0.1,
                False,
                True,
            )

    def get_attention_weights(self, x):
        """
        获取注意力权重用于DCdetector损失计算
        返回: (series_attention, prior_attention, series_inpatch, prior_inpatch)
        """
        # 获取patch数据（避免在 PatchEncoder 内部做 num_patch×num_patch 的注意力，降低显存峰值）
        patch_x = self.patch(x)  # (N, NP, VAR, PL)
        x_norm = self.revin(patch_x.transpose(2, 3), 'norm').transpose(2, 3)

        # 1) in-patch 注意力（GPU）：按 DC 风格返回最后一次的 (H, PL, PL)
        xb = x_norm.transpose(1, 2)  # [N, VAR, NP, PL]
        bs_, n_vars_, num_patch_, patch_len_ = xb.shape
        # 复用 PatchEncoder 缓存的最近一次 patch 表示（保持在CPU上使用），避免重复做 in-patch 编码
        x_patch = None
        inpatch_attn = None
        if hasattr(self.patch_tsfm, 'last_x_patch') and isinstance(self.patch_tsfm.last_x_patch, torch.Tensor):
            # 直接使用 CPU 缓存，不搬到 GPU
            x_patch = self.patch_tsfm.last_x_patch
            inpatch_attn = getattr(self.patch_tsfm, 'last_inpatch_attn', None)
        if x_patch is None or not isinstance(x_patch, torch.Tensor):
            tokens = xb.reshape(bs_ * n_vars_ * num_patch_, patch_len_, 1)
            h = self.patch_tsfm.inpatch_value_embed(tokens)
            h = h + self.patch_tsfm.inpatch_W_pos
            h = self.patch_tsfm.dropout(h)
            inpatch_out = self.patch_tsfm.inpatch_encoder(h)
            if isinstance(inpatch_out, tuple):
                h, inpatch_attn = inpatch_out  # [B', PL, d_model], [B', H, PL, PL]
            else:
                inpatch_attn = None
            # 每 patch 池化得到 patch 表示
            h_mean = h.mean(dim=1)  # [B', d_model]
            x_patch = h_mean.reshape(bs_, n_vars_, num_patch_, self.d_model)  # (N, VAR, NP, D)

        # 2) patch 间 prior（CPU低内存）：用余弦相似度近似并行 softmax
        with torch.no_grad():
            # 保证在CPU上计算先验
            x_patch_cpu = x_patch if x_patch.device.type == 'cpu' else x_patch.detach().cpu()
            N, VAR, NP, D = x_patch_cpu.shape
            prior_list = []
            for n in range(N):
                pv = []
                for v in range(VAR):
                    Y = x_patch_cpu[n, v]  # [NP, D]
                    Yn = Y / (Y.norm(dim=1, keepdim=True) + 1e-8)
                    S = Yn @ Yn.T  # [NP, NP]
                    P = torch.softmax(S, dim=-1)
                    pv.append(P.unsqueeze(0))  # [1, NP, NP]
                prior_list.append(torch.stack(pv, dim=0))  # [VAR, NP, NP]
            prior_inpatch = torch.stack(prior_list, dim=0)  # [N, VAR, NP, NP]
            # 回到模型设备
            prior_inpatch = prior_inpatch.to(x.device)

        # 为避免显存峰值，跳过空间注意力的显式计算，仅返回时间分支所需权重
        series_spatial = []
        prior_spatial = []

        # 额外：series_inpatch (N, VAR, NP, H, PL, PL)
        series_inpatch = None
        if inpatch_attn is not None:
            series_inpatch = inpatch_attn.reshape(bs_, n_vars_, num_patch_, inpatch_attn.shape[1], patch_len_, patch_len_)

        # 🆕 返回四元组：空间注意力(空) + 片间注意力 + 时间注意力
        # 1. series_spatial: 空间注意力 (VAR, NP, N, N)
        # 2. prior_spatial: 图注意力 (VAR, NP, N, N) 
        # 3. series_inpatch: patch内时间注意力 (N, VAR, NP, H, PL, PL)
        # 4. prior_inpatch: 片间注意力 (N, VAR, NP, NP)
        return [series_spatial], [prior_spatial], series_inpatch, prior_inpatch

    def get_representations(self, x):
        """
        获取中间表示用于备用DCdetector损失计算
        返回: 中间特征表示
        """
        # 获取patch数据
        patch_x = self.patch(x)  # (N, NP, VAR, PL)
        x_norm = self.revin(patch_x.transpose(2, 3), 'norm').transpose(2, 3)
        z = self.patch_tsfm(x_norm)  # (N, VAR, NP, D)
        z = z.permute(1, 2, 0, 3)  # (VAR, NP, N, D)
        z = z.reshape(-1, z.shape[2], z.shape[3])  # (VAR * NP, N, D)

        # 获取动态和静态表示
        dy_z, _ = self.spatial_tsfm(z)  # (VAR * NP, N, D)
        st_z, _ = self.da_gcn(z)  # (VAR * NP, N, D)
        
        # 融合表示
        fused_repr = torch.cat([dy_z, st_z], dim=-1)  # (VAR * NP, N, 2*D)
        
        return fused_repr

    def forward(self, x, return_recon=True):
        # x : (N, T, d)
        patch_x = self.patch(x)  # (N, NP, VAR, PL)
        x = self.revin(patch_x.transpose(2, 3), 'norm').transpose(2, 3)
        z = self.patch_tsfm(x)  # (N, VAR, NP, D)
        z = z.permute(1, 2, 0, 3)  # (VAR, NP, N, D)
        z = z.reshape(-1, z.shape[2], z.shape[3])  # (VAR * NP, N, D)

        # 在这里添加软聚类
        # 重塑维度以便于聚类
        region_embeddings = z.transpose(0, 1)  # (N, VAR*NP, D)
        
        # 应用软聚类并获取损失
        clustered_embeddings, cluster_loss = self.soft_cluster(region_embeddings)
        
        # 恢复原始维度
        z = clustered_embeddings.transpose(0, 1)  # (VAR*NP, N, D)

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
        z_flat = z.reshape(z.shape[0], -1, z.shape[-1])  # (N, NP*PL, VAR) - 只是reshape
        return (patch_x.transpose(2, 3), z), (attn, graph), z_flat, cluster_loss
