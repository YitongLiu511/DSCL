import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Union
from torch_geometric.nn import GCNConv, Sequential
from .embed import TemporalEmbedding


class MultiheadAttention(nn.Module):
    '''For temporal input'''

    def __init__(
        self,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        factor=5,
        batch_size=-1,
    ) -> None:
        super().__init__()
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.n_heads = n_heads
        self.batch_size = batch_size

        self.q = nn.Linear(d_model, dim_k * n_heads)
        self.k = nn.Linear(d_model, dim_k * n_heads)
        self.v = nn.Linear(d_model, dim_v * n_heads)
        self.o = nn.Linear(dim_v * n_heads, d_model)
        self.norm_fact = 1 / math.sqrt(d_model)

    def attention(self, Q, K, V):
        B, L = Q.shape[:2]
        scores = torch.einsum("blhe,bshe->bhls", Q, K) * self.norm_fact
        scores = scores.softmax(dim=-1)
        output = torch.einsum("bhls,bshd->blhd", scores, V).reshape(B, L, -1)
        return output, scores

    def forward(self, x, y):
        '''x : (B, L, D)'''
        B, L, _ = x.shape
        Q = self.q(x).reshape(B, L, self.n_heads, -1)  # (N, B, L, K)
        K = self.k(x).reshape(B, L, self.n_heads, -1)  # (N, B, L, K)
        V = self.v(y).reshape(B, L, self.n_heads, -1)  # (N, B, L, K)
        output, scores = self.attention(Q, K, V)
        return self.o(output), scores


class _SingleLayerTemporalTSFM(nn.Module):
    '''无位置编码'''

    def __init__(
        self,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        dim_fc: int = 128,
        dropout=0.1,
        half: bool = True,
    ) -> None:
        super().__init__()
        self.half_ = half
        self.d_model = d_model
        self.attn = MultiheadAttention(d_model, dim_k, dim_v, n_heads)
        self.conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=dim_fc,
            kernel_size=1,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if not half:
            self.conv2 = nn.Conv1d(
                in_channels=dim_fc,
                out_channels=d_model,
                kernel_size=1,
            )
            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        y = x = self.norm1(self.dropout(x + self.attn(x, x)[0]))
        y = self.dropout(torch.relu(self.conv1(y.transpose(-1, 1))))
        if self.half_:
            return y
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class TemporalTSFM(nn.Module):

    def __init__(
        self,
        d_in: int,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        dim_fc: int = 128,
        n_layers: int = 1,
        half: bool = False,
        projection: bool = True,
    ) -> None:
        super().__init__()
        self.embed = TemporalEmbedding(d_in, d_model)
        self.model_list = nn.Sequential(*[
            _SingleLayerTemporalTSFM(
                d_model, dim_k, dim_v, n_heads, dim_fc, half=half)
            for _ in range(n_layers)
        ])
        if projection:
            self.projection = nn.Linear(d_model, d_in)

    def forward(self, x):
        output = self.model_list(self.embed(x))
        if self.projection:
            output = self.projection(output)
        return output


class SpatialTSFM(TemporalTSFM):

    def __init__(
        self,
        d_in: int,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        dim_fc: int = 128,
        n_layers: int = 1,
        half: bool = False,
        projection: bool = True,
    ) -> None:
        super().__init__(
            d_in,
            d_model,
            dim_k,
            dim_v,
            n_heads,
            dim_fc,
            n_layers,
            half=half,
            projection=projection,
        )

    def forward(self, x):
        x = x.swapaxes(0, 1)
        output = self.model_list(self.embed(x))
        if self.projection:
            output = self.projection(output)
        return output.swapaxes(0, 1)


class SingleGCN(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dist_mat: torch.tensor,
        n_layers: int = 1,
        activation=F.relu,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dist_mat = dist_mat
        self.n_layers = n_layers
        self.activation = activation

        d = self.dist_mat.shape[0]
        self.sigma = nn.Linear(1, d)
        self.linears = nn.ModuleList(
            [nn.Linear(in_channels, out_channels, bias=bias)] + [
                nn.Linear(out_channels, out_channels, bias=bias)
                for _ in range(n_layers)
            ])
        self.mat_2 = dist_mat.square()
        self.mask = self.generate_mask()

    def generate_mask(self):
        # 距离为0的位置设为1
        matrix = (self.dist_mat.cpu().detach().numpy() == 0.).astype(int)
        # 对角线设为0
        matrix[range(matrix.shape[0]), range(matrix.shape[0])] = 0
        return torch.Tensor(matrix) == 1

    def forward(self, x):
        # 计算收缩矩阵, x : (N, T, D)
        sigma = torch.sigmoid(self.sigma.weight * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        exp = torch.exp(-self.mat_2 / (2 * sigma**2))
        prior = exp / (math.sqrt(2 * math.pi) * sigma)
        prior = prior.masked_fill(self.mask.to(exp.device), 0)
        prior /= prior.sum(0)
        for i in range(self.n_layers):
            wx = prior @ x
            x = self.activation(self.linears[i](wx))

        return x, prior


class MultipleGCN(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        matrices: torch.tensor,
        n_layers: int = 1,
        activation=F.relu,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.activation = activation

        self.n_graph = matrices.shape[0]
        self.matrices = matrices  # (3, N, N)

        self.sigma = nn.Linear(self.n_graph, matrices.shape[-1])
        self.alpha = nn.parameter.Parameter(
            torch.ones(self.n_graph) / self.n_graph)
        self.linears = nn.ModuleList(
            [nn.Linear(in_channels, out_channels, bias=bias)] + [
                nn.Linear(out_channels, out_channels, bias=bias)
                for _ in range(n_layers)
            ])
        self.mask = self.generate_mask()

    def generate_mask(self):
        # 距离为0的位置设为1
        matrix = (self.matrices.cpu().detach().numpy() == 0.).astype(int)
        # 对角线设为0
        matrix[:, range(matrix.shape[1]), range(matrix.shape[1])] = 0
        return torch.Tensor(matrix) == 1

    def forward(self, x):
        # x: (VAR * NP, N, D)
        B, N, D = x.shape
        prior = self.STS  # (N, N)
        
        # 重塑x以适应矩阵乘法
        x_reshaped = x.reshape(-1, N, D)  # (B, N, D)
        
        for i in range(self.n_layers):
            # 执行矩阵乘法
            wx = torch.matmul(prior, x_reshaped)  # (B, N, D)
            x_reshaped = self.activation(self.linears[i](wx))
        
        # 恢复原始形状
        x = x_reshaped.reshape(B, N, D)
        return x, prior

    @property
    def STS(self):
        sigma = self.sigma.weight.reshape(self.n_graph, 1, -1)
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        
        # 计算每个图的先验概率
        exp = torch.exp(-self.matrices / (2 * sigma**2))
        prior = exp / (math.sqrt(2 * math.pi) * sigma)
        prior = prior.masked_fill(self.mask.to(exp.device), 0)
        
        # 归一化
        prior = prior / (prior.sum(1, keepdims=True) + 1e-8)
        
        # 合并多个图的先验概率
        prior = torch.matmul(prior.permute(1, 2, 0), torch.softmax(self.alpha, 0))
        return prior  # (N, N)


def MLP(
    n_layers: int,
    n_input: int,
    n_hidden: Union[int, List[int], Tuple[int]],
    n_output: int,
    act: nn.modules,
    last_act: bool = True,
):
    if n_layers < 0:
        raise ValueError("Parameter 'n_layers' must be non-negative!")
    elif n_layers == 0:
        return nn.Identity()

    if type(n_hidden) not in {list, tuple}:
        n_hidden = [n_hidden] * max(n_layers - 1, 0)

    n_per_layer = [n_input] + list(n_hidden) + [n_output]
    assert len(n_per_layer) == n_layers + 1
    module_list = []
    for i in range(n_layers):
        module_list.extend([
            nn.Linear(n_per_layer[i], n_per_layer[i + 1]),
            act(),
        ])
    if not last_act:
        module_list.pop()
    return nn.Sequential(*module_list)


def GCN(
    n_layers: int,
    n_input: int,
    n_hidden: Union[int, List[int], Tuple[int]],
    n_output: int,
    act: nn.modules,
    last_act: bool = True,
    conv_layer: nn.Module = GCNConv,
):
    if n_layers < 0:
        raise ValueError("Parameter 'n_layers' must be non-negative!")
    elif n_layers == 0:
        return Sequential('x, edge_index', [(nn.Identity(), 'x -> x')])

    if type(n_hidden) not in {list, tuple}:
        n_hidden = [n_hidden] * max(n_layers - 1, 0)

    n_per_layer = [n_input] + n_hidden + [n_output]
    assert len(n_per_layer) == n_layers + 1
    module_list = []
    for i in range(n_layers):
        module_list.extend([
            (
                conv_layer(n_per_layer[i], n_per_layer[i + 1]),
                'x, edge_index -> x',
            ),
            act(),
        ])

    if not last_act:
        module_list.pop()
    return Sequential('x, edge_index', module_list)


class SoftClusterLayer(nn.Module):
    '''Spatial heterogeneity modeling by using a soft-clustering paradigm.
    '''

    def __init__(self, c_in, nmb_prototype, poi_sim=None, dist_mat=None, tau=0.5):
        super(SoftClusterLayer, self).__init__()
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.prototypes = nn.Linear(c_in, nmb_prototype, bias=False)
        self.tau = tau
        self.d_model = c_in
        
        # 添加POI相似度和距离矩阵
        self.poi_sim = poi_sim if poi_sim is not None else None
        self.dist_mat = dist_mat if dist_mat is not None else None
        
        # 添加贝塔混合分布参数
        self.beta_params = nn.Parameter(torch.ones(2, 2))  # 初始化贝塔分布参数
        
        # 添加空间-语义融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(c_in * 2, c_in),
            nn.GELU(),
            nn.Linear(c_in, c_in)
        )

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def compute_hard_negative_mask(self, z):
        """计算困难负样本掩码
        Args:
            z: 节点特征 [B, N, D]
        Returns:
            mask: 困难负样本掩码 [B, N, N]
        """
        if self.poi_sim is None or self.dist_mat is None:
            return None
            
        B, N, D = z.shape
        device = z.device
        
        # 重塑特征以计算相似度
        z_reshaped = z.reshape(-1, D)  # [B*N, D]
        
        # 计算特征相似度
        feat_sim = torch.matmul(z_reshaped, z_reshaped.t())  # [B*N, B*N]
        
        # 重塑回原始维度
        feat_sim = feat_sim.reshape(B, N, B, N)  # [B, N, B, N]
        
        # 标准化POI相似度和距离
        poi_sim_norm = F.normalize(self.poi_sim, p=2, dim=1)  # [N, N]
        dist_norm = F.normalize(self.dist_mat, p=2, dim=1)    # [N, N]
        
        # 创建批次掩码
        mask = torch.zeros((B, N, N), device=device)
        
        # 对每个批次计算掩码
        for b in range(B):
            # 获取当前批次的相似度
            curr_feat_sim = feat_sim[b, :, b, :]  # [N, N]
            
            # 计算语义-空间不一致性
            semantic_spatial_inconsistency = torch.abs(curr_feat_sim - poi_sim_norm)
            spatial_semantic_inconsistency = torch.abs(curr_feat_sim - dist_norm)
            
            # 合并两种不一致性
            inconsistency = semantic_spatial_inconsistency + spatial_semantic_inconsistency
            
            # 使用贝塔分布估计真负样本概率
            alpha = F.softplus(self.beta_params[0])
            beta = F.softplus(self.beta_params[1])
            true_negative_prob = torch.distributions.Beta(alpha, beta).cdf(inconsistency)
            
            # 生成困难负样本掩码
            mask[b] = (true_negative_prob > 0.5).float()
            
            # 将对角线设为0（排除自身）
            mask[b].fill_diagonal_(0)
        
        return mask

    def forward(self, z):
        """Compute the contrastive loss of batched data.
        Args:
            z: 节点特征 [B, N, D]
        Returns:
            z: 更新后的特征
            l: 对比损失
        """
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = self.l2norm(w)
            self.prototypes.weight.copy_(w)

        # 计算困难负样本掩码
        hard_negative_mask = self.compute_hard_negative_mask(z)
        
        # 特征归一化
        z_norm = self.l2norm(z.reshape(-1, self.d_model))
        
        # 计算原型分配
        self.zc = self.prototypes(z_norm)  # [B*N, K]
        
        # 使用Sinkhorn算法计算软分配
        with torch.no_grad():
            q = sinkhorn(self.zc.detach())
            
        # 计算对比损失
        if hard_negative_mask is not None:
            # 应用困难负样本掩码
            hard_negative_mask = hard_negative_mask.reshape(-1, 1)  # [B*N, 1]
            masked_zc = self.zc * hard_negative_mask
            l = -torch.mean(torch.sum(q * F.log_softmax(masked_zc / self.tau, dim=1), dim=1))
        else:
            l = -torch.mean(torch.sum(q * F.log_softmax(self.zc / self.tau, dim=1), dim=1))
            
        return z, l


@torch.no_grad()
def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):
    Q = torch.exp(out / epsilon).t(
    )  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1]  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.t()
