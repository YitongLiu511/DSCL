from pyod.models.base import BaseDetector
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import roc_auc_score
from typing import Union, List, Tuple, Literal
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch.optim as optim

from ..model.module import SpatialTSFM, TemporalTSFM
from ..model.baseline import DOMINANT, AnomalyDAE, GCN, LSTMAE
from ..model.anomaly import (
    STAnomalyFormer_v1,
    STAnomalyFormer_v2,
    STPatchFormer,
    STPatchMaskFormer,
    STPatch_MGCNFormer,
)
from .utils import predict_by_score, EarlyStopping


def kl_loss(p, q):
    res = p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))
    # 确保输入是2D张量
    if len(res.shape) > 2:
        res = res.reshape(res.size(0), -1)
    return torch.mean(res, dim=-1)


def sym_kl_loss(p, q):
    return (kl_loss(p, q) + kl_loss(q, p)) / 2


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, x):
        super(SimpleDataset, self).__init__()
        self.x = x

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.x.shape[0]


class TemporalTSFMDetector(BaseDetector):

    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_heads: int,
        dim_fc_expand: int = 1,
        n_layers: int = 1,
        device: str = 'cpu',
        epoch: int = 10,
        lr: float = 1e-4,
        batch_size: int = -1,
        contamination=0.1,
        verbose=False,
        **kwargs,
    ):
        super().__init__(contamination)
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.dim_fc_expand = dim_fc_expand
        self.n_layers = n_layers
        self.device = device
        self.epoch = epoch
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose
        self.tsfm_args = {
            "d_in": d_in,
            "d_model": d_model,
            "dim_k": d_model // n_heads,
            "dim_v": d_model // n_heads,
            "n_heads": n_heads,
            "dim_fc": n_heads * self.d_model,
            "n_layers": n_layers,
        }

    def fit(self, x, test_x, y=None):
        x_ = torch.FloatTensor(x)
        self.model = TemporalTSFM(**self.tsfm_args).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        batch_size = self.batch_size if self.batch_size != -1 else x.shape[0]
        dataset = SimpleDataset(x_)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        self.model.train()

        if y is not None:
            max_auc = 0.

        for epoch in range(self.epoch):
            for data in train_loader:
                data = data.to(self.device)
                output = self.model(data)
                score = torch.square(output - data).mean((1, 2))
                loss = score.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.verbose:
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                if y is not None:
                    score = self.decision_function(test_x)
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                    if auc >= max_auc:
                        max_auc = auc
                        self.model_copy = deepcopy(self.model)
                print(log)

        if y is not None:
            self.model = self.model_copy

        self.decision_scores_ = self.decision_function(x)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    def presict(self, x):
        score = self.decision_function(x)
        return predict_by_score(score, self.contamination)

    @torch.no_grad()
    def decision_function(self, x):
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        output = self.model(x)
        recon = torch.square(output - x)
        score = recon.mean((1, 2))
        return score.cpu().numpy()


class SpatialTSFMDetector(BaseDetector):

    def __init__(
        self,
        d_in: int,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        dim_fc_expand: int,
        n_layers: int = 1,
        device: str = 'cpu',
        epoch: int = 10,
        lr: float = 1e-4,
        batch_size: int = -1,
        contamination=0.1,
        verbose=False,
    ):
        super().__init__(contamination)
        self.d_in = d_in
        self.d_model = d_model
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.n_heads = n_heads
        self.dim_fc_expand = dim_fc_expand
        self.n_layers = n_layers
        self.device = device
        self.epoch = epoch
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose
        self.tsfm_args = {
            "d_in": d_in,
            "d_model": d_model,
            "dim_k": dim_k,
            "dim_v": dim_v,
            "n_heads": n_heads,
            "dim_fc": n_heads * self.d_model,
            "n_layers": n_layers,
        }

    def fit(self, x, y=None):
        x_ = torch.FloatTensor(x)
        self.model = SpatialTSFM(**self.tsfm_args).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        batch_size = self.batch_size if self.batch_size != -1 else x.shape[1]
        dataset = SimpleDataset(x_.swapaxes(0, 1))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        self.model.train()

        if y is not None:
            max_auc = 0.

        for epoch in range(self.epoch):
            for data in dataloader:
                data = data.to(self.device).swapaxes(0, 1)
                output = self.model(data)
                score = torch.square(output - data).mean((1, 2))
                loss = score.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.verbose:
                with torch.no_grad():
                    cuda_x = x_.to(self.device)
                    output = self.model(cuda_x)
                    score = torch.square(output - cuda_x).mean(
                        (1, 2)).cpu().numpy()
                loss = score.mean()
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss)
                if y is not None:
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                    if auc >= max_auc:
                        max_auc = auc
                        self.model_copy = deepcopy(self.model)
                print(log)

        if y is not None:
            self.model = self.model_copy

        self.decision_scores_ = self.decision_function(x)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    def presict(self, x):
        score = self.decision_function(x)
        return predict_by_score(score, self.contamination)

    @torch.no_grad()
    def decision_function(self, x):
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        output = self.model(x)
        score = torch.square(output - x).mean((1, 2))
        return score.cpu().numpy()


class DOMINANTDetector(BaseDetector):

    def __init__(
        self,
        n_hidden: Union[List[int], Tuple[int], int] = 64,
        n_layers: int = 3,
        act=nn.ReLU,
        alpha: float = 0.5,
        lr: float = 0.005,
        weight_decay: float = 0.,
        device: str = 'cpu',
        epoch: int = 5,
        verbose: bool = False,
        contamination: float = 0.1,
    ) -> None:
        super().__init__(contamination)
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.act = act
        self.alpha = alpha
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.epoch = epoch
        self.verbose = verbose

    def fit(self, G: Data, test_G: Data, y=None):
        G = G.to(self.device)
        test_G = test_G.to(self.device)
        self.model = DOMINANT(
            G.num_node_features,
            self.n_hidden,
            self.n_layers,
            self.act,
        ).to(self.device)
        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]
        optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.model.train()

        if y is not None:
            max_auc = 0.

        for epoch in range(1, self.epoch + 1):
            stru_recon, attr_recon = self.model(test_G.x, test_G.edge_index)

            stru_score = torch.square(stru_recon - A).sum(1)
            attr_score = torch.square(attr_recon - G.x).sum(1)
            score = self.alpha * stru_score + (1 - self.alpha) * attr_score
            loss = score.mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            if self.verbose:
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                if y is not None:
                    score = self.decision_function(test_G)
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                    if auc >= max_auc:
                        max_auc = auc
                        self.model_copy = deepcopy(self.model)
                print(log)

        if y is not None:
            self.model = self.model_copy

        self.decision_scores_ = self.decision_function(G)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    @torch.no_grad()
    def decision_function(self, G: Data):
        G = G.to(self.device)
        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]
        stru_recon, attr_recon = self.model(G.x, G.edge_index)
        stru_score = torch.square(stru_recon - A).sum(1).sqrt()
        attr_score = torch.square(attr_recon - G.x).sum(1).sqrt()
        score = self.alpha * stru_score + (1 - self.alpha) * attr_score
        return score.cpu().numpy()

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)


class AnomalyDAEDetector(BaseDetector):

    def __init__(
        self,
        embed_dim: int = 8,
        n_hidden: int = 64,
        act=nn.ReLU,
        alpha: float = 0.5,
        theta: float = 1.1,
        eta: float = 1.1,
        lr: float = 0.005,
        weight_decay: float = 0.,
        device: str = 'cpu',
        epoch: int = 100,
        verbose: bool = False,
        contamination: float = 0.1,
    ) -> None:
        super().__init__(contamination)
        self.embed_dim = embed_dim
        self.n_hidden = n_hidden
        self.act = act
        self.alpha = alpha
        self.theta = theta
        self.eta = eta
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.epoch = epoch
        self.verbose = verbose

    def fit(self, G: Data, test_G: Data, y=None):
        G = G.to(self.device)
        test_G = test_G.to(self.device)
        self.model = AnomalyDAE(
            G.num_nodes,
            G.num_node_features,
            self.n_hidden,
            self.embed_dim,
            self.act,
        ).to(self.device)

        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]
        Theta = torch.full_like(A, self.theta).to(self.device)
        Theta[G.edge_index[0], G.edge_index[1]] = 1.
        Eta = torch.ones_like(G.x).to(self.device)
        Eta[G.x != 0] = self.eta

        optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.model.train()

        if y is not None:
            max_auc = 0.

        for epoch in range(1, self.epoch + 1):
            stru_recon, attr_recon = self.model(G.x, G.edge_index)
            stru_error = torch.square((stru_recon - A) * Theta)
            attr_error = torch.square((attr_recon - G.x) * Eta)
            stru_score = stru_error.sum(1)
            attr_score = attr_error.sum(1)
            score = self.alpha * stru_score + (1 - self.alpha) * attr_score
            loss = score.mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            if self.verbose:
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                if y is not None:
                    score = self.decision_function(test_G)
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                    if auc >= max_auc:
                        max_auc = auc
                        self.model_copy = deepcopy(self.model)
                print(log)

        if y is not None:
            self.model = self.model_copy

        self.decision_scores_ = self.decision_function(G)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    @torch.no_grad()
    def decision_function(self, G: Data):
        G = G.to(self.device)
        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]
        Theta = torch.full_like(A, self.theta).to(self.device)
        Theta[G.edge_index[0], G.edge_index[1]] = 1.
        Eta = torch.ones_like(G.x).to(self.device)
        Eta[G.x != 0] = self.eta

        stru_recon, attr_recon = self.model(G.x, G.edge_index)
        stru_score = torch.square((stru_recon - A) * Theta).sum(1)
        attr_score = torch.square((attr_recon - G.x) * Eta).sum(1)
        score = self.alpha * stru_score + (1 - self.alpha) * attr_score
        return score.cpu().numpy()

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)


class OCGNNDetector(BaseDetector):

    def __init__(
        self,
        n_hidden: Union[List[int], Tuple[int], int] = 64,
        n_layers: int = 4,
        act=nn.ReLU,
        beta: float = 0.1,
        phi: int = 10,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = 'cpu',
        epoch: int = 100,
        verbose: bool = False,
        contamination: float = 0.1,
    ) -> None:
        super().__init__(contamination)
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.act = act
        self.beta = beta
        self.phi = phi
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.epoch = epoch
        self.verbose = verbose

    def fit(self, G: Data, test_G: Data, y=None):
        G = G.to(self.device)
        test_G = test_G.to(self.device)
        self.model = GCN(
            self.n_layers,
            G.num_features,
            self.n_hidden,
            self.n_hidden,
            self.act,
            last_act=False,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        with torch.no_grad():
            r = 0.
            c = self.model(G.x, G.edge_index).mean(0)  # cuda

        self.model.train()

        if y is not None:
            max_auc = 0.

        for epoch in range(1, self.epoch + 1):
            dV = (self.model(G.x, G.edge_index) - c).square().sum(1)
            loss = torch.relu(dV - r**2).mean() / self.beta

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.verbose:
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                if y is not None:
                    with torch.no_grad():
                        score = (self.model(test_G.x, test_G.edge_index) -
                                 c).square().sum(1).cpu()
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                    if auc >= max_auc:
                        max_auc = auc
                        self.model_copy = deepcopy(self.model)
                print(log)

            if epoch % self.phi == 0:
                with torch.no_grad():
                    r = torch.quantile(dV, 1 - self.beta).item()
                    c = self.model(G.x, G.edge_index).mean(0)

        with torch.no_grad():
            self.r = torch.quantile(dV, 1 - self.beta).item()
            self.c = self.model(G.x, G.edge_index).mean(0)

        if y is not None:
            self.model = self.model_copy

        self.decision_scores_ = self.decision_function(G)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    @torch.no_grad()
    def decision_function(self, G: Data):
        G = G.to(self.device)
        score = (self.model(G.x, G.edge_index) - self.c).square().sum(1)
        return score.cpu().numpy() - self.r**2

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)


class LSTM_AEDetector(BaseDetector):

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        dropout: float = 0.,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = 'cpu',
        batch_size: int = -1,
        epoch: int = 100,
        verbose: bool = False,
        contamination: float = 0.1,
    ):
        super().__init__(contamination)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.batch_size = batch_size
        self.epoch = epoch
        self.verbose = verbose

    def fit(self, G: Data, test_G: Data, y=None):
        x = G.x.reshape(G.x.shape[0], -1, self.input_size).to(self.device)
        test_x = test_G.x.reshape(G.x.shape[0], -1,
                                  self.input_size).to(self.device)
        self.model = LSTMAE(
            self.input_size,
            self.hidden_size,
            self.dropout,
            x.shape[-2],
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        if y is not None:
            max_auc = 0.

        batch_size = self.batch_size if self.batch_size != -1 else G.x.shape[0]
        dataset = SimpleDataset(x)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        for epoch in range(1, self.epoch + 1):
            self.model.train()

            for data in loader:
                output = self.model(data)
                loss = F.mse_loss(output, data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.verbose:
                self.model.eval()
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                if y is not None:
                    with torch.no_grad():
                        output = self.model(test_x)
                        score = torch.square(test_x - output).sum(
                            (-1, -2)).cpu().numpy()
                    auc = roc_auc_score(y, score)
                    log += ", AUC={:6f}".format(auc)
                    if auc >= max_auc:
                        max_auc = auc
                        self.model_copy = deepcopy(self.model)
                print(log)

        if y is not None:
            self.model = self.model_copy

        self.decision_scores_ = self.decision_function(G)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    @torch.no_grad()
    def decision_function(self, G: Data):
        x = G.x.reshape(G.x.shape[0], -1, self.input_size).to(self.device)
        return torch.square(x - self.model(x)).sum((-1, -2)).cpu().numpy()

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)


class STAnomalyFormerDetector_v1(BaseDetector):

    def __init__(
        self,
        d_in: int,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        n_gcn: int,
        alpha: float = 0.,
        device: str = 'cpu',
        epoch: int = 10,
        batch_size: int = -1,
        lr: float = 1e-4,
        contamination=0.1,
        verbose: bool = False,
        log_interval: int = 1,
        **kwargs,
    ):
        super().__init__(contamination)
        self.d_in = d_in
        self.d_model = d_model
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.n_heads = n_heads
        self.device = device
        self.epoch = epoch
        self.lr = lr
        self.verbose = verbose
        self.tsfm_args = {
            "d_in": d_in,
            "d_model": d_model,
            "dim_k": dim_k,
            "dim_v": dim_v,
            "n_heads": n_heads,
            "batch_size": batch_size,
        }
        self.alpha = alpha
        self.n_gcn = n_gcn
        self.log_interval = log_interval

    def fit(self, x, mat, y=None):
        x_ = torch.FloatTensor(x).to(self.device)
        mat = torch.FloatTensor(mat).to(self.device)
        self.model = STAnomalyFormer_v1(
            mat.to(self.device),
            n_gcn=self.n_gcn,
            **self.tsfm_args,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if y is not None:
            max_auc = 0.

        process = range(self.epoch) if self.verbose else tqdm(range(
            self.epoch))
        for epoch in process:
            self.model.train()
            output, score_dy, score_st = self.model(x_)

            recon = torch.square(output - x_).mean((1, 2))
            discrepancy = sym_kl_loss(score_dy, score_st)
            score = recon + self.alpha * discrepancy

            loss = score.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                output, score_dy, score_st = self.model(x_)

                recon = torch.square(output - x_).mean((1, 2))
                discrepancy = sym_kl_loss(score_dy, score_st)
                score = recon + self.alpha * discrepancy
                loss = score.mean()

            log = "Epoch {:3d}, loss={:5.6f}".format(
                epoch,
                loss.item(),
            )
            if y is not None:
                auc = roc_auc_score(y, score.cpu().numpy())
                log += ", AUC={:6f}".format(auc)
                if auc >= max_auc:
                    max_auc = auc
                    self.model_copy = deepcopy(self.model)

            if self.verbose:
                if (epoch + 1) % self.log_interval == 0:
                    print(log)
            elif y is not None:
                process.set_postfix(
                    max_auc="max: {:.4f}, current: {:.4f}".format(
                        max_auc, auc),
                    refresh=True,
                )

        if y is not None:
            self.model = self.model_copy

        self.decision_scores_ = self.decision_function(x)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )

        return self

    def presict(self, x):
        score = self.decision_function(x)
        return predict_by_score(score, self.contamination)

    def detect_anomalies(self, region_scores, time_scores):
        """检测异常并返回异常事件和分数"""
        # 确保输入张量的维度正确
        if region_scores.dim() == 3:  # [263, 1, 1]
            region_scores = region_scores.squeeze(-1).squeeze(-1)  # [263]
        
        if time_scores.dim() == 4:  # [263, 23, 14, 1]
            time_scores = time_scores.squeeze(-1)  # [263, 23, 14]
        
        # 获取时间步数
        n_times = time_scores.size(1)
        
        # 初始化异常分数张量
        anomaly_scores = torch.zeros((region_scores.size(0), n_times, 2), device=region_scores.device)
        
        # 扩展区域得分到所有时间步
        anomaly_scores[:, :, 0] = region_scores.unsqueeze(1).expand(-1, n_times)
        
        # 计算时间异常得分
        anomaly_scores[:, :, 1] = time_scores.mean(dim=-1)  # 对天数维度取平均
        
        # 计算阈值
        region_threshold = torch.quantile(region_scores, 1 - self.contamination)
        time_threshold = torch.quantile(time_scores.reshape(-1), 1 - self.contamination)
        
        # 标记异常
        anomaly_events = torch.zeros_like(anomaly_scores, dtype=torch.bool)
        anomaly_events[:, :, 0] = region_scores.unsqueeze(1).expand(-1, n_times) > region_threshold
        anomaly_events[:, :, 1] = time_scores.mean(dim=-1) > time_threshold
        
        return anomaly_events, anomaly_scores

    @torch.no_grad()
    def decision_function(self, x, return_dict=False):
        """计算异常分数"""
        self.model.eval()
        with torch.no_grad():
            # 确保输入数据在正确的设备上
            x = torch.tensor(x, dtype=torch.float).to(self.device)
            
            # 获取模型输出
            region_scores, time_scores = self.model(x)
            
            # 添加数值稳定性检查
            if torch.isnan(region_scores).any() or torch.isnan(time_scores).any():
                print("[WARNING] 检测到NaN值，进行数值稳定性处理")
                region_scores = torch.nan_to_num(region_scores, nan=0.0)
                time_scores = torch.nan_to_num(time_scores, nan=0.0)
            
            # 确保分数在合理范围内
            region_scores = torch.clamp(region_scores, min=0.0, max=10.0)
            time_scores = torch.clamp(time_scores, min=0.0, max=10.0)
            
            # 检测异常
            anomaly_events, anomaly_scores = self.detect_anomalies(region_scores, time_scores)
            
            if return_dict:
                return {
                    'region_scores': region_scores.cpu(),
                    'time_scores': time_scores.cpu(),
                    'anomaly_events': anomaly_events.cpu(),
                    'anomaly_scores': anomaly_scores.cpu()
                }
            else:
                return anomaly_scores.cpu()

    def evaluate(self, x, y_true):
        """评估模型性能"""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y_true, np.ndarray):
            y_true = torch.tensor(y_true, dtype=torch.float).to(self.device)

        # 获取异常分数
        output_dict = self.decision_function(x, return_dict=True)
        anomaly_scores = output_dict['anomaly_scores']  # shape: [N, T_patch, 2]

        # 转换为numpy数组进行评估
        anomaly_scores_np = anomaly_scores.cpu().numpy()
        y_true_np = y_true.cpu().numpy()

        # 打印调试信息
        print("[DEBUG] 输入数据形状:")
        print(f"x shape: {x.shape}")
        print(f"y_true shape: {y_true.shape}")
        print(f"anomaly_scores shape: {anomaly_scores.shape}")

        # 获取patch时间步数
        N, T_patch, _ = anomaly_scores_np.shape
        print(f"[DEBUG] 模型输出的patch时间步数: {T_patch}")

        # 截取标签的前T_patch个时间步
        if len(y_true_np.shape) == 4:  # [N, T, D, 2]
            y_true_np_patch = y_true_np[:, :T_patch, :, :]  # 截取前T_patch个时间步
            # 将标签reshape为[N*T_patch*D, 2]
            y_true_flat = y_true_np_patch.reshape(-1, 2)
            # 将分数reshape为[N*T_patch*D, 2]
            scores_flat = np.repeat(anomaly_scores_np, y_true_np_patch.shape[2], axis=1).reshape(-1, 2)
        else:  # [N, T, 2]
            y_true_np_patch = y_true_np[:, :T_patch, :]
            y_true_flat = y_true_np_patch.reshape(-1, 2)
            scores_flat = anomaly_scores_np.reshape(-1, 2)

        print(f"[DEBUG] 截取后的标签shape: {y_true_np_patch.shape}")
        print(f"[DEBUG] 展平后的标签shape: {y_true_flat.shape}")
        print(f"[DEBUG] 展平后的分数shape: {scores_flat.shape}")

        try:
            # 计算区域异常AUC
            region_auc = roc_auc_score(y_true_flat[:, 0], scores_flat[:, 0])
            # 计算时间异常AUC
            time_auc = roc_auc_score(y_true_flat[:, 1], scores_flat[:, 1])
            # 计算综合AUC
            combined_auc = roc_auc_score(y_true_flat.reshape(-1), scores_flat.reshape(-1))

            metrics = {
                'region_auc': region_auc,
                'time_auc': time_auc,
                'combined_auc': combined_auc
            }

            if self.verbose:
                print(f"区域AUC: {region_auc:.3f}, 时间AUC: {time_auc:.3f}, 综合AUC: {combined_auc:.3f}")

            return metrics

        except Exception as e:
            print(f"评估过程中出现错误: {str(e)}")
            print("错误详情:")
            print(f"y_true shape: {y_true.shape}")
            print(f"anomaly_scores shape: {anomaly_scores.shape}")
            return {
                'region_auc': 0.0,
                'time_auc': 0.0,
                'combined_auc': 0.0
            }

    def fit(self, x, mats, evaluate=None):
        """
        训练模型
        Args:
            x: 输入数据
            mats: 距离矩阵
            evaluate: 评估数据元组 (eval_x, eval_y)
        Returns:
            self
        """
        print("\n" + "="*50)
        print("训练开始前的状态检查:")
        print(f"1. 模型是否存在: {hasattr(self, 'model')}")
        if hasattr(self, 'model'):
            print(f"2. 模型训练状态: {self.model.training}")
            print(f"3. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
        print("="*50 + "\n")
        
        # 数据预处理
        if isinstance(x, np.ndarray):
            x_ = torch.FloatTensor(x).to(self.device)
        else:
            x_ = x.to(self.device)
            
        if isinstance(mats, np.ndarray):
            mats = torch.FloatTensor(mats).to(self.device)
        else:
            mats = mats.to(self.device)
            
        # 初始化早停
        self.early_stopping = EarlyStopping(
            patience=100,
            trace_func=tqdm.write,
            delta=0.01,
        )
        
        # 更新模型参数
        self.model_args.update({
            "seq_len": x_.shape[1],
            "patch_len": self.model_args["patch_len"],
            "stride": self.model_args["stride"],
            "dist_mats": mats
        })
        
        # 初始化模型
        self.model = STPatch_MGCNFormer(**self.model_args).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # 训练循环
        process = range(self.epoch) if not self.verbose else tqdm(range(self.epoch))
        for epoch in process:
            print(f"\n{'='*20} Epoch {epoch} {'='*20}")
            
            # 训练阶段
            print("\n=== 训练阶段开始 ===")
            print("训练阶段开始前的状态检查:")
            print(f"1. 模型训练状态: {self.model.training}")
            print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
            
            # 确保模型处于训练模式
            self.model.train()
            print("\n设置train()后的状态检查:")
            print(f"1. 模型训练状态: {self.model.training}")
            print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
            
            # 前向传播，确保设置return_dict=True
            print("\n前向传播前的状态检查:")
            print(f"1. 模型训练状态: {self.model.training}")
            print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
            
            # 保存原始训练状态
            original_training = self.model.training
            print(f"3. original_training值: {original_training}")
            
            # 确保模型处于训练模式
            self.model.train()
            
            # 前向传播
            output_dict = self.model(x_, return_dict=True)
            print("\n前向传播后的状态检查:")
            print(f"1. 模型训练状态: {self.model.training}")
            print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
            print(f"3. 输出字典键: {output_dict.keys()}")
            
            # 从输出字典中获取重建结果和异常分数
            reconstruction = output_dict['reconstruction']
            anomaly_scores = output_dict['anomaly_scores']
            recon_loss = output_dict['recon_loss']
            
            # 计算总损失
            loss = recon_loss
            
            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print("\n优化后的状态检查:")
            print(f"1. 模型训练状态: {self.model.training}")
            print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
            print(f"3. 当前损失值: {loss.item():.6f}")
            
            # 评估阶段
            if evaluate is not None:
                print("\n=== 评估阶段开始 ===")
                print("评估前的状态检查:")
                print(f"1. 模型训练状态: {self.model.training}")
                print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
                
                # 确保模型处于评估模式
                self.model.eval()
                print("\n设置eval()后的状态检查:")
                print(f"1. 模型训练状态: {self.model.training}")
                print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
                
                with torch.no_grad():
                    # 处理评估数据
                    if isinstance(evaluate[0], np.ndarray):
                        eval_x = torch.FloatTensor(evaluate[0]).to(self.device)
                    else:
                        eval_x = evaluate[0].to(self.device)
                    
                    # 获取评估结果
                    metrics = self.evaluate(eval_x, evaluate[1])
                    auc = (metrics['region_auc'] + metrics['time_auc'] + metrics['combined_auc']) / 3
                    
                    # 早停检查
                    self.early_stopping(auc, self.model)
                    if self.early_stopping.early_stop:
                        if self.verbose:
                            print("触发早停机制")
                        break
                
                print("\n评估后的状态检查:")
                print(f"1. 模型训练状态: {self.model.training}")
                print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
                print(f"3. 评估指标:")
                print(f"   - 区域AUC: {metrics['region_auc']:.3f}")
                print(f"   - 时间AUC: {metrics['time_auc']:.3f}")
                print(f"   - 综合AUC: {metrics['combined_auc']:.3f}")
                
                # 更新进度条
                if self.verbose:
                    process.set_postfix(
                        auc=f"区域AUC: {metrics['region_auc']:.3f}, "
                            f"时间AUC: {metrics['time_auc']:.3f}, "
                            f"综合AUC: {metrics['combined_auc']:.3f}",
                        refresh=True
                    )
            else:
                if self.verbose:
                    process.set_postfix(loss=f"{loss.item():.5f}", refresh=True)
            
            # 确保模型在下一个epoch开始时处于训练模式
            self.model.train()
        
        # 加载最佳模型
        if evaluate is not None:
            print("\n=== 加载最佳模型 ===")
            print("加载前的状态检查:")
            print(f"1. 模型训练状态: {self.model.training}")
            print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
            
            self.model.load_state_dict(torch.load(self.early_stopping.path))
            
            print("\n加载后的状态检查:")
            print(f"1. 模型训练状态: {self.model.training}")
            print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
        
        # 计算最终决策分数
        print("\n=== 计算最终决策分数 ===")
        print("计算前的状态检查:")
        print(f"1. 模型训练状态: {self.model.training}")
        print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
        
        # 确保模型处于评估模式
        self.model.eval()
        output_dict = self.decision_function(x, return_dict=True)
        self.decision_scores_ = output_dict['anomaly_scores']
        
        print("\n训练结束时的状态检查:")
        print(f"1. 模型训练状态: {self.model.training}")
        print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
        
        return self


class STAnomalyFormerDetector_v2(STAnomalyFormerDetector_v1):

    def fit(self, x, mat, y=None):
        x_ = torch.FloatTensor(x).to(self.device)
        mat = torch.FloatTensor(mat).to(self.device)
        self.model = STAnomalyFormer_v1(
            mat.to(self.device),
            n_gcn=self.n_gcn,
            **self.tsfm_args,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if y is not None:
            max_auc = 0.

        process = range(self.epoch) if self.verbose else tqdm(range(
            self.epoch))
        for epoch in process:
            self.model.train()
            output, score_dy, score_st = self.model(x_)

            recon = torch.square(output - x_).mean((1, 2))
            discrepancy = sym_kl_loss(
                score_dy,
                score_st.detach(),
            ) - sym_kl_loss(
                score_dy.detach(),
                score_st,
            )
            score = recon + self.alpha * discrepancy

            loss = score.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                output, score_dy, score_st = self.model(x_)

                recon = torch.square(output - x_).mean((1, 2))
                discrepancy = sym_kl_loss(score_dy, score_st)
                score = recon + self.alpha * discrepancy
                loss = score.mean()

            log = "Epoch {:3d}, loss={:5.6f}".format(
                epoch,
                loss.item(),
            )
            if y is not None:
                auc = roc_auc_score(y, score.cpu().numpy())
                log += ", AUC={:6f}".format(auc)
                if auc >= max_auc:
                    max_auc = auc
                    self.model_copy = deepcopy(self.model)

            if self.verbose:
                if (epoch + 1) % self.log_interval == 0:
                    print(log)
            elif y is not None:
                process.set_postfix(
                    max_auc="max: {:.4f}, current: {:.4f}".format(
                        max_auc, auc),
                    refresh=True,
                )

        if y is not None:
            self.model = self.model_copy

        self.decision_scores_ = self.decision_function(x)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )

        return self


class STAnomalyFormerDetector_v3(STAnomalyFormerDetector_v2):

    def __init__(self,
                 d_in: int,
                 d_model: int,
                 dim_k: int,
                 dim_v: int,
                 n_heads: int,
                 n_gcn: int,
                 alpha: float = 0,
                 beta: float = 0,
                 device: str = 'cpu',
                 epoch: int = 10,
                 batch_size: int = -1,
                 lr: float = 0.0001,
                 contamination=0.1,
                 verbose: bool = False,
                 log_interval: int = 1,
                 **kwargs):
        super().__init__(d_in, d_model, dim_k, dim_v, n_heads, n_gcn, alpha,
                         device, epoch, batch_size, lr, contamination, verbose,
                         log_interval, **kwargs)
        self.beta = beta

    def fit(self, x, mat, evaluate=None):
        x_ = torch.FloatTensor(x).to(self.device)
        mat = torch.FloatTensor(mat).to(self.device)
        self.model = STAnomalyFormer_v2(
            mat.to(self.device),
            n_gcn=self.n_gcn,
            **self.tsfm_args,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if y is not None:
            max_auc = 0.

        process = range(self.epoch) if not self.verbose else tqdm(
            range(self.epoch))
        for epoch in process:
            self.model.train()
            output, td1, td2, score_dy, score_st = self.model(x_)

            recon = torch.square(output - x_).mean((1, 2))
            discrepancy1 = sym_kl_loss(td1, td2.detach()) - sym_kl_loss(
                td1.detach(), td2)
            discrepancy2 = sym_kl_loss(
                score_dy,
                score_st.detach(),
            ) - sym_kl_loss(score_dy.detach(), score_st)
            score = recon + self.alpha * discrepancy2
            loss = score.mean() + self.beta * discrepancy1.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if evaluate is not None:
                output_dict = self.decision_function(evaluate[0], return_dict=True)
                score = output_dict['anomaly_scores']
                # 将二维标签展平为一维
                y_true = evaluate[1].flatten()
                # 确保score的维度与标签匹配
                if isinstance(score, np.ndarray):
                    score = score.flatten()
                else:
                    score = score.reshape(-1)
                print(f"[DEBUG] y_true.shape: {y_true.shape}, score.shape: {score.shape}")
                assert y_true.shape == score.shape, f"标签和分数长度不一致: y_true.shape={y_true.shape}, score.shape={score.shape}"
                auc = roc_auc_score(y_true, score)
                self.early_stopping(auc, self.model)

                if self.is_early_stopping and self.early_stopping.early_stop:
                    break

            if self.verbose:
                if evaluate is not None:
                    process.set_postfix(
                        auc=f"区域AUC: {metrics['region_auc']:.3f}, "
                            f"时间AUC: {metrics['time_auc']:.3f}, "
                            f"综合AUC: {metrics['combined_auc']:.3f}",
                        refresh=True,
                    )
                else:
                    process.set_postfix(loss=f"{loss.item():.5f}", refresh=True)
        self.model.load_state_dict(torch.load(self.early_stopping.path))
        self.model.eval()
        self.decision_scores_ = self.decision_function(x)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )

        return self

    @torch.no_grad()
    def decision_function(self, x):
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        output, score_dy, score_st = self.model(x)
        recon = torch.square(output - x).mean((1, 2))
        discrepancy = sym_kl_loss(score_dy, score_st)
        score = recon + self.alpha * discrepancy
        return score.cpu().numpy()


class STAnomalyFormerDetector_v4(STAnomalyFormerDetector_v3):

    def __init__(
        self,
        d_in: int,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        n_gcn: int,
        device: str = 'cpu',
        epoch: int = 10,
        batch_size: int = -1,
        lr: float = 0.0001,
        contamination=0.1,
        verbose: bool = False,
        log_interval: int = 1,
        threshold_method: str = 'quantile',
        **kwargs,
    ):
        super().__init__(
            d_in,
            d_model,
            dim_k,
            dim_v,
            n_heads,
            n_gcn,
            1.,
            1.,
            device,
            epoch,
            batch_size,
            lr,
            contamination,
            verbose,
            log_interval,
            **kwargs,
        )
        self.threshold_method = threshold_method
        self.loss_weight = torch.ones(3) / 3
        self.model_args = {
            "seq_len": 144,
            "patch_len": 12,
            "stride": 6,
            "d_in": d_in,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_gcn": n_gcn,
            "temporal_half": False,
            "spatial_half": False,
            "static_only": False,
            "dynamic_only": False,
            "temperature": 50.0,
            "anormly_ratio": contamination
        }
        self.region_threshold = None
        self.time_threshold = None

    def calculate_adaptive_threshold(self, scores, contamination=None):
        if contamination is None:
            contamination = self.contamination
            
        scores = scores.detach().cpu()
        
        # 添加调试信息
        if self.verbose:
            print(f"[DEBUG] 计算阈值:")
            print(f"分数形状: {scores.shape}")
            print(f"分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
            print(f"分数均值: {scores.mean():.4f}")
            print(f"分数标准差: {scores.std():.4f}")
        
        # 确保分数不为全0
        if torch.all(scores == 0):
            print("[WARNING] 所有分数都为0，使用默认阈值")
            return torch.tensor(0.5)
        
        if self.threshold_method == 'quantile':
            # 使用更保守的分位数
            threshold = torch.quantile(scores, 1 - contamination * 1.5)
        elif self.threshold_method == 'mean':
            # 使用均值加标准差的方法
            mean = scores.mean()
            std = scores.std()
            threshold = mean + std * (1 - contamination)
        elif self.threshold_method == 'max':
            # 使用最大值的百分比
            max_score = scores.max()
            threshold = max_score * (1 - contamination)
        else:
            raise ValueError(f"不支持的阈值计算方法: {self.threshold_method}")
        
        # 确保阈值不为0
        threshold = max(threshold, scores.mean() * 0.1)
        
        if self.verbose:
            print(f"计算得到的阈值: {threshold:.4f}")
            
        return threshold

    def detect_anomalies(self, region_scores, time_scores):
        """检测异常并返回异常事件和分数"""
        # 确保输入张量的维度正确
        if region_scores.dim() == 3:  # [263, 1, 1]
            region_scores = region_scores.squeeze(-1).squeeze(-1)  # [263]
        
        if time_scores.dim() == 4:  # [263, 23, 14, 1]
            time_scores = time_scores.squeeze(-1)  # [263, 23, 14]
        
        # 获取时间步数
        n_times = time_scores.size(1)
        
        # 初始化异常分数张量
        anomaly_scores = torch.zeros((region_scores.size(0), n_times, 2), device=region_scores.device)
        
        # 扩展区域得分到所有时间步
        anomaly_scores[:, :, 0] = region_scores.unsqueeze(1).expand(-1, n_times)
        
        # 计算时间异常得分
        anomaly_scores[:, :, 1] = time_scores.mean(dim=-1)  # 对天数维度取平均
        
        # 计算阈值
        region_threshold = torch.quantile(region_scores, 1 - self.contamination)
        time_threshold = torch.quantile(time_scores.reshape(-1), 1 - self.contamination)
        
        # 标记异常
        anomaly_events = torch.zeros_like(anomaly_scores, dtype=torch.bool)
        anomaly_events[:, :, 0] = region_scores.unsqueeze(1).expand(-1, n_times) > region_threshold
        anomaly_events[:, :, 1] = time_scores.mean(dim=-1) > time_threshold
        
        return anomaly_events, anomaly_scores

    @torch.no_grad()
    def decision_function(self, x, return_dict=False):
        self.model.eval()
        with torch.no_grad():
            # 确保输入数据在正确的设备上
            x = torch.tensor(x, dtype=torch.float).to(self.device)
            
            # 获取模型输出
            z, time_scores, region_scores = self.model(x)
            
            # 添加数值稳定性检查
            if torch.isnan(region_scores).any() or torch.isnan(time_scores).any():
                print("[WARNING] 检测到NaN值，进行数值稳定性处理")
                region_scores = torch.nan_to_num(region_scores, nan=0.0)
                time_scores = torch.nan_to_num(time_scores, nan=0.0)
            
            # 确保分数在合理范围内
            region_scores = torch.clamp(region_scores, min=0.0, max=10.0)
            time_scores = torch.clamp(time_scores, min=0.0, max=10.0)
            
            # 检测异常
            anomaly_events, anomaly_scores = self.detect_anomalies(region_scores, time_scores)
            
            if return_dict:
                return {
                    'region_scores': region_scores.cpu(),
                    'time_scores': time_scores.cpu(),
                    'anomaly_events': anomaly_events.cpu(),
                    'anomaly_scores': anomaly_scores.cpu()
                }
            else:
                return anomaly_scores.cpu()

    def evaluate(self, x, y_true):
        """评估模型性能"""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y_true, np.ndarray):
            y_true = torch.tensor(y_true, dtype=torch.float).to(self.device)

        # 获取异常分数
        output_dict = self.decision_function(x, return_dict=True)
        anomaly_scores = output_dict['anomaly_scores']  # shape: [N, T_patch, 2]

        # 转换为numpy数组进行评估
        anomaly_scores_np = anomaly_scores.cpu().numpy()
        y_true_np = y_true.cpu().numpy()

        # 打印调试信息
        print("[DEBUG] 输入数据形状:")
        print(f"x shape: {x.shape}")
        print(f"y_true shape: {y_true.shape}")
        print(f"anomaly_scores shape: {anomaly_scores.shape}")

        # 获取patch时间步数
        N, T_patch, _ = anomaly_scores_np.shape
        print(f"[DEBUG] 模型输出的patch时间步数: {T_patch}")

        # 截取标签的前T_patch个时间步
        if len(y_true_np.shape) == 4:  # [N, T, D, 2]
            y_true_np_patch = y_true_np[:, :T_patch, :, :]  # 截取前T_patch个时间步
            # 将标签reshape为[N*T_patch*D, 2]
            y_true_flat = y_true_np_patch.reshape(-1, 2)
            # 将分数reshape为[N*T_patch*D, 2]
            scores_flat = np.repeat(anomaly_scores_np, y_true_np_patch.shape[2], axis=1).reshape(-1, 2)
        else:  # [N, T, 2]
            y_true_np_patch = y_true_np[:, :T_patch, :]
            y_true_flat = y_true_np_patch.reshape(-1, 2)
            scores_flat = anomaly_scores_np.reshape(-1, 2)

        print(f"[DEBUG] 截取后的标签shape: {y_true_np_patch.shape}")
        print(f"[DEBUG] 展平后的标签shape: {y_true_flat.shape}")
        print(f"[DEBUG] 展平后的分数shape: {scores_flat.shape}")

        try:
            # 计算区域异常AUC
            region_auc = roc_auc_score(y_true_flat[:, 0], scores_flat[:, 0])
            # 计算时间异常AUC
            time_auc = roc_auc_score(y_true_flat[:, 1], scores_flat[:, 1])
            # 计算综合AUC
            combined_auc = roc_auc_score(y_true_flat.reshape(-1), scores_flat.reshape(-1))

            metrics = {
                'region_auc': region_auc,
                'time_auc': time_auc,
                'combined_auc': combined_auc
            }

            if self.verbose:
                print(f"区域AUC: {region_auc:.3f}, 时间AUC: {time_auc:.3f}, 综合AUC: {combined_auc:.3f}")

            return metrics

        except Exception as e:
            print(f"评估过程中出现错误: {str(e)}")
            print("错误详情:")
            print(f"y_true shape: {y_true.shape}")
            print(f"anomaly_scores shape: {anomaly_scores.shape}")
            return {
                'region_auc': 0.0,
                'time_auc': 0.0,
                'combined_auc': 0.0
            }

    def fit(self, x, mats, evaluate=None):
        """
        训练模型
        Args:
            x: 输入数据
            mats: 距离矩阵
            evaluate: 评估数据元组 (eval_x, eval_y)
        Returns:
            self
        """
        print("\n" + "="*50)
        print("训练开始前的状态检查:")
        print(f"1. 模型是否存在: {hasattr(self, 'model')}")
        if hasattr(self, 'model'):
            print(f"2. 模型训练状态: {self.model.training}")
            print(f"3. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
        print("="*50 + "\n")
        
        # 数据预处理
        if isinstance(x, np.ndarray):
            x_ = torch.FloatTensor(x).to(self.device)
        else:
            x_ = x.to(self.device)
            
        if isinstance(mats, np.ndarray):
            mats = torch.FloatTensor(mats).to(self.device)
        else:
            mats = mats.to(self.device)
            
        # 初始化早停
        self.early_stopping = EarlyStopping(
            patience=100,
            trace_func=tqdm.write,
            delta=0.01,
        )
        
        # 更新模型参数
        self.model_args.update({
            "seq_len": x_.shape[1],
            "patch_len": self.model_args["patch_len"],
            "stride": self.model_args["stride"],
            "dist_mats": mats
        })
        
        # 初始化模型
        self.model = STPatch_MGCNFormer(**self.model_args).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # 训练循环
        process = range(self.epoch) if not self.verbose else tqdm(range(self.epoch))
        for epoch in process:
            print(f"\n{'='*20} Epoch {epoch} {'='*20}")
            
            # 训练阶段
            print("\n=== 训练阶段开始 ===")
            print("训练阶段开始前的状态检查:")
            print(f"1. 模型训练状态: {self.model.training}")
            print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
            
            # 确保模型处于训练模式
            self.model.train()
            print("\n设置train()后的状态检查:")
            print(f"1. 模型训练状态: {self.model.training}")
            print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
            
            # 前向传播，确保设置return_dict=True
            print("\n前向传播前的状态检查:")
            print(f"1. 模型训练状态: {self.model.training}")
            print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
            
            # 保存原始训练状态
            original_training = self.model.training
            print(f"3. original_training值: {original_training}")
            
            # 确保模型处于训练模式
            self.model.train()
            
            # 前向传播
            output_dict = self.model(x_, return_dict=True)
            print("\n前向传播后的状态检查:")
            print(f"1. 模型训练状态: {self.model.training}")
            print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
            print(f"3. 输出字典键: {output_dict.keys()}")
            
            # 从输出字典中获取重建结果和异常分数
            reconstruction = output_dict['reconstruction']
            anomaly_scores = output_dict['anomaly_scores']
            recon_loss = output_dict['recon_loss']
            
            # 计算总损失
            loss = recon_loss
            
            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print("\n优化后的状态检查:")
            print(f"1. 模型训练状态: {self.model.training}")
            print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
            print(f"3. 当前损失值: {loss.item():.6f}")
            
            # 评估阶段
            if evaluate is not None:
                print("\n=== 评估阶段开始 ===")
                print("评估前的状态检查:")
                print(f"1. 模型训练状态: {self.model.training}")
                print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
                
                # 确保模型处于评估模式
                self.model.eval()
                print("\n设置eval()后的状态检查:")
                print(f"1. 模型训练状态: {self.model.training}")
                print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
                
                with torch.no_grad():
                    # 处理评估数据
                    if isinstance(evaluate[0], np.ndarray):
                        eval_x = torch.FloatTensor(evaluate[0]).to(self.device)
                    else:
                        eval_x = evaluate[0].to(self.device)
                    
                    # 获取评估结果
                    metrics = self.evaluate(eval_x, evaluate[1])
                    auc = (metrics['region_auc'] + metrics['time_auc'] + metrics['combined_auc']) / 3
                    
                    # 早停检查
                    self.early_stopping(auc, self.model)
                    if self.early_stopping.early_stop:
                        if self.verbose:
                            print("触发早停机制")
                        break
                
                print("\n评估后的状态检查:")
                print(f"1. 模型训练状态: {self.model.training}")
                print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
                print(f"3. 评估指标:")
                print(f"   - 区域AUC: {metrics['region_auc']:.3f}")
                print(f"   - 时间AUC: {metrics['time_auc']:.3f}")
                print(f"   - 综合AUC: {metrics['combined_auc']:.3f}")
                
                # 更新进度条
                if self.verbose:
                    process.set_postfix(
                        auc=f"区域AUC: {metrics['region_auc']:.3f}, "
                            f"时间AUC: {metrics['time_auc']:.3f}, "
                            f"综合AUC: {metrics['combined_auc']:.3f}",
                        refresh=True
                    )
            else:
                if self.verbose:
                    process.set_postfix(loss=f"{loss.item():.5f}", refresh=True)
            
            # 确保模型在下一个epoch开始时处于训练模式
            self.model.train()
        
        # 加载最佳模型
        if evaluate is not None:
            print("\n=== 加载最佳模型 ===")
            print("加载前的状态检查:")
            print(f"1. 模型训练状态: {self.model.training}")
            print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
            
            self.model.load_state_dict(torch.load(self.early_stopping.path))
            
            print("\n加载后的状态检查:")
            print(f"1. 模型训练状态: {self.model.training}")
            print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
        
        # 计算最终决策分数
        print("\n=== 计算最终决策分数 ===")
        print("计算前的状态检查:")
        print(f"1. 模型训练状态: {self.model.training}")
        print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
        
        # 确保模型处于评估模式
        self.model.eval()
        output_dict = self.decision_function(x, return_dict=True)
        self.decision_scores_ = output_dict['anomaly_scores']
        
        print("\n训练结束时的状态检查:")
        print(f"1. 模型训练状态: {self.model.training}")
        print(f"2. 模型模式: {'训练模式' if self.model.training else '评估模式'}")
        
        return self

# 在文件末尾添加
__all__ = [
    'TemporalTSFMDetector',
    'SpatialTSFMDetector',
    'DOMINANTDetector',
    'AnomalyDAEDetector',
    'OCGNNDetector',
    'LSTM_AEDetector',
    'STAnomalyFormerDetector_v1',
    'STAnomalyFormerDetector_v2',
    'STAnomalyFormerDetector_v3',
    'STAnomalyFormerDetector_v4',
    'STPatchFormerDetector',
    'STPMFormerDector',
    'STPatch_MGCNDetector'
]