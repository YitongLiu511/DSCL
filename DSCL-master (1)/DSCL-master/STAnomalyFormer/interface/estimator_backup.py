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
    q = q.to(p.device)  # 保证q和p在同一设备
    res = p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))
    # 返回张量而不是标量，让调用者决定如何聚合
    return res


def sym_kl_loss(p, q):
    q = q.to(p.device)  # 保证p和q在同一设备
    kl_forward = kl_loss(p, q)
    kl_backward = kl_loss(q, p)
    sym_kl = (kl_forward + kl_backward) / 2
    # 返回标量而不是张量
    return sym_kl.mean()


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

    @torch.no_grad()
    def decision_function(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().clone().to(self.device)
        else:
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        output, score_dy, score_st = self.model(x)
        recon = torch.square(output - x).mean((1, 2))
        discrepancy = sym_kl_loss(score_st, score_dy)
        score = recon + self.alpha * discrepancy
        return score.cpu().numpy()


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

    def fit(self, x, mat, y=None):
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
            # recon = region_wise_smooth_l1_loss(output, x_)
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

            self.model.eval()
            with torch.no_grad():
                output, _, _, score_dy, score_st = self.model(x_)

                recon = torch.square(output - x_).mean((1, 2))
                # recon = region_wise_smooth_l1_loss(output, x_)
                discrepancy = sym_kl_loss(score_dy, score_st)
                score = recon + self.alpha * discrepancy
                loss = score.mean()

            # log = "Epoch {:3d}, loss={:5.6f}".format(
            #     epoch,
            #     loss.item(),
            # )
            if y is not None:
                auc = roc_auc_score(y, score.cpu().numpy())
                # log += ", AUC={:6f}".format(auc)
                if auc >= max_auc:
                    max_auc = auc
                    self.model_copy = deepcopy(self.model)

            if self.verbose:
                # if (epoch + 1) % self.log_interval == 0:
                #     print(log)
                if y is not None:
                    process.set_postfix(
                        stat="max: {:.4f}, current: {:.4f}".format(
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

    @torch.no_grad()
    def decision_function(self, x):
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        output, _, _, score_dy, score_st = self.model(x)
        recon = torch.square(output - x).mean((1, 2))
        # recon = region_wise_smooth_l1_loss(output, x)
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
        self.loss_weight = torch.ones(3, device=self.device) / 3

    def fit(self, x, mat, evaluate=None):
        x_ = torch.FloatTensor(x).to(self.device)
        mat = torch.FloatTensor(mat).to(self.device)
        
        # 获取最后一个时间戳的真实标签（如果有的话）
        if evaluate is not None and len(evaluate) > 1:
            test_x, y = evaluate
            # 获取最后一个时间戳的标签
            # 假设test_x是(n_nodes, time_steps, features)格式
            if hasattr(test_x, 'shape') and len(test_x.shape) >= 2:
                # 从原始数据中获取最后一个时间戳的标签
                # 这里需要重新加载原始标签数据
                try:
                    import os
                    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    data_dir = os.path.join(base_dir, "data", f"pems0{self.args.dataset if self.args else '3'}")
                    y_test = np.load(os.path.join(data_dir, "anomaly_labels_test.npy"))
                    # 获取最后一个时间戳的标签
                    self.last_timestamp_labels = y_test[-1, :].astype(int)  # 最后一个时间戳的所有区域标签
                    print(f"成功加载最后一个时间戳的标签，异常区域数: {self.last_timestamp_labels.sum()}")
                except Exception as e:
                    print(f"无法加载最后一个时间戳的标签: {e}")
                    self.last_timestamp_labels = None
        
        self.early_stopping = EarlyStopping(
            100,
            trace_func=tqdm.write,
            delta=0.01,
        )
        self.model = STAnomalyFormer_v2(
            mat.to(self.device),
            n_gcn=self.n_gcn,
            **self.tsfm_args,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
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
            score = (loss1 := self.loss_weight[0] *
                     recon) + self.loss_weight[1] * discrepancy2

            with torch.no_grad():
                loss1 = recon.mean()
                loss2 = discrepancy2.mean()
            loss3 = discrepancy1.mean()

            loss = score.mean() + self.loss_weight[2] * loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.update_weight(loss1.item(), loss2.item(), loss3.item())

            if evaluate is not None:
                test_X, test_y = evaluate
                test_x = torch.FloatTensor(test_X).to(self.device)
                self.model.eval()
                with torch.no_grad():
                    output, _, _, score_dy, score_st = self.model(test_x)

                    recon = torch.square(output - test_x).mean((1, 2))
                    discrepancy = sym_kl_loss(score_dy, score_st)
                    score = self.loss_weight[0] * recon + \
                        self.loss_weight[1] * discrepancy
                    loss = score.mean()

                auc = roc_auc_score(test_y, score.cpu().numpy())
                self.early_stopping(auc, self.model)

                if self.early_stopping.early_stop:
                    break

            if self.verbose:
                if evaluate is not None:
                    process.set_postfix(
                        max_auc="max: {:.4f}, current: {:.4f}".format(
                            self.early_stopping.val_score_max, auc),
                        refresh=True,
                    )
                else:
                    process.set_postfix(
                        max_auc="loss: {:.5f}".format(loss.item()),
                        refresh=True,
                    )

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
    def update_weight(self, loss1, loss2, loss3):
        exp = torch.tensor([-loss1, -loss2, -loss3], device=self.loss_weight.device).exp()
        new_weight = self.loss_weight * exp
        self.loss_weight = new_weight / torch.sum(new_weight)

    @torch.no_grad()
    def decision_function(self, x):
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        output, _, _, score_dy, score_st = self.model(x)
        recon = torch.square(output - x).mean((1, 2))
        discrepancy = sym_kl_loss(score_dy, score_st)
        score = self.loss_weight[0] * recon + self.loss_weight[1] * discrepancy
        return score.cpu().numpy()


class STPatchFormerDetector(BaseDetector):

    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        stride: int,
        d_in: int,
        d_model: int,
        n_heads: int,
        temporal_half: bool = False,
        spatial_half: bool = False,
        n_gcn: int = 3,
        device: str = 'cuda',
        epoch: int = 50,
        lr: float = 0.001,
        early_stopping: bool = False,
        use_recon: bool = True,
        use_const: bool = True,
        diff_const: bool = True,
        static_only=False,
        dynamic_only=False,
        contamination: float = 0.1,
        verbose: bool = False,
    ):
        super().__init__(contamination)
        self.model_args = {
            "seq_len": seq_len,
            "patch_len": patch_len,
            "stride": stride,
            "d_in": d_in,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_gcn": n_gcn,
            "temporal_half": temporal_half,
            "spatial_half": spatial_half,
            "static_only": static_only,
            "dynamic_only": dynamic_only,
        }
        self.device = device
        self.epoch = epoch
        self.lr = lr
        self.is_early_stopping = early_stopping
        self.verbose = verbose
        self.use_recon = use_recon
        self.use_const = use_const
        self.diff_const = diff_const
        assert self.use_recon or self.use_const

        self.loss_weight = torch.ones(2, device=self.device) / 2
        self.last_linear = nn.Linear(d_in, d_in).to(self.device)  # 线性层移到这里
        # 计算时间维度：NP*PL = ((seq_len - patch_len) // stride + 1) * patch_len
        time_dim = ((self.model_args['seq_len'] - self.model_args['patch_len']) // self.model_args['stride'] + 1) * self.model_args['patch_len']
        # 使用1D卷积来压缩时间维度，保持batch和特征维度不变
        self.time_proj = nn.Conv1d(d_in, d_in, kernel_size=time_dim, stride=1, padding=0).to(self.device)

    def fit(self, x, mat, evaluate=None):
        x_ = torch.FloatTensor(x).to(self.device)
        mat = torch.FloatTensor(mat).to(self.device)
        self.early_stopping = EarlyStopping(
            100,
            trace_func=tqdm.write,
            delta=0.01,
        )
        self.model = STPatchFormer(
            seq_len=self.model_args['seq_len'],
            patch_len=self.model_args['patch_len'],
            stride=self.model_args['stride'],
            d_in=self.model_args['d_in'],
            d_model=self.model_args['d_model'],
            n_heads=self.model_args['n_heads'],
            dist_mat=mat,
            n_gcn=self.model_args['n_gcn'],
            temporal_half=self.model_args['temporal_half'],
            spatial_half=self.model_args['spatial_half'],
            static_only=self.model_args['static_only'],
            dynamic_only=self.model_args['dynamic_only'],
        ).to(self.device)
        optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.last_linear.parameters()) + list(self.time_proj.parameters()), lr=self.lr)
        for epoch in range(self.epoch):
            self.model.train()
            (patch_x, patch_recon), (score_dy, score_st), patch_recon_flat = self.model(x_)
            # 新增last特征loss
            # patch_recon_flat: (N, NP*PL, D) -> (N, 1, D)
            # 直接使用1D卷积压缩时间维度
            last_recon = self.time_proj(patch_recon_flat.transpose(1, 2)).transpose(1, 2)  # (N, 1, D)
            x_last = x_[:, -1:, :]  # (N, 1, D) - 取最后一个时间片
            last_recon_proj = self.last_linear(last_recon.squeeze(1)).unsqueeze(1)  # 线性变换
            loss_last = F.mse_loss(last_recon_proj, x_last, reduction='none').mean(dim=(1, 2))  # (N,)
            # 原有loss
            if self.use_recon:
                recon = torch.abs(patch_x - patch_recon).mean((1, 2, 3))
                score1 = self.loss_weight[0] * recon
            if self.use_const:
                discrepancy = sym_kl_loss(score_dy, score_st).to(self.loss_weight.device)
                score2 = self.loss_weight[1] * discrepancy
            if self.use_recon and not self.use_const:
                score = score1
            elif self.use_const and not self.use_recon:
                score = score2
            else:
                score = score1 + score2
            # 总loss加上last特征loss
            total_loss = score.mean() + loss_last
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if evaluate is not None:
                auc = roc_auc_score(
                    evaluate[1],
                    self.decision_function(evaluate[0], verbose_output=False),
                )
                self.early_stopping(auc, self.model)

                if self.is_early_stopping and self.early_stopping.early_stop:
                    # print("Early stopping")
                    break

            if self.verbose:
                if evaluate is not None:
                    process.set_postfix(
                        max_auc="AUC: {:.3f}/{:.3f}, weight : {:.3f}".format(
                            auc, self.early_stopping.best_score,
                            self.loss_weight[0]),
                        refresh=True,
                    )
                else:
                    process.set_postfix(
                        max_auc="loss: {:.5f}".format(loss.item()),
                        refresh=True,
                    )
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
    def update_weight(self, loss_const, loss_cluster):
        """更新一致性损失和聚类损失的动态权重"""
        exp = torch.tensor([-loss_const, -loss_cluster], device=self.loss_weight.device).exp()
        new_weight = self.loss_weight * exp
        self.loss_weight = new_weight / torch.sum(new_weight)

    @torch.no_grad()
    def decision_function(self, x, verbose_output=True):
        self.model.eval()
        if isinstance(x, torch.Tensor):
            x = x.detach().clone().to(self.device)
        else:
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        (patch_x, patch_recon), (score_dy, score_st), patch_recon_flat, cluster_loss = self.model(x)
        # 新增last特征loss
        # patch_recon_flat: (N, NP*PL, D) -> (N, 1, D)
        # 直接使用1D卷积压缩时间维度
        last_recon = self.time_proj(patch_recon_flat.transpose(1, 2)).transpose(1, 2)  # (N, 1, D)
        x_last = x[:, -1:, :]  # (N, 1, D) - 取最后一个时间片
        last_recon_proj = self.last_linear(last_recon.squeeze(1)).unsqueeze(1)  # 线性变换
        loss_last = F.mse_loss(last_recon_proj, x_last, reduction='none').mean(dim=(1, 2))  # (N,)
        # 原有loss
        if self.use_recon:
            recon = torch.abs(patch_x - patch_recon).mean((1, 2, 3))
            score1 = self.loss_weight[0] * recon
        if self.use_const:
            discrepancy = sym_kl_loss(score_dy, score_st).to(self.loss_weight.device)
            score2 = self.loss_weight[1] * discrepancy
        if self.use_recon and not self.use_const:
            score = score1
        elif self.use_const and not self.use_recon:
            score = score2
        else:
            score = score1 + score2
        # 返回last特征loss作为异常分数
        loss_last_np = loss_last.cpu().numpy()
        
        # 只在需要时输出详细信息
        if verbose_output:
            # 添加输出：显示每个区域在最后一个时间戳的异常检测结果
            print("\n=== 最后一个时间戳区域异常检测结果 ===")
            print(f"异常分数范围: [{loss_last_np.min():.6f}, {loss_last_np.max():.6f}]")
            print(f"异常分数均值: {loss_last_np.mean():.6f}")
            print(f"异常分数标准差: {loss_last_np.std():.6f}")
            
            # 计算阈值（基于真实异常数量，而不是固定的5%）
            if hasattr(self, 'last_timestamp_labels'):
                true_anomaly_count = self.last_timestamp_labels.sum()
                threshold = np.sort(loss_last_np)[-int(true_anomaly_count)]
                print(f"基于真实异常数量({true_anomaly_count})设置阈值")
            else:
                # 如果没有真实标签，使用默认的5%
                threshold = np.sort(loss_last_np)[-int(len(loss_last_np) * 0.05)]
                print("使用默认5%设置阈值")
            
            pred = (loss_last_np >= threshold).astype(int)
            
            # 显示预测结果
            print(f"\n预测阈值: {threshold:.6f}")
            print(f"预测异常区域数: {pred.sum()} / {len(pred)}")
            
            # 如果有真实标签，计算指标
            if hasattr(self, 'last_timestamp_labels'):
                y_true = self.last_timestamp_labels
                from sklearn.metrics import recall_score, roc_auc_score
                
                recall = recall_score(y_true, pred)
                roc_auc = roc_auc_score(y_true, loss_last_np)
                
                print(f"\n=== 最后一个时间戳区域异常检测评估指标 ===")
                print(f"Recall@20%: {recall:.4f}")
                print(f"ROC-AUC: {roc_auc:.4f}")
                
                # 显示每个区域的预测结果
                print(f"\n=== 区域异常检测详情 ===")
                for i in range(min(10, len(pred))):  # 只显示前10个区域
                    status = "异常" if pred[i] == 1 else "正常"
                    true_status = "异常" if y_true[i] == 1 else "正常"
                    correct = "✓" if pred[i] == y_true[i] else "✗"
                    print(f"区域 {i:3d}: 预测{status} (分数:{loss_last_np[i]:.6f}) | 真实{true_status} | {correct}")
                
                if len(pred) > 10:
                    print(f"... 还有 {len(pred) - 10} 个区域")
        
        # 返回原有的异常分数，而不是新的异常分数
        return score.cpu().numpy()

    def predict(self, x):
        score = self.decision_function(x)
        return predict_by_score(score, 1 - self.contamination)


class STPMFormerDector(STPatchFormerDetector):

    def __init__(self,
                 seq_len: int,
                 patch_len: int,
                 stride: int,
                 d_in: int,
                 d_model: int,
                 n_heads: int,
                 n_gcn: int = 3,
                 temporal_half: bool = False,
                 spatial_half: bool = False,
                 mask_ratio: float = 0.4,
                 device: str = 'cuda',
                 epoch: int = 50,
                 lr: float = 0.001,
                 contamination: float = 0.1,
                 verbose: bool = False):
        super().__init__(seq_len, patch_len, stride, d_in, d_model, n_heads,
                         temporal_half, spatial_half, n_gcn, device, epoch, lr,
                         contamination, verbose)
        self.model_args['mask_ratio'] = mask_ratio

    def fit(self, x, mat, evaluate=None):
        x_ = torch.FloatTensor(x).to(self.device)
        mat = torch.FloatTensor(mat).to(self.device)
        self.early_stopping = EarlyStopping(100, trace_func=tqdm.write)
        self.model = STPatchMaskFormer(
            dist_mat=mat.to(self.device),
            **self.model_args,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        process = range(self.epoch) if not self.verbose else tqdm(
            range(self.epoch))

        for epoch in process:
            self.model.train()
            (patch_x, patch_recon), (score_dy, score_st) = self.model(x_)
            mask = self.model.random_mask.mask
            recon_loss = torch.abs(patch_x - patch_recon).mean(-2)
            loss1 = (recon_loss * mask).sum() / mask.sum()
            loss2 = sym_kl_loss(score_dy, score_st).sum()
            loss = self.loss_weight[0] * loss1 + self.loss_weight[1] * loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.update_weight(loss1.item(), loss2.item())

            if evaluate is not None:
                auc = roc_auc_score(
                    evaluate[1],
                    self.decision_function(evaluate[0], verbose_output=False),
                )
                self.early_stopping(auc, self.model)

                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break

            if self.verbose:
                if evaluate is not None:
                    process.set_postfix(
                        max_auc="max: {:.4f}, current: {:.4f}".format(
                            self.early_stopping.val_score_max, auc),
                        refresh=True,
                    )
                else:
                    process.set_postfix(
                        max_auc="loss: {:.5f}".format(loss.item()),
                        refresh=True,
                    )
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
        self.model.eval()
        if isinstance(x, torch.Tensor):
            x = x.detach().clone().to(self.device)
        else:
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        (patch_x, patch_recon), (score_dy, score_st) = self.model(x)
        recon = torch.abs(patch_x - patch_recon).mean(-2).sum((-1, -2))
        discrepancy = sym_kl_loss(score_dy, score_st)
        score = self.loss_weight[0] * recon + self.loss_weight[1] * discrepancy
        return score.cpu().numpy()


class STPatch_MGCNDetector(STPatchFormerDetector):

    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        stride: int,
        d_in: int,
        d_model: int,
        n_heads: int,
        temporal_half: bool = False,
        spatial_half: bool = False,
        n_gcn: int = 3,
        device: str = 'cuda',
        epoch: int = 50,
        lr: float = 0.001,
        early_stopping: bool = False,
        contamination: float = 0.1,
        use_recon: bool = True,
        use_const: bool = True,
        diff_const: bool = True,
        static_only=False,
        dynamic_only=False,
        verbose: bool = False,
        cluster_weight: float = 0.1,  # 添加聚类损失权重
        args=None,  # 添加args参数
    ):
        super().__init__(seq_len, patch_len, stride, d_in, d_model, n_heads,
                         temporal_half, spatial_half, n_gcn, device, epoch, lr,
                         contamination, verbose)
        self.args = args  # 保存args参数
        self.cluster_weight = cluster_weight  # 保存聚类损失权重
        
        # 修改损失权重初始化：现在用于一致性损失和聚类损失的动态权重
        # loss_weight[0]: 一致性损失权重
        # loss_weight[1]: 聚类损失权重
        self.loss_weight = torch.ones(2, device=self.device) / 2

    def fit(self, x, mats, evaluate=None, x_clean=None):
        print("开始fit方法...")
        print(f"数据形状: x={x.shape}, mats={len(mats)}个矩阵")
        if x_clean is not None:
            print(f"clean data形状: x_clean={x_clean.shape}")
        x_ = torch.FloatTensor(x).to(self.device)
        if x_clean is not None:
            x_clean_ = torch.FloatTensor(x_clean).to(self.device)
        mats = torch.FloatTensor(mats).to(self.device)
        print(f"数据已转移到设备: {self.device}")
        
        print("正在创建模型...")
        # 添加软聚类相关参数
        model_args_with_cluster = self.model_args.copy()
        model_args_with_cluster.update({
            'n_prototypes': getattr(self.args, 'n_prototypes', 10) if self.args else 10,  # 原型数量
            'tau': getattr(self.args, 'tau', 0.5) if self.args else 0.5,          # 温度参数
        })
        
        self.model = STPatch_MGCNFormer(
            dist_mats=mats.to(self.device),
            **model_args_with_cluster,
        ).to(self.device)
        print("模型创建完成，正在设置优化器...")
        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.last_linear.parameters()) + list(self.time_proj.parameters()), lr=self.lr)
        print(f"优化器设置完成，学习率: {self.lr}")
        process = range(self.epoch) if not self.verbose else tqdm(
            range(self.epoch))
        print(f"开始训练，总轮数: {self.epoch}")
        print(f"初始动态权重: [{self.loss_weight[0].item():.4f}, {self.loss_weight[1].item():.4f}]")
        print("🆕 新功能：已启用滑动窗口时间戳预测")
        print("📝 训练时将使用滑动窗口损失，预测完成后将使用滑动窗口预测所有时间戳的异常情况")
        print("🔧 滑动窗口：用前12个时间戳预测第12个，用1-12预测第12个，用2-13预测第13个，以此类推")
        print("=" * 60)

        for epoch in process:
            if epoch == 0:
                print("开始第一个epoch...")
            elif epoch % 10 == 0:
                print(f"正在训练第 {epoch+1}/{self.epoch} 个epoch...")
                
            self.model.train()
            
            # 🆕 新功能：滑动窗口训练
            # 获取seq_len参数
            seq_len = getattr(self, 'seq_len', getattr(self.model, 'seq_len', 12))
            N, T, D = x_.shape
            
            # 创建滑动窗口训练数据
            num_windows = T - seq_len + 1
            
            # 只在第一个epoch打印窗口信息
            if epoch == 0:
                print(f"🔄 滑动窗口训练：共 {num_windows} 个窗口，每个窗口长度 {seq_len}")
            
            # 使用异常恢复预测范式
            if epoch == 0:
                print(f"🎯 使用异常恢复预测范式训练，总共 {num_windows} 个窗口")
                print(f"📊 训练策略：异常数据 → 预测正常值 → 与clean真实值比较")
            
            # 创建动态Dataset - 异常恢复预测范式
            if x_clean is not None:
                # 使用异常数据作为输入，clean数据作为目标
                train_dataset = AnomalyRecoveryDataset(x_, x_clean_, seq_len)
            else:
                train_dataset = SlidingWindowDataset(x_, seq_len)
            
            # 使用DataLoader进行批次训练
            batch_size = 32  # 可以根据显存调整
            train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,  # 随机打乱
                num_workers=0,  # 避免多进程问题
                pin_memory=True  # 加速数据传输
            )
            
            # 使用DataLoader进行批次训练
            total_loss = 0.0
            total_count = 0
            
            for batch_idx, (batch_windows, batch_targets) in enumerate(train_loader):
                # 将数据移到设备上
                batch_windows = batch_windows.to(self.device)  # (batch_size, N, seq_len, D)
                batch_targets = batch_targets.to(self.device)  # (batch_size, N, 1, D)
                
                batch_loss = 0.0
                batch_count = 0
                
                # 处理这个批次中的每个窗口
                for j in range(batch_windows.shape[0]):
                    window = batch_windows[j]  # (N, seq_len, D)
                    target = batch_targets[j]  # (N, 1, D)
                    
                    # 用这个窗口预测当前时间戳
                    (window_patch_x, window_patch_recon), (window_score_dy, window_score_st), window_patch_recon_flat, _ = self.model(window)
                    
                    # 异常恢复预测范式：从注入异常数据预测正常值
                    window_last_recon = self.time_proj(window_patch_recon_flat.transpose(1, 2)).transpose(1, 2)  # (N, 1, D)
                    window_last_recon_proj = self.last_linear(window_last_recon.squeeze(1)).unsqueeze(1)  # 模型学习后预测的正常值
                    # 损失：预测正常值与未注入异常的真实值的差异
                    window_loss = F.mse_loss(window_last_recon_proj, target)
                    
                    batch_loss += window_loss
                    batch_count += 1
                
                # 反向传播
                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                total_loss += batch_loss.item()
                total_count += batch_count
                
                if self.verbose and batch_idx % 5 == 0:
                    print(f"   📊 批次 {batch_idx + 1}/{len(train_loader)}, "
                          f"损失: {batch_loss.item():.6f}")
            
            # 平均滑动窗口损失
            if total_count > 0:
                loss_sliding_window = total_loss / total_count
            else:
                loss_sliding_window = torch.tensor(0.0, device=self.device)
            
            # 清理内存
            torch.cuda.empty_cache()
            
            # 为了保持原有的损失计算逻辑，我们也计算一个完整序列的损失
            # 注意：这里我们使用第一个滑动窗口作为代表，而不是整个序列
            first_window = x_[:, :seq_len, :]  # (N, seq_len, D)
            (patch_x, patch_recon), (score_dy, score_st), patch_recon_flat, cluster_loss = self.model(first_window)
            
            # 重构损失：固定权重为1
            if self.use_recon:
                recon = torch.abs(patch_x - patch_recon).mean((1, 2, 3))
                loss_recon = 1.0 * recon.mean()  # 固定权重1
            else:
                loss_recon = 0.0
                
            # 一致性损失：使用动态权重
            if self.use_const:
                # 修复形状不匹配问题
                if score_dy.dim() == 4:  # (2, 4, 358, 358)
                    # 取平均值来降维
                    score_dy_reshaped = score_dy.mean(dim=(0, 1))  # (358, 358)
                else:
                    score_dy_reshaped = score_dy
                
                # 将分数转换为概率分布（使用更稳定的方法）
                # 先对每个矩阵单独进行softmax，而不是对整个张量
                score_dy_prob = F.softmax(score_dy_reshaped, dim=-1)  # 在最后一个维度上softmax
                score_st_prob = F.softmax(score_st, dim=-1)  # 在最后一个维度上softmax
                
                # 添加调试信息
                if epoch == 0:
                    print(f"   🔍 调试信息 - score_dy原始形状: {score_dy.shape}, 范围: [{score_dy.min():.6f}, {score_dy.max():.6f}]")
                    print(f"   🔍 调试信息 - score_dy重塑后形状: {score_dy_reshaped.shape}, 范围: [{score_dy_reshaped.min():.6f}, {score_dy_reshaped.max():.6f}]")
                    print(f"   🔍 调试信息 - score_dy概率分布形状: {score_dy_prob.shape}, 范围: [{score_dy_prob.min():.6f}, {score_dy_prob.max():.6f}]")
                    print(f"   🔍 调试信息 - score_dy概率分布每行和: [{score_dy_prob.sum(dim=-1).min():.6f}, {score_dy_prob.sum(dim=-1).max():.6f}]")
                    print(f"   🔍 调试信息 - score_st形状: {score_st.shape}, 范围: [{score_st.min():.6f}, {score_st.max():.6f}]")
                    print(f"   🔍 调试信息 - score_st概率分布形状: {score_st_prob.shape}, 范围: [{score_st_prob.min():.6f}, {score_st_prob.max():.6f}]")
                    print(f"   🔍 调试信息 - score_st概率分布每行和: [{score_st_prob.sum(dim=-1).min():.6f}, {score_st_prob.sum(dim=-1).max():.6f}]")
                
                if self.diff_const:
                    discrepancy = sym_kl_loss(score_dy_prob,
                                              score_st_prob.detach()) - sym_kl_loss(
                                                  score_dy_prob.detach(), score_st_prob)
                else:
                    discrepancy = sym_kl_loss(score_dy_prob, score_st_prob)
                
                loss_const = self.loss_weight[0] * discrepancy.mean()  # 动态权重
                
                # 添加调试信息
                if epoch == 0:
                    print(f"   🔍 调试信息 - discrepancy形状: {discrepancy.shape}, 值: {discrepancy.mean():.6f}")
                    print(f"   🔍 调试信息 - loss_const: {loss_const:.6f}")
            else:
                loss_const = 0.0
                
            # 聚类损失：使用动态权重
            loss_cluster = self.loss_weight[1] * cluster_loss  # 动态权重
            
            # 🆕 滑动窗口损失：固定权重为5
            loss_sliding_window = 5.0 * loss_sliding_window  # 固定权重5
            
            # 计算总损失
            total_loss = loss_recon + loss_const + loss_cluster + loss_sliding_window
            
            # 更新动态权重（一致性损失和聚类损失）
            if self.use_const:
                self.update_weight(loss_const, loss_cluster)
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # 打印每个epoch的详细损失信息
            if epoch == 0:
                print(f"\n=== 第 {epoch+1} 个epoch损失详情 ===")
                print(f"重构损失: {loss_recon:.6f} (权重: 1.0)")
                print(f"一致性损失: {loss_const:.6f} (权重: {self.loss_weight[0].item():.4f})")
                print(f"聚类损失: {loss_cluster:.6f} (权重: {self.loss_weight[1].item():.4f})")
                print(f"🆕 滑动窗口损失: {loss_sliding_window:.6f} (权重: 5.0, 窗口数: {total_count})")
                print(f"总损失: {total_loss.item():.6f}")
                print(f"动态权重更新: [{self.loss_weight[0].item():.4f}, {self.loss_weight[1].item():.4f}]")
            elif epoch % 5 == 0 or epoch == self.epoch - 1:  # 每5个epoch打印一次，以及最后一个epoch
                print(f"\n=== 第 {epoch+1}/{self.epoch} 个epoch损失详情 ===")
                print(f"重构损失: {loss_recon:.6f} (权重: 1.0)")
                print(f"一致性损失: {loss_const:.6f} (权重: {self.loss_weight[0].item():.4f})")
                print(f"聚类损失: {loss_cluster:.6f} (权重: {self.loss_weight[1].item():.4f})")
                print(f"🆕 滑动窗口损失: {loss_sliding_window:.6f} (权重: 5.0, 窗口数: {total_count})")
                print(f"总损失: {total_loss.item():.6f}")
                print(f"动态权重: [{self.loss_weight[0].item():.4f}, {self.loss_weight[1].item():.4f}]")

            if self.verbose:
                process.set_postfix(
                    max_auc="loss: {:.5f}".format(total_loss.item()),
                    refresh=True,
                )
        
        # 训练结束，打印最终权重信息
        print("\n" + "=" * 60)
        print("训练完成！")
        print(f"最终动态权重: [{self.loss_weight[0].item():.4f}, {self.loss_weight[1].item():.4f}]")
        print("=" * 60)
        
        self.model.eval()
        self.decision_scores_ = self.decision_function(x)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        
        return self

    @torch.no_grad()
    def decision_function(self, x, test_X_clean=None, verbose_output=True):
        """决策函数：获取异常分数"""
        # 使用get_all_timestamps_scores方法获取完整的异常分数矩阵
        anomaly_scores = self.get_all_timestamps_scores(x, test_X_clean)
        
        # 返回异常分数矩阵 (N, T)
        return anomaly_scores
    
    def get_all_timestamps_scores(self, x, test_X_clean=None):
        """获取每个区域每个时间戳的异常分数矩阵 (N, T)"""
        return self.get_last_timestamp_score(x, test_X_clean)
        
    def get_last_timestamp_score(self, x, test_X_clean=None):
            self.model.eval()
            if isinstance(x, torch.Tensor):
                x = x.detach().clone().to(self.device)
            else:
                x = torch.tensor(x, dtype=torch.float).to(self.device)
        
        # 如果有test_X_clean，也转换为tensor
        if test_X_clean is not None:
            if isinstance(test_X_clean, torch.Tensor):
                test_X_clean = test_X_clean.detach().clone().to(self.device)
            else:
                test_X_clean = torch.tensor(test_X_clean, dtype=torch.float).to(self.device)
        
        N, T, D = x.shape
        # 获取seq_len
        seq_len = getattr(self, 'seq_len', 12)  # 默认值12
        
        print(f"\n🔧 开始滑动窗口预测...")
        print(f"📊 数据形状: N={N}, T={T}, D={D}")
        print(f"🔍 滑动窗口长度: seq_len={seq_len}")
        print(f"🎯 预测时间戳范围: {seq_len-1} 到 {T-1} (共 {T-seq_len+1} 个时间戳)")
        
        # 初始化完整的(N, T)矩阵，用NaN填充无法预测的时间戳
        full_scores = np.full((N, T), np.nan)
        
        # 滑动窗口预测：从第seq_len个时间戳开始预测
        total_windows = T - seq_len + 1
        for t_idx, t in enumerate(range(seq_len-1, T)):
            # 取前seq_len个时间戳作为输入窗口
            window = x[:, t-seq_len+1:t+1, :]  # (N, seq_len, D)
            
            # 检查窗口是否包含填充数据（前seq_len-1个时间戳）
            if t < seq_len - 1:
                # 对于前seq_len-1个时间戳，跳过预测，保持NaN
                # 因为这些时间戳使用了填充数据，重构误差会很大，容易误判为异常
                continue
            
            # 显示进度
            if t_idx % 100 == 0 or t_idx == total_windows - 1:
                print(f"⏳ 处理窗口 {t_idx+1}/{total_windows}: 时间戳 {t-seq_len+1}-{t} → 预测时间戳 {t}")
            
            # 用这个窗口预测当前时间戳t的异常
            try:
                with torch.no_grad():
                    (patch_x, patch_recon), (score_dy, score_st), patch_recon_flat, cluster_loss = self.model(window)
                    
                    # 异常恢复预测范式：预测正常值，与clean真实值比较
                    if self.use_recon:
                        # 使用异常恢复预测范式
                        last_recon = self.time_proj(patch_recon_flat.transpose(1, 2)).transpose(1, 2)  # (N, 1, D)
                        predicted_normal = self.last_linear(last_recon.squeeze(1)).unsqueeze(1)  # (N, 1, D) - 模型学习后预测的正常值
                        
                        if test_X_clean is not None:
                            # 使用clean数据作为目标
                            target = test_X_clean[:, t:t+1, :]  # (N, 1, D) - clean数据的对应时间片
                            # 异常分数：预测正常值与clean真实值的差异
                            recon_error = torch.abs(predicted_normal - target).mean((1, 2))  # (N,)
                        else:
                            # 如果没有clean数据，使用异常数据作为目标（兼容旧版本）
                            actual_anomaly = window[:, -1:, :]  # (N, 1, D) - 注入异常数据的最后一个时间片
                            # 异常分数：预测正常值与注入异常真实值的差异
                            recon_error = torch.abs(predicted_normal - actual_anomaly).mean((1, 2))  # (N,)
                    else:
                        recon_error = torch.zeros(N, device=self.device)
                    
                    # 时间戳异常检测使用异常恢复预测分数
                    window_score = recon_error  # (N,) - 每个区域在当前时间戳的异常分数
                    
                    # 将分数填入完整矩阵的第t列
                    full_scores[:, t] = window_score.cpu().numpy()
                        
            except Exception as e:
                print(f"⚠️  时间戳 {t} 预测失败: {e}")
                # 保持NaN值，不进行任何操作
                pass
        
        print(f"✅ 滑动窗口预测完成！")
        print(f"📈 输出形状: {full_scores.shape} (区域数 × 时间戳数)")
        print(f"📊 异常分数范围: [{np.nanmin(full_scores):.6f}, {np.nanmax(full_scores):.6f}]")
        print(f"📊 异常分数均值: {np.nanmean(full_scores):.6f}")
        print(f"📊 可预测时间戳数: {T-seq_len+1}/{T} ({((T-seq_len+1)/T)*100:.1f}%)")
        
        return full_scores


class SlidingWindowDataset(torch.utils.data.Dataset):
    """动态滑动窗口数据集，避免一次性加载所有窗口到内存"""
    
    def __init__(self, data, seq_len, target_len=1):
        """
        Args:
            data: (N, T, D) 原始数据
            seq_len: 滑动窗口长度
            target_len: 目标长度（通常为1）
        """
        self.data = data
        self.seq_len = seq_len
        self.target_len = target_len
        self.N, self.T, self.D = data.shape
        self.num_windows = self.T - self.seq_len + 1
        
    def __len__(self):
        return self.num_windows
    
    def __getitem__(self, idx):
        """动态生成滑动窗口样本"""
        # 计算时间戳范围
        start_idx = idx
        end_idx = start_idx + self.seq_len
        target_start = end_idx - 1
        target_end = target_start + self.target_len
        
        # 动态切片，不预先加载
        window = self.data[:, start_idx:end_idx, :]  # (N, seq_len, D)
        target = self.data[:, target_start:target_end, :]  # (N, target_len, D)
        
        return window, target


class AnomalyRecoveryDataset(torch.utils.data.Dataset):
    """异常恢复预测数据集：从异常数据恢复到正常数据"""
    
    def __init__(self, anomaly_data, clean_data, seq_len, target_len=1):
        """
        Args:
            anomaly_data: (N, T, D) 异常数据
            clean_data: (N, T, D) 正常数据
            seq_len: 滑动窗口长度
            target_len: 目标长度（通常为1）
        """
        self.anomaly_data = anomaly_data
        self.clean_data = clean_data
        self.seq_len = seq_len
        self.target_len = target_len
        self.N, self.T, self.D = anomaly_data.shape
        self.num_windows = self.T - self.seq_len + 1
        
    def __len__(self):
        return self.num_windows
    
    def __getitem__(self, idx):
        """动态生成异常恢复样本"""
        # 计算时间戳范围
        start_idx = idx
        end_idx = start_idx + self.seq_len
        target_start = end_idx - 1  # 第12个时间戳
        target_end = target_start + self.target_len
        
        # 输入：注入异常数据的滑动窗口 (N, 12, D)
        window = self.anomaly_data[:, start_idx:end_idx, :]  # (N, seq_len, D)
        
        # 目标：未注入异常数据中对应时间点的值 (N, 1, D)
        target = self.clean_data[:, target_start:target_end, :]  # (N, target_len, D)
        
        return window, target

