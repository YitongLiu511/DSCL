import numpy as np
import torch
from typing import Callable, Optional

# 对单个窗口计算该窗口最后时刻 t 的每节点异常分数，返回 (N,)
WindowScoreFn = Callable[[torch.Tensor, int], torch.Tensor]

def stpatch_recovery_scorer(detector, test_X_clean: Optional[np.ndarray] = None) -> WindowScoreFn:
    """
    适配 STPatch 系列 detector，返回窗口打分函数：
    - model(window) -> patch_recon_flat
    - time_proj -> last_linear -> predicted_normal (N,1,D)
    - reference: test_X_clean[:, t:t+1, :] 若存在，否则 window[:, -1:, :]
    - recon_error = mean(|predicted_normal - reference|, dim=(1,2)) -> (N,)
    """
    device = getattr(detector, 'device', 'cuda')

    # 预处理参考数据（若提供）
    if test_X_clean is not None and isinstance(test_X_clean, np.ndarray):
        # 统一到 (N, T, D)
        # 调用方需保证与 x 一致；此处仅存引用，按 t 切片时再转 Tensor
        clean_ref = test_X_clean
    else:
        clean_ref = None

    def score_fn(window: torch.Tensor, t: int) -> torch.Tensor:
        # window: (N, seq_len, D)
        if not getattr(detector, 'use_recon', True):
            return torch.zeros(window.shape[0], device=device)

        with torch.no_grad():
            (patch_x, patch_recon), (score_dy, score_st), patch_recon_flat, cluster_loss = detector.model(window)
            last_recon = detector.time_proj(patch_recon_flat.transpose(1, 2)).transpose(1, 2)  # (N, 1, D)
            predicted_normal = detector.last_linear(last_recon.squeeze(1)).unsqueeze(1)  # (N, 1, D)

            if clean_ref is not None:
                ref = torch.tensor(clean_ref[:, t:t+1, :], dtype=torch.float, device=device)
            else:
                ref = window[:, -1:, :]  # (N, 1, D)

            recon_error = torch.abs(predicted_normal - ref).mean(dim=(1, 2))  # (N,)
            return recon_error

    return score_fn


def compute_timestamp_scores(
    x: np.ndarray | torch.Tensor,
    seq_len: int,
    batch_size: int,
    window_score_fn: WindowScoreFn,
    device: str = 'cuda',
    verbose: bool = True,
) -> np.ndarray:
    """
    计算每个区域每个时间戳的异常分数矩阵 (N, T)。t < seq_len 的列填 NaN。
    """
    # 规范输入
    if isinstance(x, torch.Tensor):
        x_t = x.detach().clone().to(device)
        N, T, D = x_t.shape
    else:
        x_t = torch.tensor(x, dtype=torch.float, device=device)
        N, T, D = x.shape

    if verbose:
        print("\n🔧 开始滑动窗口预测...")
        print(f"📊 数据形状: N={N}, T={T}, D={D}")
        print(f"🔍 滑动窗口长度: seq_len={seq_len}")
        print(f"🎯 预测时间戳范围: {seq_len-1} 到 {T-1} (共 {T-seq_len+1} 个时间戳)")
        print(f"⚡ 批次大小: {batch_size} (每{batch_size}个窗口显示一次进度)")

    # 结果矩阵
    full_scores = np.full((N, T), np.nan, dtype=np.float32)

    total_windows = T - seq_len
    if total_windows <= 0:
        return full_scores

    batch_count = 0
    for batch_start in range(seq_len, T, batch_size):
        batch_end = min(batch_start + batch_size, T)
        batch_timestamps = list(range(batch_start, batch_end))

        batch_count += 1
        total_batches = (total_windows + batch_size - 1) // batch_size
        if verbose:
            print(f"⏳ 处理批次 {batch_count}/{total_batches}: 时间戳 {batch_start}-{batch_end-1} (共{len(batch_timestamps)}个窗口)")

        # 逐窗口处理，避免维度问题
        for t in batch_timestamps:
            try:
                window = x_t[:, t-seq_len:t, :]  # (N, seq_len, D)
                scores_t = window_score_fn(window, t)  # (N,)
                full_scores[:, t] = scores_t.detach().cpu().numpy()
            except Exception as e:
                print(f"⚠️  时间戳 {t} 预测失败: {e}")
                # 保持 NaN
                continue

    if verbose:
        print("✅ 滑动窗口预测完成！")
        print(f"📈 输出形状: {full_scores.shape} (区域数 × 时间戳数)")
        try:
            print(f"📊 异常分数范围: [{np.nanmin(full_scores):.6f}, {np.nanmax(full_scores):.6f}]")
            print(f"📊 异常分数均值: {np.nanmean(full_scores):.6f}")
        except Exception:
            pass
        print(f"📊 可预测时间戳数: {T-seq_len+1}/{T} ({((T-seq_len+1)/T)*100:.1f}%)")
        print(f"⚡ 进度显示优化: 每{batch_size}个窗口显示一次进度")

    return full_scores 


def dcd_style_kl_scorer(
    detector=None,
    patch_len: int = 5,
    stride: int = 1,
    temperature: float = 50.0,
    weighting: str = "uniform",   # or "triangle"
    invert_softmax: bool = True,   # True → softmax(-A/τ)，与原DCdetector符号方向一致
    use_raw_score: bool = True,    # True → 用未归一 raw_A，False → 用 softmax 分数
) -> WindowScoreFn:
    """
    DCdetector 风格的窗口打分器：
    - 对每个节点的窗口 (seq_len, D) 做重叠分片（长度 patch_len，步长 stride）
    - 计算 in-patch 与 patch-wise 两视角的概率分布
    - 用对称 KL 得到每个 patch 分数，并覆盖聚合成时间戳级分数 A_t
    - 返回窗口最后一个时间戳（t=seq_len-1）的分数（每节点一个值）
    """
    device = getattr(detector, 'device', 'cuda') if detector is not None else 'cuda'

    def _np_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = x - np.max(x, axis=axis, keepdims=True)
        ex = np.exp(x)
        return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-12)

    def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + eps
        return x / norms

    def _cosine_sim_matrix(X_: np.ndarray) -> np.ndarray:
        Xn = _l2_normalize_rows(X_)
        return np.clip(Xn @ Xn.T, -1.0, 1.0)

    def _kl_sym(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
        p = p.astype(np.float64)
        q = q.astype(np.float64)
        p = p / (p.sum() + eps)
        q = q / (q.sum() + eps)
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        kl_pq = float(np.sum(p * (np.log(p) - np.log(q))))
        kl_qp = float(np.sum(q * (np.log(q) - np.log(p))))
        return 0.5 * (kl_pq + kl_qp)

    def _compute_window_scores_np(
        X: np.ndarray,
        patch_len_: int,
        stride_: int,
        temperature_: float,
        weighting_: str,
        invert_softmax_: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        输入 X: (T, D)
        返回 (scores, raw_A)，长度均为 T
        """
        T, D = X.shape
        p = int(patch_len_)
        s = int(stride_)
        if T < p:
            return np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)

        starts = list(range(0, T - p + 1, s))
        K = len(starts)
        idx_pairs = [(st, st + p) for st in starts]

        # in-patch 概率 P_k（长度 p）
        P_list: list[np.ndarray] = []
        for st, ed in idx_pairs:
            Xk = X[st:ed]                      # (p, D)
            S_pos = _cosine_sim_matrix(Xk)     # (p, p)
            Pk_rows = _np_softmax(S_pos, axis=1)
            Pk = Pk_rows.mean(axis=0)          # (p,)
            P_list.append(Pk)

        # patch-wise 概率 W（K,） → 上采样到时间轴，再切片为 N_k
        E = np.stack([X[st:ed].mean(axis=0) for st, ed in idx_pairs], axis=0)  # (K, D)
        S_patch = _cosine_sim_matrix(E)                                        # (K, K)
        W_rows = _np_softmax(S_patch, axis=1)
        W = W_rows.mean(axis=0)                                                # (K,)

        time_weights = np.zeros(T, dtype=np.float64)
        for i, (st, ed) in enumerate(idx_pairs):
            time_weights[st:ed] += W[i] / p
        if time_weights.sum() <= 0:
            time_weights[:] = 1.0 / max(1, T)

        N_list: list[np.ndarray] = []
        for st, ed in idx_pairs:
            Nk = time_weights[st:ed].copy()
            ssum = Nk.sum()
            if ssum <= 0:
                Nk[:] = 1.0 / p
            else:
                Nk /= ssum
            N_list.append(Nk)

        # 对称 KL 得到每个 patch 的分数
        patch_scores = np.array([_kl_sym(P_list[i], N_list[i]) for i in range(K)], dtype=np.float64)

        # 覆盖聚合 → 时间戳级原始分数 raw_A
        raw_A = np.zeros(T, dtype=np.float64)
        raw_W = np.zeros(T, dtype=np.float64)
        for i, (st, ed) in enumerate(idx_pairs):
            if weighting_ == "triangle" and p > 1:
                center = (st + ed - 1) / 2.0
                w = 1.0 - (np.abs(np.arange(st, ed) - center) / ((p - 1) / 2.0))
                w = np.clip(w, 0.0, 1.0)
            else:
                w = np.ones(ed - st, dtype=np.float64)
            raw_A[st:ed] += patch_scores[i] * w
            raw_W[st:ed] += w
        mask = raw_W > 0
        raw_A[mask] = raw_A[mask] / raw_W[mask]

        # 能量化（softmax 归一）
        tau = max(1e-12, temperature_)
        if invert_softmax_:
            scores = _np_softmax(-raw_A / tau, axis=0)
        else:
            scores = _np_softmax(raw_A / tau, axis=0)

        return scores.astype(np.float32), raw_A.astype(np.float32)

    def score_fn(window: torch.Tensor, t: int) -> torch.Tensor:
        # window: (N, seq_len, D)，返回 (N,)（最后一个时间戳的分数）
        assert window.ndim == 3, "window 应为 (N, seq_len, D)"
        N, T, D = window.shape
        out = np.zeros(N, dtype=np.float32)
        w_cpu = window.detach().cpu().numpy()
        for n in range(N):
            Xn = w_cpu[n]  # (T, D)
            scores_n, raw_A_n = _compute_window_scores_np(
                X=Xn,
                patch_len_=patch_len,
                stride_=stride,
                temperature_=temperature,
                weighting_=weighting,
                invert_softmax_=invert_softmax,
            )
            out[n] = raw_A_n[-1] if use_raw_score else scores_n[-1]
        return torch.tensor(out, dtype=torch.float, device=device)

    return score_fn 