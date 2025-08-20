import torch
from torch import nn
import math

from .tsfm import TemporalTransformer


def random_masking(xb, mask_ratio):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(
        bs,
        L,
        nvars,
        device=xb.device,
    )  # noise in [0, 1], bs x L x nvars

    # sort noise for each sample
    ids_shuffle = torch.argsort(
        noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle,
                                dim=1)  # ids_restore: [bs x L x nvars]

    # keep the first subset
    ids_keep = ids_shuffle[:, :
                           len_keep, :]  # ids_keep: [bs x len_keep x nvars]
    x_kept = torch.gather(
        x,
        dim=1,
        index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D),
    )  # x_kept: [bs x len_keep x nvars  x patch_len]

    # removed x
    x_removed = torch.zeros(
        bs,
        L - len_keep,
        nvars,
        D,
        device=xb.device,
    )  # x_removed: [bs x (L-len_keep) x nvars x patch_len]
    x_ = torch.cat(
        [x_kept, x_removed],
        dim=1,
    )  # x_: [bs x L x nvars x patch_len]

    # combine the kept part and the removed one
    x_masked = torch.gather(
        x_,
        dim=1,
        index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, D),
    )  # x_masked: [bs x num_patch x nvars x patch_len]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones(
        [bs, L, nvars],
        device=x.device,
    )  # mask: [bs x num_patch x nvars]
    mask[:, :len_keep, :] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(
        mask,
        dim=1,
        index=ids_restore,
    )  # [bs x num_patch x nvars]
    return x_masked, x_kept, mask, ids_restore


def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty(
            (q_len, d_model)
        )  # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'sincos':
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(
            f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)


class Patch(nn.Module):

    def __init__(self, seq_len, patch_len, stride):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
        tgt_len = patch_len + stride * (self.num_patch - 1)
        self.s_begin = seq_len - tgt_len

    def forward(self, x):
        """
        x: [bs x seq_len x n_vars]
        """
        # 动态计算s_begin，适应不同长度的输入
        actual_seq_len = x.shape[1]
        
        # 🆕 修复：确保patch_len不会大于seq_len
        effective_patch_len = min(self.patch_len, actual_seq_len)
        effective_stride = min(self.stride, actual_seq_len)
        
        # 如果patch_len大于seq_len，调整策略
        if self.patch_len > actual_seq_len:
            print(f"   ⚠️  patch_len({self.patch_len}) > seq_len({actual_seq_len})，调整为: patch_len={effective_patch_len}, stride={effective_stride}")
        
        # 计算目标长度和起始位置
        tgt_len = effective_patch_len + effective_stride * (self.num_patch - 1)
        s_begin = max(0, actual_seq_len - tgt_len)
        
        # 确保至少有一个patch
        if tgt_len > actual_seq_len:
            effective_stride = max(1, (actual_seq_len - effective_patch_len) // (self.num_patch - 1)) if self.num_patch > 1 else 1
            tgt_len = effective_patch_len + effective_stride * (self.num_patch - 1)
            s_begin = max(0, actual_seq_len - tgt_len)
            print(f"   🔧  调整stride为: {effective_stride}，目标长度: {tgt_len}")
        
        x = x[:, s_begin:, :]
        
        # 使用调整后的参数进行unfold
        x = x.unfold(
            dimension=1,
            size=effective_patch_len,
            step=effective_stride,
        )  # xb: [bs x num_patch x n_vars x effective_patch_len]
        
        return x


class PatchEncoder(nn.Module):

    def __init__(
        self,
        c_in,
        num_patch,
        patch_len,
        d_model=128,
        n_heads=16,
        d_ff=256,
        shared_embedding=True,
        attn_dropout=0.,
        dropout=0.,
        half: bool = False,
        pe='zeros',
        learn_pe=True,
        use_inpatch_attn: bool = False,
        inpatch_pe: str = 'sincos',
    ):

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding
        self.n_heads = n_heads

        # 选项1：使用 in-patch 注意力将 (patch_len) 序列编码成 d_model
        self.use_inpatch_attn = use_inpatch_attn
        if self.use_inpatch_attn:
            # 将每个时间步标量嵌入到 d_model，并在 patch_len 维度上做 Transformer，再池化
            self.inpatch_value_embed = nn.Linear(1, d_model)
            self.inpatch_W_pos = positional_encoding(
                inpatch_pe,
                True,
                patch_len,
                d_model,
            )
            # 开启返回注意力权重，以便上游提取每个patch内的时间戳注意力
            self.inpatch_encoder = TemporalTransformer(
                d_model=d_model,
                dim_k=d_model // n_heads,
                dim_v=d_model // n_heads,
                n_heads=n_heads,
                dim_fc=d_ff,
                dropout=attn_dropout,
                half=half,
                return_attn=True,
            )
            # 用于缓存最近一次forward的patch内注意力 (B', H, PL, PL)（注意：可能在CPU上）
            self.last_inpatch_attn = None
            # 微批大小：控制 in-patch 注意力在 batch 维度上的分块，避免一次性在 GPU 上产生巨型张量
            # 可按需调整；默认 2048 在 24GB 显存上较稳妥
            self.inpatch_batch_size = 2048
        else:
            # 选项2：旧实现，直接把整个 patch 向量线性映射到 d_model
            # Input encoding: projection of feature vectors onto a d-dim vector space
            if not shared_embedding:
                self.W_P = nn.ModuleList()
                for _ in range(self.n_vars):
                    self.W_P.append(nn.Linear(
                        patch_len,
                        d_model,
                    ))
            else:
                self.W_P = nn.Linear(patch_len, d_model)

        # Positional encoding
        self.W_pos = positional_encoding(
            pe,
            learn_pe,
            num_patch,
            d_model,
        )

        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        self.encoder = TemporalTransformer(
            d_model=d_model,
            dim_k=d_model // n_heads,
            dim_v=d_model // n_heads,
            n_heads=n_heads,
            dim_fc=d_ff,
            dropout=attn_dropout,
            half=half,
            return_attn=False,
        )

    def forward(self, x):
        bs, num_patch, n_vars, patch_len = x.shape

        if self.use_inpatch_attn:
            # 1) in-patch 自注意力：对每个 patch 的时间步做编码并池化为 d_model
            xb = x.transpose(1, 2)  # [bs, n_vars, num_patch, patch_len]
            bs_, n_vars_, num_patch_, patch_len_ = xb.shape
            Bp = bs_ * n_vars_ * num_patch_  # 展平后的 batch 大小
            tokens = xb.reshape(Bp, patch_len_, 1)
            
            # 🆕 内存优化：动态调整子批次大小，避免显存峰值
            # 根据patch数量和序列长度动态调整
            if num_patch_ <= 100:
                sub_bs = min(2048, Bp)
            elif num_patch_ <= 300:
                sub_bs = min(1024, Bp)
            elif num_patch_ <= 500:
                sub_bs = min(512, Bp)
            else:
                sub_bs = min(256, Bp)
            
            # 如果显存不足，进一步减小批次
            try:
                torch.cuda.empty_cache()
                # 测试显存是否足够
                test_tensor = torch.randn(sub_bs, patch_len_, 1, device=x.device)
                del test_tensor
            except RuntimeError:
                # 显存不足，减小批次大小
                sub_bs = max(64, sub_bs // 2)
                print(f"   ⚠️  显存不足，调整子批次大小为: {sub_bs}")
            
            outs_h = []
            outs_attn_cpu = []
            for start in range(0, Bp, sub_bs):
                end = min(start + sub_bs, Bp)
                t_i = tokens[start:end]
                h_i = self.inpatch_value_embed(t_i)               # [b, PL, d_model]
                h_i = h_i + self.inpatch_W_pos                    # 位置编码
                h_i = self.dropout(h_i)
                out_i = self.inpatch_encoder(h_i)
                if isinstance(out_i, tuple):
                    h_i, attn_i = out_i  # attn_i: [b, H, PL, PL]
                    # 统一放在CPU，避免在GPU上长期保存大张量
                    if isinstance(attn_i, torch.Tensor) and attn_i.device.type != 'cpu':
                        attn_i = attn_i.detach().cpu()
                    outs_attn_cpu.append(attn_i)
                outs_h.append(h_i)
                # 及时释放显存碎片
                del t_i, h_i, out_i
                torch.cuda.empty_cache()
            h = torch.cat(outs_h, dim=0)
            if len(outs_attn_cpu) > 0:
                self.last_inpatch_attn = torch.cat(outs_attn_cpu, dim=0)
                # 记录形状，便于上游还原为 [bs, n_vars, num_patch, H, PL, PL]
                self.last_inpatch_shape = (bs_, n_vars_, num_patch_, patch_len_)
            # 清理列表以释放引用
            del outs_h, outs_attn_cpu
            h = h.mean(dim=1)                                     # [B', d_model]
            x_patch = h.reshape(bs_, n_vars_, num_patch_, self.d_model)
            # 2) 分片间注意力：在 num_patch 维度上做 Transformer（保持原有设计）
            u = x_patch.reshape(bs_ * n_vars_, num_patch_, self.d_model)
            u = self.dropout(u + self.W_pos)
            # 🆕 内存优化：当 num_patch 很大时使用更激进的子批处理
            if num_patch_ >= 256:
                # 轻量化编码：按子批在NP维做块状编码，避免一次显存峰值
                step = max(64, num_patch_ // 8)  # 动态调整步长
                parts = []
                for s in range(0, num_patch_, step):
                    e = min(s + step, num_patch_)
                    parts.append(self.encoder(u[:, s:e, :]))
                    # 及时清理中间结果
                    if len(parts) > 1:
                        torch.cuda.empty_cache()
                z = torch.cat(parts, dim=1)
                del parts
                torch.cuda.empty_cache()
            else:
                z = self.encoder(u)                               # [bs*n_vars, NP, d_model]
            x_out = z.reshape(bs_, n_vars_, num_patch_, self.d_model)
            # 缓存最近一次的 patch 表示（放CPU，避免占用显存）供 get_attention_weights 使用
            try:
                self.last_x_patch = x_out.detach().cpu()
            except Exception:
                self.last_x_patch = None
            return x_out
        else:
            # 旧路径：直接将 patch 向量线性映射到 d_model，再做分片间注意力
            if not self.shared_embedding:
                x_out = []
                for i in range(n_vars):
                    z = self.W_P[i](x[:, :, i, :])
                    x_out.append(z)
                x = torch.stack(x_out, dim=2)
            else:
                x = self.W_P(x)  # [bs, num_patch, n_vars, d_model]
            x = x.transpose(1, 2)                                 # [bs, n_vars, num_patch, d_model]
            u = x.reshape(bs * n_vars, num_patch, self.d_model)   # [bs*n_vars, NP, d_model]
            u = self.dropout(u + self.W_pos)
            z = self.encoder(u)                                   # [bs*n_vars, NP, d_model]
            return z.reshape((-1, n_vars, num_patch, self.d_model))
        # z: [bs x nvars x d_model x num_patch]


class RandomMasking(nn.Module):

    def __init__(self, mask_ratio: float = 0.4) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, xb):
        if self.training:
            x_masked, _, mask, _ = random_masking(xb, self.mask_ratio)
            self.mask = mask.bool()
        else:
            x_masked = xb
        return x_masked
