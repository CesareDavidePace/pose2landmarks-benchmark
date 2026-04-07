import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _require_mamba():
    try:
        from mamba_ssm import Mamba
    except ImportError as exc:
        if "undefined symbol" in str(exc) or "selective_scan_cuda" in str(exc):
            raise ImportError(
                "mamba_ssm has CUDA/PyTorch version mismatch. "
                "This is a known issue with PyTorch 2.5.0. "
                "Models using Mamba will be skipped."
            ) from exc
        raise ImportError("mamba_ssm not installed. Install with `pip install mamba-ssm`") from exc
    return Mamba


def _require_s4():
    try:
        from s4.models.s4 import S4Block
        return S4Block
    except Exception:
        try:
            from s4 import S4
            return S4
        except Exception as exc:
            raise ImportError("s4 not installed. Install a compatible S4 package.") from exc


class TemporalConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, drop_rate=0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop_rate)
        self.residual = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        y = self.drop(self.act(self.norm(self.conv(x))))
        return y + self.residual(x)


class TCNBaseline(nn.Module):
    def __init__(self,
                 num_joints_in, num_joints_out,
                 dim_in=3, dim_out=3,
                 hidden_size=256, num_layers=4,
                 kernel_size=3, dilation_base=2,
                 drop_rate=0.1):
        super().__init__()
        self.num_joints_out = num_joints_out
        self.dim_out = dim_out
        self.in_dim = num_joints_in * dim_in
        self.out_dim = num_joints_out * dim_out

        layers = []
        in_ch = self.in_dim
        for i in range(num_layers):
            dilation = dilation_base ** i
            layers.append(TemporalConvBlock(in_ch, hidden_size, kernel_size, dilation, drop_rate))
            in_ch = hidden_size
        self.network = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden_size, self.out_dim, kernel_size=1)

    def forward(self, x, **_):
        B, F, J, C = x.shape
        x = x.view(B, F, J * C).transpose(1, 2)
        y = self.head(self.network(x)).transpose(1, 2)
        return y.view(B, F, self.num_joints_out, self.dim_out)


class DilatedCNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilations=(1, 2, 4), drop_rate=0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      padding=(kernel_size - 1) // 2 * d, dilation=d)
            for d in dilations
        ])
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop_rate)
        self.residual = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        y = sum(conv(x) for conv in self.convs) / len(self.convs)
        y = self.drop(self.act(self.norm(y)))
        return y + self.residual(x)


class DilatedCNNBaseline(nn.Module):
    def __init__(self,
                 num_joints_in, num_joints_out,
                 dim_in=3, dim_out=3,
                 hidden_size=256, num_layers=4,
                 kernel_size=3, drop_rate=0.1):
        super().__init__()
        self.num_joints_out = num_joints_out
        self.dim_out = dim_out
        self.in_dim = num_joints_in * dim_in
        self.out_dim = num_joints_out * dim_out

        layers = []
        in_ch = self.in_dim
        for _ in range(num_layers):
            layers.append(DilatedCNNBlock(in_ch, hidden_size, kernel_size=kernel_size, drop_rate=drop_rate))
            in_ch = hidden_size
        self.network = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden_size, self.out_dim, kernel_size=1)

    def forward(self, x, **_):
        B, F, J, C = x.shape
        x = x.view(B, F, J * C).transpose(1, 2)
        y = self.head(self.network(x)).transpose(1, 2)
        return y.view(B, F, self.num_joints_out, self.dim_out)


class LocalAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, window_size=9, drop_rate=0.1, ff_mult=4):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=drop_rate, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def _build_mask(self, length, device):
        radius = max(1, self.window_size // 2)
        idx = torch.arange(length, device=device)
        dist = (idx[None, :] - idx[:, None]).abs()
        mask = torch.where(dist <= radius, 0.0, float("-inf"))
        return mask

    def forward(self, x):
        mask = self._build_mask(x.size(1), x.device)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class LocalAttentionTransformerBaseline(nn.Module):
    """
    Efficient transformer variant using local attention.
    Why: local windows reduce quadratic cost while preserving short-range motion cues.
    """
    def __init__(self,
                 num_joints_in, num_joints_out,
                 dim_in=3, dim_out=3,
                 d_model=256, nhead=8, num_layers=4,
                 window_size=9, drop_rate=0.1):
        super().__init__()
        self.num_joints_out = num_joints_out
        self.dim_out = dim_out
        self.in_dim = num_joints_in * dim_in
        self.out_dim = num_joints_out * dim_out

        self.proj_in = nn.Linear(self.in_dim, d_model)
        self.layers = nn.ModuleList([
            LocalAttentionLayer(d_model, nhead, window_size=window_size, drop_rate=drop_rate)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.out_dim)

    def forward(self, x, **_):
        B, F, J, C = x.shape
        x = x.view(B, F, J * C)
        x = self.proj_in(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        y = self.head(x).view(B, F, self.num_joints_out, self.dim_out)
        return y


class LinformerSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, max_len, proj_len, drop_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.max_len = max_len
        self.proj_len = proj_len

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj_k = nn.Linear(max_len, proj_len, bias=False)
        self.proj_v = nn.Linear(max_len, proj_len, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(drop_rate)

    def _pad_or_truncate(self, x):
        length = x.size(1)
        if length == self.max_len:
            return x, length
        if length > self.max_len:
            return x[:, :self.max_len], self.max_len
        pad = x.new_zeros(x.size(0), self.max_len - length, x.size(-1))
        return torch.cat([x, pad], dim=1), length

    def forward(self, x):
        x, orig_len = self._pad_or_truncate(x)
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        k = self.proj_k(k.transpose(1, 2)).transpose(1, 2)
        v = self.proj_v(v.transpose(1, 2)).transpose(1, 2)

        B = q.size(0)
        q = q.view(B, -1, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.nhead, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        out = self.out(out)
        return out[:, :orig_len]


class LinformerLayer(nn.Module):
    def __init__(self, d_model, nhead, max_len, proj_len, drop_rate=0.1, ff_mult=4):
        super().__init__()
        self.attn = LinformerSelfAttention(d_model, nhead, max_len, proj_len, drop_rate=drop_rate)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class LinformerBaseline(nn.Module):
    """
    Efficient transformer variant using Linformer projections.
    Why: low-rank K/V projection keeps long-range context with lower memory cost.
    """
    def __init__(self,
                 num_joints_in, num_joints_out,
                 dim_in=3, dim_out=3,
                 d_model=256, nhead=8, num_layers=4,
                 max_len=243, proj_ratio=0.25,
                 drop_rate=0.1):
        super().__init__()
        self.num_joints_out = num_joints_out
        self.dim_out = dim_out
        self.in_dim = num_joints_in * dim_in
        self.out_dim = num_joints_out * dim_out

        proj_len = max(1, int(max_len * proj_ratio))
        self.proj_in = nn.Linear(self.in_dim, d_model)
        self.layers = nn.ModuleList([
            LinformerLayer(d_model, nhead, max_len=max_len, proj_len=proj_len, drop_rate=drop_rate)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.out_dim)

    def forward(self, x, **_):
        B, F, J, C = x.shape
        x = x.view(B, F, J * C)
        x = self.proj_in(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        y = self.head(x).view(B, F, self.num_joints_out, self.dim_out)
        return y


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, drop_rate=0.1):
        super().__init__()
        mamba_cls = _require_mamba()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = mamba_cls(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        y = self.mamba(self.norm(x))
        return x + self.drop(y)


class MambaBaseline(nn.Module):
    def __init__(self,
                 num_joints_in, num_joints_out,
                 dim_in=3, dim_out=3,
                 d_model=256, depth=4,
                 d_state=16, d_conv=4, expand=2,
                 drop_rate=0.1):
        super().__init__()
        self.num_joints_out = num_joints_out
        self.dim_out = dim_out
        self.in_dim = num_joints_in * dim_in
        self.out_dim = num_joints_out * dim_out

        self.proj_in = nn.Linear(self.in_dim, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, drop_rate=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.out_dim)

    def forward(self, x, **_):
        B, F, J, C = x.shape
        x = x.view(B, F, J * C)
        x = self.proj_in(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        y = self.head(x).view(B, F, self.num_joints_out, self.dim_out)
        return y


class S4BlockWrapper(nn.Module):
    def __init__(self, d_model, drop_rate=0.1):
        super().__init__()
        s4_cls = _require_s4()
        try:
            self.s4 = s4_cls(d_model=d_model, dropout=drop_rate, transposed=False)
        except TypeError:
            self.s4 = s4_cls(d_model)

    def forward(self, x):
        out = self.s4(x)
        if isinstance(out, tuple):
            out = out[0]
        return out


class S4Baseline(nn.Module):
    def __init__(self,
                 num_joints_in, num_joints_out,
                 dim_in=3, dim_out=3,
                 d_model=256, depth=4,
                 drop_rate=0.1):
        super().__init__()
        self.num_joints_out = num_joints_out
        self.dim_out = dim_out
        self.in_dim = num_joints_in * dim_in
        self.out_dim = num_joints_out * dim_out

        self.proj_in = nn.Linear(self.in_dim, d_model)
        self.blocks = nn.ModuleList([
            S4BlockWrapper(d_model, drop_rate=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.out_dim)

    def forward(self, x, **_):
        B, F, J, C = x.shape
        x = x.view(B, F, J * C)
        x = self.proj_in(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        y = self.head(x).view(B, F, self.num_joints_out, self.dim_out)
        return y


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency: torch.Tensor):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # x: (B, T, V, C)
        agg = torch.einsum("btvc,vw->btwc", x, self.adjacency.to(x.device))
        return self.lin(agg)


class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, adjacency: torch.Tensor, kernel_size=3, drop_rate=0.1):
        super().__init__()
        self.gcn = GraphConv(in_ch, out_ch, adjacency)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=(kernel_size, 1),
                      padding=((kernel_size - 1) // 2, 0)),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(drop_rate),
        )
        self.residual = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        # x: (B, C, T, V)
        y = self.gcn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        y = self.tcn(y)
        res = self.residual(x)
        return F.relu(y + res)


class STGCNBaseline(nn.Module):
    def __init__(self,
                 num_joints_in, num_joints_out,
                 adjacency: torch.Tensor,
                 dim_in=3, dim_out=3,
                 hidden_dim=128, num_layers=3,
                 kernel_size=3, drop_rate=0.1):
        super().__init__()
        self.num_joints_in = num_joints_in
        self.num_joints_out = num_joints_out
        self.dim_out = dim_out

        # Project joints first if input != output
        self.joint_proj = nn.Linear(num_joints_in, num_joints_out) if num_joints_in != num_joints_out else nn.Identity()
        
        self.input_proj = nn.Conv2d(dim_in, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            STGCNBlock(hidden_dim, hidden_dim, adjacency, kernel_size=kernel_size, drop_rate=drop_rate)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, dim_out)

    def forward(self, x, **_):
        B, F, J, C = x.shape
        
        # Project to output joint space first if needed
        if J != self.num_joints_out:
            x = x.permute(0, 1, 3, 2)  # (B, F, C, J)
            x = self.joint_proj(x).permute(0, 1, 3, 2)  # (B, F, V_out, C)
        
        x = x.permute(0, 3, 1, 2)  # (B, C, T, V)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = x.permute(0, 2, 3, 1)  # (B, T, V, H)
        y = self.out_proj(x).view(B, F, self.num_joints_out, self.dim_out)
        return y


class GCNAttnBaseline(nn.Module):
    def __init__(self,
                 num_joints_in, num_joints_out,
                 adjacency: torch.Tensor,
                 dim_in=3, dim_out=3,
                 hidden_dim=128, num_layers=2,
                 num_heads=4, drop_rate=0.1):
        super().__init__()
        self.num_joints_in = num_joints_in
        self.num_joints_out = num_joints_out
        self.dim_out = dim_out

        # Project joints first if input != output
        self.joint_proj = nn.Linear(num_joints_in, num_joints_out) if num_joints_in != num_joints_out else nn.Identity()
        
        self.gcn = GraphConv(dim_in, hidden_dim, adjacency)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=drop_rate, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.out_proj = nn.Linear(hidden_dim, dim_out)

    def forward(self, x, **_):
        B, F, J, C = x.shape
        
        # Project to output joint space first if needed
        if J != self.num_joints_out:
            x = x.permute(0, 1, 3, 2)  # (B, F, C, J)
            x = self.joint_proj(x).permute(0, 1, 3, 2)  # (B, F, V_out, C)
            J = self.num_joints_out
        
        x = self.gcn(x)  # (B, T, V, H)
        x = x.permute(0, 2, 1, 3).reshape(B * J, F, -1)
        for attn, norm in zip(self.attn_layers, self.norms):
            res = x
            attn_out, _ = attn(norm(x), norm(x), norm(x))
            x = res + attn_out
        x = x.view(B, J, F, -1).permute(0, 2, 1, 3)
        y = self.out_proj(x).view(B, F, self.num_joints_out, self.dim_out)
        return y


class CNNAttnBaseline(nn.Module):
    def __init__(self,
                 num_joints_in, num_joints_out,
                 dim_in=3, dim_out=3,
                 hidden_size=256, num_layers=2,
                 kernel_size=3, num_heads=4,
                 drop_rate=0.1):
        super().__init__()
        self.num_joints_out = num_joints_out
        self.dim_out = dim_out
        self.in_dim = num_joints_in * dim_in
        self.out_dim = num_joints_out * dim_out

        layers = []
        in_ch = self.in_dim
        for _ in range(num_layers):
            layers.append(TemporalConvBlock(in_ch, hidden_size, kernel_size=kernel_size, drop_rate=drop_rate))
            in_ch = hidden_size
        self.network = nn.Sequential(*layers)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=drop_rate, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, self.out_dim)

    def forward(self, x, **_):
        B, F, J, C = x.shape
        x = x.view(B, F, J * C).transpose(1, 2)
        x = self.network(x).transpose(1, 2)
        res = x
        attn_out, _ = self.attn(self.norm(x), self.norm(x), self.norm(x))
        x = res + attn_out
        y = self.head(x).view(B, F, self.num_joints_out, self.dim_out)
        return y


class CNNMambaBaseline(nn.Module):
    def __init__(self,
                 num_joints_in, num_joints_out,
                 dim_in=3, dim_out=3,
                 hidden_size=256, num_layers=2,
                 kernel_size=3,
                 d_state=16, d_conv=4, expand=2,
                 drop_rate=0.1):
        super().__init__()
        self.num_joints_out = num_joints_out
        self.dim_out = dim_out
        self.in_dim = num_joints_in * dim_in
        self.out_dim = num_joints_out * dim_out

        layers = []
        in_ch = self.in_dim
        for _ in range(num_layers):
            layers.append(TemporalConvBlock(in_ch, hidden_size, kernel_size=kernel_size, drop_rate=drop_rate))
            in_ch = hidden_size
        self.network = nn.Sequential(*layers)
        self.mamba = MambaBlock(hidden_size, d_state=d_state, d_conv=d_conv, expand=expand, drop_rate=drop_rate)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, self.out_dim)

    def forward(self, x, **_):
        B, F, J, C = x.shape
        x = x.view(B, F, J * C).transpose(1, 2)
        x = self.network(x).transpose(1, 2)
        x = self.mamba(x)
        x = self.norm(x)
        y = self.head(x).view(B, F, self.num_joints_out, self.dim_out)
        return y


class CNNS4Baseline(nn.Module):
    def __init__(self,
                 num_joints_in, num_joints_out,
                 dim_in=3, dim_out=3,
                 hidden_size=256, num_layers=2,
                 kernel_size=3,
                 drop_rate=0.1):
        super().__init__()
        self.num_joints_out = num_joints_out
        self.dim_out = dim_out
        self.in_dim = num_joints_in * dim_in
        self.out_dim = num_joints_out * dim_out

        layers = []
        in_ch = self.in_dim
        for _ in range(num_layers):
            layers.append(TemporalConvBlock(in_ch, hidden_size, kernel_size=kernel_size, drop_rate=drop_rate))
            in_ch = hidden_size
        self.network = nn.Sequential(*layers)
        self.s4 = S4BlockWrapper(hidden_size, drop_rate=drop_rate)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, self.out_dim)

    def forward(self, x, **_):
        B, F, J, C = x.shape
        x = x.view(B, F, J * C).transpose(1, 2)
        x = self.network(x).transpose(1, 2)
        x = self.s4(x)
        x = self.norm(x)
        y = self.head(x).view(B, F, self.num_joints_out, self.dim_out)
        return y

