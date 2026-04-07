# file: lib/model/pose_baselines.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from . import loss as loss_module
from lib.utils.metrics import BodyAngleCalculator
from lib.utils.markers_names_move4d import markers_names
from .baselines_extra import (
    TCNBaseline,
    DilatedCNNBaseline,
    LocalAttentionTransformerBaseline,
    LinformerBaseline,
    MambaBaseline,
    S4Baseline,
    STGCNBaseline,
    GCNAttnBaseline,
    CNNAttnBaseline,
    CNNMambaBaseline,
    CNNS4Baseline,
)

# ---------- 1. Modelli “core” ------------------------------------------------

# pose_baselines_lit.py
import torch, torch.nn as nn
import math

# ---------- 1. MLP -----------------------------------------------------------
class MLPBaseline(nn.Module):
    """
    Frame-wise MLP (fonte: Sensors 2024, sec. 2.3.1).
    Inp:  (B,F,J_in*3+meta)  Out: (B,F,J_out*3)
    """
    def __init__(self,
                 num_joints_in, num_joints_out,
                 dim_in=3, dim_out=3,
                 use_meta=False, drop_rate=0.1):
        super().__init__()
        meta = 3 if use_meta else 0        # height, weight, sex
        self.in_dim  = num_joints_in * dim_in + meta
        self.out_dim = num_joints_out * dim_out

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, 256), nn.BatchNorm1d(256),
            nn.Tanh(), nn.Dropout(drop_rate),

            nn.Linear(256, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(drop_rate),

            nn.Linear(128, 224), nn.BatchNorm1d(224),
            nn.ReLU(), nn.Dropout(drop_rate),

            nn.Linear(224, self.out_dim)
        )

    def forward(self, x, meta=None):
        # x:(B,F,J,C)   meta facoltativo:(B,F,3)
        B,F,J,C = x.shape
        x = x.reshape(B*F, J*C)
        if meta is not None:
            meta = meta.reshape(B*F, -1)
            x = torch.cat([x, meta], -1)
        y = self.net(x).view(B, F, -1, C)
        return y

# ---------- 2. LSTM ----------------------------------------------------------
class LSTMBaseline(nn.Module):
    """
    2-layer LSTM da 128 unit (fonte: Sensors 2024, sec. 2.3.2).
    """
    def __init__(self,
                 num_joints_in, num_joints_out,
                 dim_in=3, dim_out=3,
                 hidden_size=128, num_layers=2,
                 drop_rate=0.1):
        super().__init__()
        self.in_dim  = num_joints_in * dim_in
        self.out_dim = num_joints_out * dim_out

        self.lstm = nn.LSTM(
            input_size=self.in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=drop_rate)

        self.head = nn.Linear(hidden_size, self.out_dim)

    def forward(self, x, **_):
        # x: (B,F,J,C)
        B,F,J,C = x.shape
        x = x.view(B, F, J*C)
        y, _ = self.lstm(x)
        y = self.head(y).view(B, F, -1, C)
        return y

# ---------- 2b. GRU ---------------------------------------------------------
class GRUBaseline(nn.Module):
    """
    2-layer GRU baseline (same interface as LSTMBaseline).
    """
    def __init__(self,
                 num_joints_in, num_joints_out,
                 dim_in=3, dim_out=3,
                 hidden_size=128, num_layers=2,
                 drop_rate=0.1):
        super().__init__()
        self.in_dim = num_joints_in * dim_in
        self.out_dim = num_joints_out * dim_out

        self.gru = nn.GRU(
            input_size=self.in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=drop_rate if num_layers > 1 else 0.0,
        )

        self.head = nn.Linear(hidden_size, self.out_dim)

    def forward(self, x, **_):
        B, F, J, C = x.shape
        x = x.view(B, F, J * C)
        y, _ = self.gru(x)
        y = self.head(y).view(B, F, -1, C)
        return y

# ---------- 2c. SRU ---------------------------------------------------------
class SRULayer(nn.Module):
    """
    Simple SRU layer (CPU/GPU friendly, no custom CUDA kernels).
    """
    def __init__(self, input_size, hidden_size, drop_rate=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size, hidden_size * 3)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, c0=None):
        B, F, _ = x.shape
        u = self.linear(x)  # (B, F, 3H)
        x_tilde, f, r = torch.chunk(u, 3, dim=-1)
        f = torch.sigmoid(f)
        r = torch.sigmoid(r)

        c_prev = x.new_zeros(B, self.hidden_size) if c0 is None else c0
        h_out = []
        for t in range(F):
            c_prev = f[:, t] * c_prev + (1.0 - f[:, t]) * x_tilde[:, t]
            h_t = r[:, t] * torch.tanh(c_prev) + (1.0 - r[:, t]) * x_tilde[:, t]
            h_out.append(self.dropout(h_t))

        h = torch.stack(h_out, dim=1)
        return h


class SRUBaseline(nn.Module):
    """
    SRU baseline with configurable depth.
    """
    def __init__(self,
                 num_joints_in, num_joints_out,
                 dim_in=3, dim_out=3,
                 hidden_size=128, num_layers=2,
                 drop_rate=0.1):
        super().__init__()
        self.in_dim = num_joints_in * dim_in
        self.out_dim = num_joints_out * dim_out

        self.input_proj = nn.Linear(self.in_dim, hidden_size)
        self.layers = nn.ModuleList([
            SRULayer(hidden_size, hidden_size, drop_rate=drop_rate)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_size, self.out_dim)

    def forward(self, x, **_):
        B, F, J, C = x.shape
        x = x.view(B, F, J * C)
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        y = self.head(h).view(B, F, -1, C)
        return y

# ---------- 3. MotionMixer --------------------------------------------------
class MotionMixerBlock(nn.Module):
    def __init__(self, num_frames, dim, mlp_ratio=4.0, drop_rate=0.1):
        super().__init__()
        token_dim = max(1, int(num_frames * mlp_ratio))
        channel_dim = max(1, int(dim * mlp_ratio))

        self.norm_t = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_frames, token_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(token_dim, num_frames),
            nn.Dropout(drop_rate),
        )

        self.norm_c = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(channel_dim, dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        # x: (B, F, D)
        y = self.norm_t(x).transpose(1, 2)  # (B, D, F)
        y = self.token_mlp(y).transpose(1, 2)
        x = x + y

        z = self.norm_c(x)
        z = self.channel_mlp(z)
        x = x + z
        return x


class MotionMixerBaseline(nn.Module):
    """
    MLP-Mixer style temporal baseline.
    """
    def __init__(self,
                 num_joints_in, num_joints_out,
                 num_frames,
                 dim_in=3, dim_out=3,
                 dim_feat=256, depth=4,
                 mlp_ratio=4.0, drop_rate=0.1):
        super().__init__()
        self.in_dim = num_joints_in * dim_in
        self.out_dim = num_joints_out * dim_out
        self.num_frames = num_frames

        self.proj_in = nn.Linear(self.in_dim, dim_feat)
        self.blocks = nn.ModuleList([
            MotionMixerBlock(num_frames, dim_feat, mlp_ratio=mlp_ratio, drop_rate=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim_feat)
        self.head = nn.Linear(dim_feat, self.out_dim)

    def forward(self, x, **_):
        B, F, J, C = x.shape
        if F != self.num_frames:
            raise ValueError(f"MotionMixer expects {self.num_frames} frames, got {F}")
        x = x.view(B, F, J * C)
        x = self.proj_in(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        y = self.head(x).view(B, F, -1, C)
        return y

# ---------- 4. SiMLPe -------------------------------------------------------
class SiMLPeBaseline(nn.Module):
    """
    Simple MLP baseline with temporal mixing + channel MLP.
    """
    def __init__(self,
                 num_joints_in, num_joints_out,
                 num_frames,
                 dim_in=3, dim_out=3,
                 hidden_dim=256,
                 drop_rate=0.1):
        super().__init__()
        self.in_dim = num_joints_in * dim_in
        self.out_dim = num_joints_out * dim_out
        self.num_frames = num_frames

        self.temporal_mlp = nn.Sequential(
            nn.Linear(num_frames, num_frames),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(num_frames, num_frames),
            nn.Dropout(drop_rate),
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, self.out_dim),
        )

    def forward(self, x, **_):
        B, F, J, C = x.shape
        if F != self.num_frames:
            raise ValueError(f"SiMLPe expects {self.num_frames} frames, got {F}")
        x = x.view(B, F, J * C)
        x = self.temporal_mlp(x.transpose(1, 2)).transpose(1, 2)
        y = self.channel_mlp(x).view(B, F, -1, C)
        return y

# ---------- 3. Transformer ---------------------------------------------------
class TransformerBaseline(nn.Module):
    """
    Encoder (6 layer, 14 heads) come in lifting.pdf](file-service://file-Epyw9UUMgNAjV9WAPBuMrt)
    """
    def __init__(self,
                 num_joints_in, num_joints_out,
                 dim_in=3, dim_out=3,
                 d_model=256, nhead=8,
                 num_layers=6, dim_ff=1024,
                 drop_rate=0.1, max_len=512):
        super().__init__()
        self.in_dim  = num_joints_in * dim_in
        self.out_dim = num_joints_out * dim_out

        self.proj_in = nn.Conv1d(self.in_dim, d_model, kernel_size=1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=drop_rate,
            batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.head = nn.Linear(d_model, self.out_dim)

        # sinusoidal PE
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) *
                        (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # (max_len,d_model)

    def forward(self, x, **_):
        # x: (B,F,J,C)  -> (B,F,d_model)
        B,F,J,C = x.shape
        x = x.view(B, F, J*C).transpose(1,2)   # (B,in_dim,F)
        x = self.proj_in(x).transpose(1,2)     # (B,F,d_model)
        x = x + self.pe[:F]
        y = self.encoder(x)
        y = self.head(y).view(B, F, -1, C)
        return y


# ---------- 2. Lightning Module “universale” ---------------------------------

class PoseBaselinePL(pl.LightningModule):
    """
    Wrapper Lightning per MLP, RNN e Mixer con stessa logica
    di training / valid / test usato per il benchmark pubblico.
    """
    def __init__(self,
                 model_type="mlp",              # mlp | lstm | gru | sru | transformer | motion_mixer | simlpe | tcn | dilated_cnn | local_attn_transformer | linformer | mamba | s4 | stgcn | gcn_attn | cnn_attn | cnn_mamba | cnn_s4
                 frame_window=243,              # lunghezza sequenza in input
                 num_joints_in=17,
                 num_joints_out=25,
                 dim_in=3, dim_out=3,
                 optimizer_config=None,
                 scheduler_config=None,
                 loss_config=None,
                 metrics_fn="mpjpe",
                 predictions_output_dir=None,
                 dataset_type="move4d",
                 **model_hparams):
        super().__init__()
        self.save_hyperparameters()

        # --- crea il modello scelto -----------------
        model_type = model_type.lower().replace("-", "_")
        if model_type in {"motionmixer", "motion_mixer"}:
            model_type = "motion_mixer"
        elif model_type in {"simlpe", "simlpe_baseline"}:
            model_type = "simlpe"
        elif model_type in {"dilatedcnn"}:
            model_type = "dilated_cnn"
        elif model_type in {"local_attn", "local_attention"}:
            model_type = "local_attn_transformer"
        elif model_type in {"efficient_local_attn"}:
            model_type = "local_attn_transformer"
        elif model_type in {"linformer_transformer"}:
            model_type = "linformer"
        elif model_type in {"st_gcn"}:
            model_type = "stgcn"
        elif model_type in {"gcn_attention", "gcn_attn"}:
            model_type = "gcn_attn"
        elif model_type in {"cnn_attention"}:
            model_type = "cnn_attn"
        elif model_type in {"ssm_mamba", "mamba_ssm"}:
            model_type = "mamba"
        elif model_type in {"ssm_s4", "s4_ssm"}:
            model_type = "s4"
        elif model_type in {"cnn_ssm_mamba"}:
            model_type = "cnn_mamba"
        elif model_type in {"cnn_ssm_s4"}:
            model_type = "cnn_s4"

        if model_type == "mlp":
            self.model = MLPBaseline(num_joints_in, num_joints_out,
                                     dim_in=dim_in, dim_out=dim_out,
                                     **model_hparams)
        elif model_type == "lstm":
            self.model = LSTMBaseline(num_joints_in, num_joints_out,
                                       dim_in=dim_in, dim_out=dim_out,
                                       **model_hparams)
        elif model_type == "gru":
            self.model = GRUBaseline(num_joints_in, num_joints_out,
                                      dim_in=dim_in, dim_out=dim_out,
                                      **model_hparams)
        elif model_type == "sru":
            self.model = SRUBaseline(num_joints_in, num_joints_out,
                                      dim_in=dim_in, dim_out=dim_out,
                                      **model_hparams)
        elif model_type == "transformer":
            self.model = TransformerBaseline(num_joints_in, num_joints_out,
                                             dim_in=dim_in, dim_out=dim_out,
                                             max_len=frame_window,
                                             **model_hparams)
        elif model_type == "motion_mixer":
            self.model = MotionMixerBaseline(num_joints_in, num_joints_out,
                                             num_frames=frame_window,
                                             dim_in=dim_in, dim_out=dim_out,
                                             **model_hparams)
        elif model_type == "simlpe":
            self.model = SiMLPeBaseline(num_joints_in, num_joints_out,
                                        num_frames=frame_window,
                                        dim_in=dim_in, dim_out=dim_out,
                                        **model_hparams)
        elif model_type == "tcn":
            self.model = TCNBaseline(num_joints_in, num_joints_out,
                                     dim_in=dim_in, dim_out=dim_out,
                                     **model_hparams)
        elif model_type == "dilated_cnn":
            self.model = DilatedCNNBaseline(num_joints_in, num_joints_out,
                                            dim_in=dim_in, dim_out=dim_out,
                                            **model_hparams)
        elif model_type == "local_attn_transformer":
            self.model = LocalAttentionTransformerBaseline(num_joints_in, num_joints_out,
                                                           dim_in=dim_in, dim_out=dim_out,
                                                           **model_hparams)
        elif model_type == "linformer":
            self.model = LinformerBaseline(num_joints_in, num_joints_out,
                                           dim_in=dim_in, dim_out=dim_out,
                                           **model_hparams)
        elif model_type == "mamba":
            self.model = MambaBaseline(num_joints_in, num_joints_out,
                                       dim_in=dim_in, dim_out=dim_out,
                                       **model_hparams)
        elif model_type == "s4":
            self.model = S4Baseline(num_joints_in, num_joints_out,
                                    dim_in=dim_in, dim_out=dim_out,
                                    **model_hparams)
        elif model_type == "stgcn":
            self.model = STGCNBaseline(num_joints_in, num_joints_out,
                                       dim_in=dim_in, dim_out=dim_out,
                                       **model_hparams)
        elif model_type == "gcn_attn":
            self.model = GCNAttnBaseline(num_joints_in, num_joints_out,
                                         dim_in=dim_in, dim_out=dim_out,
                                         **model_hparams)
        elif model_type == "cnn_attn":
            self.model = CNNAttnBaseline(num_joints_in, num_joints_out,
                                         dim_in=dim_in, dim_out=dim_out,
                                         **model_hparams)
        elif model_type == "cnn_mamba":
            self.model = CNNMambaBaseline(num_joints_in, num_joints_out,
                                          dim_in=dim_in, dim_out=dim_out,
                                          **model_hparams)
        elif model_type == "cnn_s4":
            self.model = CNNS4Baseline(num_joints_in, num_joints_out,
                                       dim_in=dim_in, dim_out=dim_out,
                                       **model_hparams)
        else:
            raise ValueError(f"Unknown model_type '{model_type}'")

        # --- loss & metric shared by all public benchmark baselines ---
        self.dataset_type = dataset_type
        self.loss_config = loss_config or {}
        self.lambda_3d_velocity = self.loss_config.get("lambda_3d_velocity", 0.0)
        self.lambda_bone_length_prior = self.loss_config.get("lambda_bone_length_prior", 0.0)
        self.lambda_bone_orientation = self.loss_config.get("lambda_bone_orientation", 0.0)
        self.lambda_mse = self.loss_config.get("lambda_mse", 0.0)

        self.loss_3d_pos = getattr(loss_module, "masked_mpjpe", F.mse_loss)
        self.loss_3d_velocity = getattr(loss_module, "masked_loss_velocity", F.mse_loss)
        self.loss_bone_length_prior = getattr(loss_module, "loss_bone_length_prior", F.mse_loss)
        self.loss_bone_orientation = getattr(loss_module, "loss_bone_orientation", F.mse_loss)
        self.loss_mse = F.mse_loss

        if hasattr(loss_module, metrics_fn):
            self.metrics_fn = getattr(loss_module, metrics_fn)
        else:
            raise ValueError(f"metrics_fn '{metrics_fn}' non trovato in loss_module")

        if self.dataset_type == "amass":
            self.segments = getattr(loss_module, "AMASS_SEGMENTS", None)
        else:
            self.segments = getattr(loss_module, "SEGMENTS", None)

        if self.segments is None:
            raise ValueError(f"Segments not found for dataset_type='{self.dataset_type}'")

        # optim & sched
        self.optimizer_config = optimizer_config or {
            "name": "adamw",
            "params": {"lr": 5e-4, "weight_decay": 0.05}
        }
        self.scheduler_config = scheduler_config or {"name": "none", "params": {}}

        # prediction dump
        self.predictions_output_dir = predictions_output_dir
        self.raw_predictions = []

        # angle tracking for MOVE4D
        self.bodyAngleCalculator = BodyAngleCalculator()
        self.test_angles_per_action = {}
        self.test_movement_mpjpe = {}

    # ---------- common forward ----------
    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    # ---------- loss helper -------------
    def _compute_total_loss(self, pred, gt, mask):
        l_pos = self.loss_3d_pos(pred, gt, mask=mask)
        l_vel = self.lambda_3d_velocity * self.loss_3d_velocity(pred, gt, mask=mask)
        l_len = self.lambda_bone_length_prior * self.loss_bone_length_prior(
            pred, gt, mask=mask, segments=self.segments
        )
        l_ori = self.lambda_bone_orientation * self.loss_bone_orientation(
            pred, gt, mask=mask, segments=self.segments
        )
        
        # MSE loss - apply mask by setting masked positions to 0
        if self.lambda_mse > 0:
            pred_masked = pred * mask.unsqueeze(-1)
            gt_masked = gt * mask.unsqueeze(-1)
            l_mse = self.lambda_mse * self.loss_mse(pred_masked, gt_masked)
        else:
            l_mse = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            
        total_loss = l_pos + l_vel + l_len + l_ori + l_mse
        loss_parts = {"3d_pos": l_pos, "3d_vel": l_vel, "bone_len": l_len, "bone_ori": l_ori, "mse": l_mse}
        
        return total_loss, loss_parts

    # ---------- steps -------------------
    def _shared_step(self, batch, stage):
        kp3d, al = batch["kp3d"], batch["al"]
        mask = (kp3d != 0).any(dim=-1)
        mask_al = (al != 0).any(dim=-1)

        pred = self(kp3d)
        mpjpe = self.metrics_fn(pred, al, mask=mask_al)
        loss, parts = self._compute_total_loss(pred, al, mask_al)

        self.log(f"{stage}_loss", loss, batch_size=len(kp3d), prog_bar=True)
        self.log(f"{stage}_mpjpe", mpjpe, batch_size=len(kp3d), prog_bar=True)
        
        # Log individual loss components if they are significant
        if self.lambda_mse > 0:
            self.log(f"{stage}_mse_loss", parts["mse"], batch_size=len(kp3d))
            
        return loss

    def training_step(self, batch, batch_idx):   return self._shared_step(batch, "train")
    def validation_step(self, batch, batch_idx): return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        kp3d, al = batch["kp3d"], batch["al"]
        actions = batch.get("action", None)
        mask_al = (al != 0).any(dim=-1)

        pred = self(kp3d)
        mpjpe = self.metrics_fn(pred, al, mask=mask_al)
        loss, parts = self._compute_total_loss(pred, al, mask_al)

        self.log("test_loss", loss, batch_size=len(kp3d), prog_bar=True)
        self.log("test_mpjpe", mpjpe, batch_size=len(kp3d), prog_bar=True)
        if self.lambda_mse > 0:
            self.log("test_mse_loss", parts["mse"], batch_size=len(kp3d))
        
        # Collect per-movement MPJPE for MOVE4D dataset
        if actions is not None and self.dataset_type == "move4d":
            for i in range(len(kp3d)):
                action_name = actions[i] if isinstance(actions[i], str) else str(actions[i])
                if isinstance(actions[i], np.ndarray):
                    action_name = str(actions[i].item() if actions[i].size == 1 else actions[i][0])
                
                sample_mpjpe = self.metrics_fn(pred[i:i+1], al[i:i+1], mask=mask_al[i:i+1])
                
                if action_name not in self.test_movement_mpjpe:
                    self.test_movement_mpjpe[action_name] = []
                self.test_movement_mpjpe[action_name].append(sample_mpjpe.item())

        if self.dataset_type == "move4d":
            for i in range(len(kp3d)):
                valid_frames = mask_al[i].any(dim=-1)
                if valid_frames.sum().item() == 0:
                    continue
                pred_seq = pred[i][valid_frames].detach().cpu().numpy()
                gt_seq = al[i][valid_frames].detach().cpu().numpy()

                angle_errors = self._compute_move4d_angle_errors(pred_seq, gt_seq)
                if angle_errors is None:
                    continue

                if actions is not None:
                    if isinstance(actions[i], np.ndarray):
                        if actions[i].size == 1:
                            action_id = str(actions[i].item())
                        else:
                            action_id = str(actions[i][0])
                    elif isinstance(actions[i], (tuple, list)):
                        action_id = str(actions[i][0])
                    else:
                        action_id = str(actions[i])
                else:
                    action_id = f"{self.dataset_type}_sequence"

                if action_id not in self.test_angles_per_action:
                    self.test_angles_per_action[action_id] = {}

                for joint_name, joint_vals in angle_errors.items():
                    for angle_name, val in joint_vals.items():
                        key = f"{joint_name}_{angle_name}"
                        self.test_angles_per_action[action_id].setdefault(key, []).append(val)

        return loss

    def on_test_epoch_end(self):
        if self.dataset_type == "move4d" and self.test_angles_per_action:
            for action_id, angle_dict in self.test_angles_per_action.items():
                safe_action = action_id.replace(" ", "_")
                for angle_key, values in angle_dict.items():
                    if not values:
                        continue
                    mean_val = float(np.mean(values))
                    std_val = float(np.std(values))
                    self.log(f"test_angle_mean_{safe_action}_{angle_key}", mean_val, prog_bar=False)
                    self.log(f"test_angle_std_{safe_action}_{angle_key}", std_val, prog_bar=False)
        
        # Log per-movement MPJPE statistics for MOVE4D
        if self.dataset_type == "move4d" and self.test_movement_mpjpe:
            print(f"\n=== Per-Movement MPJPE Statistics ===")
            for movement, mpjpe_values in self.test_movement_mpjpe.items():
                if mpjpe_values:
                    mean_mpjpe = float(np.mean(mpjpe_values))
                    std_mpjpe = float(np.std(mpjpe_values))
                    print(f"{movement}: {mean_mpjpe:.4f} ± {std_mpjpe:.4f} mm ({len(mpjpe_values)} samples)")
                    self.log(f"test_mpjpe_mean_{movement}", mean_mpjpe, prog_bar=False)
                    self.log(f"test_mpjpe_std_{movement}", std_mpjpe, prog_bar=False)

        self.test_angles_per_action = {}
        self.test_movement_mpjpe = {}

    def _compute_move4d_angle_errors(self, pred_seq, gt_seq):
        if pred_seq.shape[1] != len(markers_names) or gt_seq.shape[1] != len(markers_names):
            return None

        errors = {"hip": {"FE": [], "AB-AD": [], "ROT": []},
                  "knee": {"FE": [], "AB-AD": [], "ROT": []},
                  "ankle": {"FE": [], "AB-AD": [], "ROT": []}}

        for frame_idx in range(pred_seq.shape[0]):
            pred_markers = {name: pred_seq[frame_idx, idx] for idx, name in enumerate(markers_names)}
            gt_markers = {name: gt_seq[frame_idx, idx] for idx, name in enumerate(markers_names)}

            angles_pred = self.bodyAngleCalculator.compute_angles(pred_markers)
            angles_gt = self.bodyAngleCalculator.compute_angles(gt_markers)

            for joint in errors:
                for key in errors[joint]:
                    errors[joint][key].append(abs(angles_pred[joint][key] - angles_gt[joint][key]))

        return {joint: {key: float(np.mean(vals)) if vals else 0.0 for key, vals in errors[joint].items()}
                for joint in errors}

    # ---------- optimizers --------------
    def configure_optimizers(self):
        # optimizer
        opt_cfg = self.optimizer_config
        opt_class = {"adamw": torch.optim.AdamW,
                     "adam": torch.optim.Adam,
                     "sgd": torch.optim.SGD,
                     "rmsprop": torch.optim.RMSprop}[opt_cfg["name"].lower()]
        optimizer = opt_class(self.parameters(), **opt_cfg["params"])

        # scheduler (optional)
        sched_cfg = self.scheduler_config
        if sched_cfg["name"] in ("", "none", None):
            return {"optimizer": optimizer}

        sched_class = {
            "step": torch.optim.lr_scheduler.StepLR,
            "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
            "cosine_warmup": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "one_cycle": torch.optim.lr_scheduler.OneCycleLR
        }[sched_cfg["name"].lower()]
        scheduler = sched_class(optimizer, **sched_cfg["params"])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": sched_cfg.get("interval", "epoch"),
                "frequency": sched_cfg.get("frequency", 1),
                "monitor": sched_cfg.get("monitor", "val_loss")
            }
        }
