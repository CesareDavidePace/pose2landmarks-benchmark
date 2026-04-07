import sys
import os

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.model.pose_baseline import PoseBaselinePL
from lib.utils.graph_utils import build_dataset_adjacency, get_default_marker_names


def build_model(model_type, dataset_type, num_joints, frame_window):
    common = dict(
        model_type=model_type,
        frame_window=frame_window,
        num_joints_in=num_joints,
        num_joints_out=num_joints,
        dim_in=3,
        dim_out=3,
        dataset_type=dataset_type,
        metrics_fn="masked_mpjpe",
        loss_config={},
    )

    model_kwargs = {}
    if model_type in {"transformer"}:
        model_kwargs.update({"d_model": 64, "nhead": 4, "num_layers": 2, "dim_ff": 256, "drop_rate": 0.1})
    elif model_type in {"local_attn_transformer"}:
        model_kwargs.update({"d_model": 64, "nhead": 4, "num_layers": 2, "window_size": 7, "drop_rate": 0.1})
    elif model_type in {"linformer"}:
        model_kwargs.update({"d_model": 64, "nhead": 4, "num_layers": 2, "max_len": frame_window, "proj_ratio": 0.5})
    elif model_type in {"lstm", "gru", "sru"}:
        model_kwargs.update({"hidden_size": 64, "num_layers": 2, "drop_rate": 0.1})
    elif model_type in {"tcn", "dilated_cnn"}:
        model_kwargs.update({"hidden_size": 64, "num_layers": 2, "kernel_size": 3, "drop_rate": 0.1})
        if model_type == "tcn":
            model_kwargs["dilation_base"] = 2
    elif model_type == "motion_mixer":
        model_kwargs.update({"dim_feat": 64, "depth": 2, "mlp_ratio": 2.0, "drop_rate": 0.1})
    elif model_type == "simlpe":
        model_kwargs.update({"hidden_dim": 64, "drop_rate": 0.1})
    elif model_type in {"mamba"}:
        model_kwargs.update({"d_model": 64, "depth": 2, "d_state": 8, "d_conv": 4, "expand": 2, "drop_rate": 0.1})
    elif model_type in {"s4"}:
        model_kwargs.update({"d_model": 64, "depth": 2, "drop_rate": 0.1})
    elif model_type in {"stgcn", "gcn_attn"}:
        adjacency = build_dataset_adjacency(dataset_type, num_joints, verbose=False)
        model_kwargs.update({"adjacency": adjacency, "hidden_dim": 32, "num_layers": 2, "drop_rate": 0.1})
        if model_type == "stgcn":
            model_kwargs["kernel_size"] = 3
        else:
            model_kwargs["num_heads"] = 2
    elif model_type in {"cnn_attn"}:
        model_kwargs.update({"hidden_size": 64, "num_layers": 2, "kernel_size": 3, "num_heads": 2, "drop_rate": 0.1})
    elif model_type in {"cnn_mamba"}:
        model_kwargs.update({"hidden_size": 64, "num_layers": 2, "kernel_size": 3, "d_state": 8, "d_conv": 4, "expand": 2})
    elif model_type in {"cnn_s4"}:
        model_kwargs.update({"hidden_size": 64, "num_layers": 2, "kernel_size": 3, "drop_rate": 0.1})

    return PoseBaselinePL(**common, **model_kwargs)


def run_forward(model, num_joints, frame_window, device=None):
    model.eval()
    if device is not None:
        model = model.to(device)
        dummy = torch.randn(2, frame_window, num_joints, 3, device=device)
    else:
        dummy = torch.randn(2, frame_window, num_joints, 3)
    with torch.no_grad():
        out = model(dummy)
    return out.shape


def main():
    model_types = [
        "mlp", "lstm", "gru", "sru",
        "transformer", "motion_mixer", "simlpe",
        "tcn", "dilated_cnn",
        "local_attn_transformer", "linformer",
        "mamba", "s4",
        "stgcn", "gcn_attn",
        "cnn_attn", "cnn_mamba", "cnn_s4",
    ]

    failures = []
    skipped = []

    dataset_type = "move4d"
    num_joints = len(get_default_marker_names(dataset_type))
    frame_window = 16

    for model_type in model_types:
        try:
            if model_type in {"mamba", "cnn_mamba"} and not torch.cuda.is_available():
                raise ImportError("CUDA not available. Mamba requires CUDA.")
            model = build_model(model_type, dataset_type, num_joints, frame_window)
            device = "cuda" if model_type in {"mamba", "cnn_mamba"} and torch.cuda.is_available() else None
            out_shape = run_forward(model, num_joints, frame_window, device=device)
            print(f"[OK] {model_type}: output {out_shape}")
        except ImportError as exc:
            skipped.append((model_type, str(exc)))
            print(f"[SKIP] {model_type}: {exc}")
        except Exception as exc:
            failures.append((model_type, str(exc)))
            print(f"[FAIL] {model_type}: {exc}")

    for dataset_type in ("amass",):
        num_joints = len(get_default_marker_names(dataset_type))
        for model_type in ("stgcn", "gcn_attn"):
            try:
                model = build_model(model_type, dataset_type, num_joints, frame_window)
                out_shape = run_forward(model, num_joints, frame_window)
                print(f"[OK] {model_type} ({dataset_type}): output {out_shape}")
            except Exception as exc:
                failures.append((f"{model_type}:{dataset_type}", str(exc)))
                print(f"[FAIL] {model_type} ({dataset_type}): {exc}")

    if skipped:
        print("\nSkipped (missing deps):")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")

    if failures:
        print("\nFailures:")
        for name, reason in failures:
            print(f"  - {name}: {reason}")
        sys.exit(1)

    print("\nAll models passed.")


if __name__ == "__main__":
    main()
