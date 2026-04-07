
import os
# set order of GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from lib.data.move4d_data_module import MOVE4DDataModule
from lib.data.amass.amass_data_module import AMASSDataModule
from lib.model.pose_baseline import PoseBaselinePL
from lib.utils.graph_utils import build_dataset_adjacency
import torch
torch.set_float32_matmul_precision('medium')
# import argparse
import yaml
import argparse
import time
import json
import torch.nn as nn

def load_config(config_path="config.yaml"):
    """Loads training configuration from a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def init_weights_xavier(m):
    """Applies Xavier initialization to Conv and Linear layers."""
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def print_versioning():
    # Print versions in green
    print("\033[92m")
    print("Versions:")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch Lightning: {pl.__version__}")
    print("\033[0m")


def validate_gpu_config(cfg):
    """Validate GPU configuration and provide recommendations."""
    gpus = cfg["gpus"]
    batch_size = cfg["batch_size"]
    
    if not isinstance(gpus, list):
        raise ValueError("gpus must be a list of GPU IDs")
    
    if len(gpus) == 0:
        raise ValueError("At least one GPU must be specified")
    
    # Check if GPUs are available
    available_gpus = torch.cuda.device_count()
    print(f"\033[92mAvailable GPUs: {available_gpus}\033[0m")
    
    for gpu_id in gpus:
        if gpu_id >= available_gpus:
            raise ValueError(f"GPU {gpu_id} not available. Only {available_gpus} GPUs detected.")
    
    # Multi-GPU recommendations
    if len(gpus) > 1:
        print(f"\033[93mMulti-GPU Training Configuration:\033[0m")
        print(f"  - Using {len(gpus)} GPUs: {gpus}")
        print(f"  - Total batch size: {batch_size}")
        print(f"  - Effective batch size per GPU: {batch_size // len(gpus)}")
        print(f"  - Strategy: {cfg.get('strategy', 'ddp')}")
        
        if batch_size % len(gpus) != 0:
            print(f"\033[93mWarning: Batch size ({batch_size}) is not evenly divisible by number of GPUs ({len(gpus)})\033[0m")
            print(f"         This may cause uneven GPU utilization.")
        
        if batch_size // len(gpus) < 4:
            print(f"\033[93mWarning: Effective batch size per GPU is very small ({batch_size // len(gpus)})\033[0m")
            print(f"         Consider increasing total batch size or using fewer GPUs.")
    
    return True


def normalize_model_type(model_type):
    if model_type is None:
        return "mlp"
    norm = model_type.lower().replace("-", "_")
    if norm in {"motionmixer", "motion_mixer"}:
        return "motion_mixer"
    if norm in {"simlpe", "simlpe_baseline"}:
        return "simlpe"
    if norm in {"dilatedcnn"}:
        return "dilated_cnn"
    if norm in {"local_attn", "local_attention", "efficient_local_attn"}:
        return "local_attn_transformer"
    if norm in {"linformer_transformer"}:
        return "linformer"
    if norm in {"st_gcn"}:
        return "stgcn"
    if norm in {"gcn_attention"}:
        return "gcn_attn"
    if norm in {"cnn_attention"}:
        return "cnn_attn"
    if norm in {"ssm_mamba", "mamba_ssm"}:
        return "mamba"
    if norm in {"ssm_s4", "s4_ssm"}:
        return "s4"
    if norm in {"cnn_ssm_mamba"}:
        return "cnn_mamba"
    if norm in {"cnn_ssm_s4"}:
        return "cnn_s4"
    return norm


def get_frame_window(cfg):
    return cfg.get("frame_window", cfg.get("maxlen", 1))


def build_baseline_model(cfg, loss_config):
    model_type = normalize_model_type(cfg.get("model_type", "mlp"))
    if model_type in {"gaitbert", "dstformer", "dst_former"}:
        raise ValueError(
            "Legacy DSTformer configs are not part of the public benchmark release. "
            "Use a baseline model_type from pose_baseline.py/baselines_extra.py "
            "(e.g., mlp, transformer, lstm, gru, tcn, stgcn, cnn_attn, ...)."
        )
    frame_window = get_frame_window(cfg)
    dim_in = cfg.get("dim_in", 3)
    dim_out = cfg.get("dim_out", 3)
    maxlen = cfg.get("maxlen", frame_window)

    if model_type in {"transformer", "motion_mixer", "simlpe", "local_attn_transformer", "linformer"} and frame_window != maxlen:
        print(f"\033[93mWarning: frame_window ({frame_window}) != maxlen ({maxlen}) for {model_type}.\033[0m")

    model_hparams = {}
    if model_type == "transformer":
        dim_feat = cfg.get("dim_feat", 256)
        mlp_ratio = cfg.get("mlp_ratio", 4.0)
        model_hparams.update({
            "d_model": dim_feat,
            "nhead": cfg.get("num_heads", 8),
            "num_layers": cfg.get("depth", 6),
            "dim_ff": int(dim_feat * mlp_ratio),
            "drop_rate": cfg.get("dropout", 0.1),
        })
    elif model_type == "local_attn_transformer":
        model_hparams.update({
            "d_model": cfg.get("dim_feat", 256),
            "nhead": cfg.get("num_heads", 8),
            "num_layers": cfg.get("depth", 4),
            "window_size": cfg.get("attn_window", 9),
            "drop_rate": cfg.get("dropout", 0.1),
        })
    elif model_type == "linformer":
        model_hparams.update({
            "d_model": cfg.get("dim_feat", 256),
            "nhead": cfg.get("num_heads", 8),
            "num_layers": cfg.get("depth", 4),
            "max_len": frame_window,
            "proj_ratio": cfg.get("linformer_proj_ratio", 0.25),
            "drop_rate": cfg.get("dropout", 0.1),
        })
    elif model_type in {"lstm", "gru", "sru"}:
        model_hparams.update({
            "hidden_size": cfg.get("rnn_hidden_size", cfg.get("dim_feat", 128)),
            "num_layers": cfg.get("rnn_num_layers", cfg.get("depth", 2)),
            "drop_rate": cfg.get("dropout", 0.1),
        })
    elif model_type in {"tcn", "dilated_cnn"}:
        model_hparams.update({
            "hidden_size": cfg.get("tcn_hidden_size", cfg.get("dim_feat", 256)),
            "num_layers": cfg.get("tcn_num_layers", cfg.get("depth", 4)),
            "kernel_size": cfg.get("tcn_kernel_size", 3),
            "drop_rate": cfg.get("dropout", 0.1),
        })
        if model_type == "tcn":
            model_hparams["dilation_base"] = cfg.get("tcn_dilation_base", 2)
    elif model_type == "mlp":
        model_hparams.update({
            "drop_rate": cfg.get("dropout", 0.1),
            "use_meta": cfg.get("use_meta", False),
        })
    elif model_type == "motion_mixer":
        model_hparams.update({
            "dim_feat": cfg.get("dim_feat", 256),
            "depth": cfg.get("depth", 4),
            "mlp_ratio": cfg.get("mlp_ratio", 4.0),
            "drop_rate": cfg.get("dropout", 0.1),
        })
    elif model_type == "simlpe":
        model_hparams.update({
            "hidden_dim": cfg.get("dim_feat", 256),
            "drop_rate": cfg.get("dropout", 0.1),
        })
    elif model_type == "mamba":
        model_hparams.update({
            "d_model": cfg.get("dim_feat", 256),
            "depth": cfg.get("depth", 4),
            "d_state": cfg.get("mamba_d_state", 16),
            "d_conv": cfg.get("mamba_d_conv", 4),
            "expand": cfg.get("mamba_expand", 2),
            "drop_rate": cfg.get("dropout", 0.1),
        })
    elif model_type == "s4":
        model_hparams.update({
            "d_model": cfg.get("dim_feat", 256),
            "depth": cfg.get("depth", 4),
            "drop_rate": cfg.get("dropout", 0.1),
        })
    elif model_type in {"stgcn", "gcn_attn"}:
        adjacency = build_dataset_adjacency(
            cfg.get("dataset", "move4d"),
            cfg["num_joints_out"],
            marker_names=cfg.get("graph_marker_names"),
            verbose=True,
        )
        model_hparams.update({
            "adjacency": adjacency,
            "hidden_dim": cfg.get("dim_feat", 128),
            "num_layers": cfg.get("depth", 3),
            "drop_rate": cfg.get("dropout", 0.1),
        })
        if model_type == "stgcn":
            model_hparams["kernel_size"] = cfg.get("tcn_kernel_size", 3)
        else:
            model_hparams["num_heads"] = cfg.get("num_heads", 4)
    elif model_type == "cnn_attn":
        model_hparams.update({
            "hidden_size": cfg.get("dim_feat", 256),
            "num_layers": cfg.get("depth", 2),
            "kernel_size": cfg.get("tcn_kernel_size", 3),
            "num_heads": cfg.get("num_heads", 4),
            "drop_rate": cfg.get("dropout", 0.1),
        })
    elif model_type == "cnn_mamba":
        model_hparams.update({
            "hidden_size": cfg.get("dim_feat", 256),
            "num_layers": cfg.get("depth", 2),
            "kernel_size": cfg.get("tcn_kernel_size", 3),
            "d_state": cfg.get("mamba_d_state", 16),
            "d_conv": cfg.get("mamba_d_conv", 4),
            "expand": cfg.get("mamba_expand", 2),
            "drop_rate": cfg.get("dropout", 0.1),
        })
    elif model_type == "cnn_s4":
        model_hparams.update({
            "hidden_size": cfg.get("dim_feat", 256),
            "num_layers": cfg.get("depth", 2),
            "kernel_size": cfg.get("tcn_kernel_size", 3),
            "drop_rate": cfg.get("dropout", 0.1),
        })
    else:
        raise ValueError(f"Unknown model_type '{model_type}' for baselines")

    return PoseBaselinePL(
        model_type=model_type,
        frame_window=frame_window,
        num_joints_in=cfg["num_joints_in"],
        num_joints_out=cfg["num_joints_out"],
        dim_in=dim_in,
        dim_out=dim_out,
        metrics_fn=cfg["metrics_fn"],
        optimizer_config=cfg["optimizer_config"],
        scheduler_config=cfg["scheduler_config"],
        loss_config=loss_config,
        dataset_type=cfg.get("dataset", "move4d").lower(),
        **model_hparams,
    )


def get_core_model(model):
    return model.model if hasattr(model, "model") else model


def resolve_eval_device(cfg):
    device_pref = cfg.get("model_eval_device", "cpu")
    if device_pref == "cuda" and torch.cuda.is_available():
        if len(cfg.get("gpus", [])) > 1:
            print("\033[93mModel eval on CUDA disabled for multi-GPU; using CPU.\033[0m")
            return torch.device("cpu")
        return torch.device("cuda:0")
    return torch.device("cpu")


def make_dummy_input(cfg):
    frame_window = get_frame_window(cfg)
    dim_in = cfg.get("dim_in", 3)
    batch_size = cfg.get("model_eval_batch_size", 1)
    x = torch.randn(
        batch_size,
        frame_window,
        cfg["num_joints_in"],
        dim_in,
    )
    # Add meta tensor if use_meta is enabled (for MLP baseline)
    model_type = normalize_model_type(cfg.get("model_type", "mlp"))
    use_meta = cfg.get("use_meta", False)
    if model_type == "mlp" and use_meta:
        meta = torch.randn(batch_size, frame_window, 3)  # height, weight, sex
        return (x, meta)
    return (x,)


def estimate_flops(model, inputs):
    flops = None
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, inputs).total()
    except Exception:
        try:
            from thop import profile
            flops, _ = profile(model, inputs=inputs, verbose=False)
        except Exception:
            flops = None
    return flops


def compute_model_stats(model, cfg):
    core = get_core_model(model)
    device = resolve_eval_device(cfg)
    inputs = make_dummy_input(cfg)

    total_params = sum(p.numel() for p in core.parameters())
    trainable_params = sum(p.numel() for p in core.parameters() if p.requires_grad)

    original_device = next(core.parameters()).device if total_params > 0 else torch.device("cpu")
    original_mode = core.training
    core.eval()
    core.to(device)
    inputs = tuple(t.to(device) for t in inputs)

    flops = None
    elapsed = None
    try:
        flops = estimate_flops(core, inputs)

        warmup = int(cfg.get("model_eval_warmup", 10))
        iters = int(cfg.get("model_eval_iters", 50))
        with torch.no_grad():
            for _ in range(warmup):
                _ = core(*inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iters):
                _ = core(*inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
    except RuntimeError as exc:
        print(f"\033[93mSkipping model stats forward pass: {exc}\033[0m")

    core.to(original_device)
    if original_mode:
        core.train()

    avg_time_ms = (elapsed / max(1, int(cfg.get("model_eval_iters", 50)))) * 1000.0 if elapsed is not None else None
    stats = {
        "params_total": total_params,
        "params_trainable": trainable_params,
        "model_size_mb": (total_params * 4) / (1024 ** 2),
        "inference_time_ms": avg_time_ms,
        "flops": flops,
        "eval_device": str(device),
    }
    return stats


def print_model_stats(stats):
    if not stats:
        return
    flops = stats["flops"]
    flops_str = f"{flops:.0f}" if isinstance(flops, (int, float)) else "n/a"
    print("\033[92mModel stats:\033[0m")
    print(f"  - Params (total/trainable): {stats['params_total']} / {stats['params_trainable']}")
    print(f"  - Model size (MB): {stats['model_size_mb']:.2f}")
    print(f"  - FLOPs: {flops_str}")
    if stats["inference_time_ms"] is None:
        print(f"  - Inference time (ms): n/a on {stats['eval_device']}")
    else:
        print(f"  - Inference time (ms): {stats['inference_time_ms']:.3f} on {stats['eval_device']}")


def save_model_stats(stats, trainer, cfg):
    if not stats:
        return
    output_dir = None
    if trainer.logger and hasattr(trainer.logger, "log_dir"):
        output_dir = trainer.logger.log_dir
    if not output_dir:
        output_dir = cfg.get("wandb_save_dir", os.getcwd())

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model_stats.json")
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\033[92mSaved model stats to: {output_path}\033[0m")

    if trainer.logger and hasattr(trainer.logger, "log_metrics"):
        loggable = {
            "model/params_total": stats["params_total"],
            "model/params_trainable": stats["params_trainable"],
            "model/model_size_mb": stats["model_size_mb"],
        }
        if stats["inference_time_ms"] is not None:
            loggable["model/inference_time_ms"] = stats["inference_time_ms"]
        if stats["flops"] is not None:
            loggable["model/flops"] = stats["flops"]
        trainer.logger.log_metrics(loggable, step=0)



def run_robust_testing(checkpoint_path, cfg, data_module, use_single_gpu=True):
    """
    Robust testing function that handles distributed training issues.
    Falls back to single GPU testing if distributed testing fails.
    """
    test_results = None
    
    try:
        print(f"\033[93mAttempting to load model from checkpoint: {checkpoint_path}\033[0m")
        
        # Set predictions output directory
        predictions_dir = os.path.join(
            os.path.dirname(checkpoint_path), 
            "..", 
            "predictions"
        )
        predictions_dir = os.path.abspath(predictions_dir)
        
        # Load baseline model checkpoint
        model = PoseBaselinePL.load_from_checkpoint(
            checkpoint_path,
            map_location="cuda:0" if use_single_gpu else None,
            predictions_output_dir=predictions_dir
        )
        
        # Create a fresh trainer for testing (single GPU to avoid distributed issues)
        test_trainer = pl.Trainer(
            accelerator="gpu",
            devices=[0] if use_single_gpu else cfg["gpus"],
            logger=False,  # Disable logging for the test phase
            enable_progress_bar=True,
            enable_model_summary=False,
        )
        
        print(f"\033[92mStarting robust test phase...\033[0m")
        test_results = test_trainer.test(model, data_module)
        
        if test_results:
            print(f"\033[92mTest completed successfully!\033[0m")
            for key, value in test_results[0].items():
                print(f"  {key}: {value:.4f}")
        
        return test_results
        
    except Exception as e:
        print(f"\033[91mError during testing: {e}\033[0m")
        import traceback
        traceback.print_exc()
        return None


def main(config_path):

    # start time
    start_time = time.time()

    # Print versions
    print_versioning()

    # Load Configuration
    cfg = load_config(config_path)

    print("\033[92mConfiguration:\033[0m")
    print(yaml.dump(cfg))

    # Validate GPU configuration
    validate_gpu_config(cfg)

    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # Multi-GPU environment variables for better performance
    if len(cfg["gpus"]) > 1:
        os.environ["NCCL_DEBUG"] = "WARN"  # Reduce NCCL verbosity (INFO -> WARN)
        os.environ["PYTHONPATH"] = os.getcwd()  # Ensure proper module finding

    pl.seed_everything(cfg["seed"], workers=True)

    # Adjust num_workers for multi-GPU training
    num_workers = cfg["num_workers"]
    if len(cfg["gpus"]) > 1:
        # Reduce num_workers per GPU to avoid too many processes
        num_workers = max(1, cfg["num_workers"] // len(cfg["gpus"]))
        print(f"\033[93mAdjusted num_workers for multi-GPU: {num_workers} per GPU\033[0m")

    if cfg["dataset"].upper() == "MOVE4D":
        camera_subset_config = cfg.get("camera_subset_config")
        camera_experiment_name = cfg.get("camera_experiment_name")
        if camera_subset_config and not os.path.exists(camera_subset_config):
            print(f"\033[93mCamera config not found ({camera_subset_config}). Disabling camera subset.\033[0m")
            camera_subset_config = None
            camera_experiment_name = None
        # Instantiate DataModule
        data_module = MOVE4DDataModule(
            root_dir=cfg["data_root"], 
            batch_size=cfg["batch_size"], 
            num_workers=num_workers,
            n_frames=cfg["maxlen"],
            sample_stride=cfg["sample_stride"],
            data_stride_train=cfg["data_stride_train"],
            data_stride_test=cfg["data_stride_test"],
            split_file=cfg["split_file"],
            augmentation_config=cfg["augmentation"] if "augmentation" in cfg else None,
            anatomical_markers_group=cfg["anatomical_markers_group"],
            target_fps=cfg["fps"],
            input_type=cfg["input_type"],
            camera_subset_config=camera_subset_config,
            camera_experiment_name=camera_experiment_name,
        )
    else:
        data_module = AMASSDataModule(
            root_dir=cfg["data_root"],
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            n_frames=cfg["maxlen"],
            sample_stride=cfg["sample_stride"],
            data_stride_train=cfg["data_stride_train"],
            data_stride_test=cfg["data_stride_test"],
            split_file=cfg.get("subject_split_file"),
            augmentation_config=cfg.get("augmentation"),
            datasets_to_use=cfg.get("datasets_to_use"),
        )
            

    loss_config = {
        "lambda_mpjpe": cfg["lambda_mpjpe"],
        "lambda_scale": cfg["lambda_scale"],
        "lambda_3d_velocity": cfg["lambda_3d_velocity"],
        "lambda_3d_acceleration": cfg.get("lambda_3d_acceleration", 0.0),
        "lambda_segment_length_consistency": cfg.get("lambda_segment_length_consistency", 0.0),
        "foot_velocity_weight": cfg.get("foot_velocity_weight", 1.0),
        "foot_acceleration_weight": cfg.get("foot_acceleration_weight", cfg.get("foot_velocity_weight", 1.0)),
        "use_weighted_velocity": cfg.get("use_weighted_velocity", False),
        "lambda_bone_length_prior": cfg["lambda_bone_length_prior"],
        "lambda_bone_orientation": cfg["lambda_bone_orientation"],
        "lambda_angle": cfg["lambda_angle"],
        "lambda_mse": cfg.get("lambda_mse", 0),
        "lambda_frequency": cfg.get("lambda_frequency", 0.0),
        "lambda_temporal_consistency": cfg.get("lambda_temporal_consistency", 0.0)
    }
    
    model = build_baseline_model(cfg, loss_config)

    checkpoint_path = cfg.get("checkpoint_path")
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\033[93mWarm-start loading from checkpoint: {checkpoint_path}\033[0m")
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # Support Lightning .ckpt and plain state_dict checkpoints
        pretrained = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

        # Legacy support: some checkpoints store model weights under model_pos
        if isinstance(pretrained, dict) and "model_pos" in pretrained:
            pretrained = pretrained["model_pos"]

        # Remove DataParallel prefix and keep only matching keys
        if isinstance(pretrained, dict):
            pretrained = {k.replace("module.", ""): v for k, v in pretrained.items()}
            model_dict = model.state_dict()
            filtered = {k: v for k, v in pretrained.items() if k in model_dict and model_dict[k].shape == v.shape}
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            print(
                f"\033[92mLoaded {len(filtered)} layers from checkpoint; "
                f"missing={len(missing)}, unexpected={len(unexpected)}\033[0m"
            )
        else:
            print("\033[93mCheckpoint format not recognized as a state_dict. Skipping warm-start load.\033[0m")
    else:
        print("\033[92mTraining baseline from scratch.\033[0m")

    model_stats = compute_model_stats(model, cfg)
    print_model_stats(model_stats)

    validation = cfg["validation"]

    # Define callbacks only if validation is enabled
    callbacks = []

    if validation:
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg["patience"],
            verbose=True,
            mode="min"
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            filename="baseline-{epoch:02d}-{val_loss:.2f}",
            save_top_k=cfg["save_top_k"],
            mode="min",
            save_on_train_epoch_end=False,  # Save at the end of validation
            save_last=True  # Also save the last checkpoint
        )
    else:
        print("\033[93mSkipping validation. Using train_loss for early stopping and checkpointing.\033[0m")

        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor="train_loss",  # Monitor training loss instead of validation loss
            patience=cfg["patience"],
            verbose=True,
            mode="min"
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="train_loss",  # Save best model based on training loss
            filename="baseline-{epoch:02d}-{train_loss:.2f}",
            save_top_k=1,
            mode="min",
            save_on_train_epoch_end=True,  # Save at the end of training epoch
            save_last=True  # Also save the last checkpoint
        )

    callbacks = [early_stop_callback, checkpoint_callback]

    # Multi-GPU strategy configuration
    strategy = "auto"  # Default strategy
    if len(cfg["gpus"]) > 1:
        strategy_name = cfg.get("strategy", "ddp")
        # DDP configuration optimized for throughput
        # Why find_unused_parameters=False:
        # - Improves DDP throughput by skipping unused parameter checks
        # - Safe when ablation modes don't create unused modules (set to None instead)
        # - Only use True if you have conditional architecture that skips modules in forward()
        if strategy_name == "ddp":
            from pytorch_lightning.strategies import DDPStrategy
            ablation_mode = cfg.get("ablation_mode", "full")
            # Baseline family is safe with find_unused_parameters=True for mixed experiments
            find_unused = True
            strategy = DDPStrategy(find_unused_parameters=find_unused)
            print(f"\033[92mDDP find_unused_parameters={find_unused} (ablation_mode={ablation_mode})\033[0m")
        else:
            strategy = strategy_name
        print(f"\033[92mUsing multi-GPU training with strategy: {strategy_name}\033[0m")
        print(f"\033[92mTraining on GPUs: {cfg['gpus']}\033[0m")
    else:
        print(f"\033[92mUsing single GPU training on GPU: {cfg['gpus'][0]}\033[0m")

    # Setup Wandb Logger
    wandb_logger = None
    if cfg.get("use_wandb", False):
        wandb_project = cfg.get("wandb_project", "pose2landmark-benchmark")
        wandb_name = cfg.get("wandb_name", None)
        wandb_tags = cfg.get("wandb_tags", [])
        
        # Create a descriptive name if not provided
        if wandb_name is None:
            model_type = cfg.get("model_type", "baseline")
            dataset_type = cfg.get("dataset", "move4d")
            ablation_mode = cfg.get("ablation_mode", "full")
            depth = cfg.get("depth", "na")
            num_heads = cfg.get("num_heads", "na")
            wandb_name = f"{model_type}_{dataset_type}_{ablation_mode}_d{depth}_h{num_heads}"
        
        wandb_logger = WandbLogger(
            project=wandb_project,
            name=wandb_name,
            tags=wandb_tags,
            log_model=cfg.get("wandb_log_model", "all"),  # "all", "best", or False
            save_dir=cfg.get("wandb_save_dir", "./wandb_logs"),
            config=cfg
        )
        print(f"\033[92mWandB logging enabled - Project: {wandb_project}, Name: {wandb_name}\033[0m")
    else:
        print("\033[93mWandB logging disabled - using default CSV logger\033[0m")

    # Training Configuration
    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],
        accelerator="gpu",
        devices=cfg["gpus"],
        strategy=strategy,
        log_every_n_steps=10,
        callbacks=callbacks,  # Ensure callbacks are correctly set
        deterministic=True,
        accumulate_grad_batches=cfg["accumulate_grad_batches"],
        sync_batchnorm=len(cfg["gpus"]) > 1,  # Enable batch norm sync for multi-GPU
        logger=wandb_logger if wandb_logger else True,  # Use wandb logger if enabled, otherwise default
    )

    # save the configuration in the lightning logs
    if trainer.logger:
        trainer.logger.log_hyperparams(cfg)

    save_model_stats(model_stats, trainer, cfg)

    # Train the model
    if validation:
        trainer.fit(model, data_module)
    else:
        trainer.fit(model, data_module, val_dataloaders=None)  # ⬅ Skips validation

    # Ensure all processes finish training before proceeding to testing
    if len(cfg["gpus"]) > 1:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                current_rank = trainer.global_rank
                dist.barrier()
                # Only rank 0 prints the completion message to avoid duplicates
                if current_rank == 0:
                    print(f"\033[92mRank {current_rank}: Training completed, synchronized with other ranks\033[0m")
        except Exception as e:
            print(f"\033[93mWarning: Could not synchronize distributed processes: {e}\033[0m")

    # Determine current rank (handle both distributed and non-distributed cases)
    try:
        current_rank = trainer.global_rank
    except:
        current_rank = 0  # Default to rank 0 if not in distributed mode

    # Only perform testing on rank 0 to avoid file access issues in distributed training
    if current_rank == 0:
        # Give a small delay to ensure checkpoint file is fully written
        time.sleep(2)
        
        best_model_path = checkpoint_callback.best_model_path
        last_model_path = checkpoint_callback.last_model_path
        
        # Try to find a valid checkpoint (best first, then last, then provided checkpoint)
        checkpoint_to_use = None
        checkpoint_source = ""
        
        if best_model_path and os.path.exists(best_model_path):
            checkpoint_to_use = best_model_path
            checkpoint_source = "best"
        elif last_model_path and os.path.exists(last_model_path):
            checkpoint_to_use = last_model_path
            checkpoint_source = "last"
        elif cfg.get("checkpoint_path") and cfg["checkpoint_path"].endswith(".ckpt") and os.path.exists(cfg["checkpoint_path"]):
            checkpoint_to_use = cfg["checkpoint_path"]
            checkpoint_source = "provided"
        
        if checkpoint_to_use:
            print(f"\033[93mFound {checkpoint_source} checkpoint: {checkpoint_to_use}\033[0m")
            
            # For distributed training, skip the distributed trainer for testing
            # and go directly to robust single-GPU testing to avoid deadlocks
            if len(cfg["gpus"]) > 1:
                print(f"\033[93mUsing robust single-GPU testing for distributed training...\033[0m")
                test_results = run_robust_testing(checkpoint_to_use, cfg, data_module, use_single_gpu=True)
                
                if test_results is None:
                    print(f"\033[91mTesting failed. Manual testing may be required.\033[0m")
                    print(f"\033[91mCheckpoint available at: {checkpoint_to_use}\033[0m")
            else:
                # Single GPU training - use the original trainer
                try:
                    best_model = PoseBaselinePL.load_from_checkpoint(
                        checkpoint_to_use, 
                        map_location=f"cuda:{cfg['gpus'][0]}",
                        predictions_output_dir=os.path.join(
                            os.path.dirname(checkpoint_to_use), 
                            "..", 
                            "predictions"
                        )
                    )
                        
                    print(f"\033[92mStarting test phase with single GPU trainer...\033[0m")
                    trainer.test(best_model, data_module)
                    print(f"\033[92mTesting completed successfully!\033[0m")
                    
                except Exception as e:
                    print(f"\033[93mSingle GPU testing failed: {e}\033[0m")
                    print(f"\033[93mFalling back to robust testing...\033[0m")
                    
                    # Fallback to robust testing
                    test_results = run_robust_testing(checkpoint_to_use, cfg, data_module, use_single_gpu=True)
                    
                    if test_results is None:
                        print(f"\033[91mAll testing methods failed. Manual testing may be required.\033[0m")
                        print(f"\033[91mCheckpoint available at: {checkpoint_to_use}\033[0m")
        else:
            print("\033[91mNo valid checkpoint found for testing. Skipping test phase.\033[0m")
            print(f"\033[91mChecked paths:\033[0m")
            if best_model_path:
                print(f"  - Best model path: {best_model_path} (exists: {os.path.exists(best_model_path) if best_model_path else 'N/A'})")
            if last_model_path:
                print(f"  - Last model path: {last_model_path} (exists: {os.path.exists(last_model_path) if last_model_path else 'N/A'})")
            if cfg.get("checkpoint_path"):
                print(f"  - Config checkpoint: {cfg['checkpoint_path']} (exists: {os.path.exists(cfg['checkpoint_path'])})")
    else:
        print(f"\033[93mRank {current_rank}: Skipping test phase (only rank 0 performs testing)\033[0m")

    # Calculate and print elapsed time (only on rank 0)
    if current_rank == 0:
        # end time
        end_time = time.time()

        # calculate elapsed time 
        elapsed_time = end_time - start_time
        elapsed_time_minutes = elapsed_time // 60
        elapsed_time_seconds = elapsed_time % 60
        print(f"\033[95mElapsed time: {elapsed_time_minutes:.0f} minutes {elapsed_time_seconds:.0f} seconds\033[0m")

    # Ensure WandB finishes properly to avoid hanging
    if cfg.get("use_wandb", False):
        import wandb
        wandb.finish()
        print(f"\033[92mWandB logging finished and synced.\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pose baselines on MOVE4D/AMASS datasets")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file (YAML)")
    parser.add_argument("--test-only", action="store_true", help="Only run testing using existing checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint path for testing")
    parser.add_argument("--robust-test", action="store_true", help="Force robust single-GPU testing mode")

    args = parser.parse_args()
    
    if args.test_only:
        # Load config
        cfg = load_config(args.config)
        
        # Override checkpoint if provided
        if args.checkpoint:
            cfg["checkpoint_path"] = args.checkpoint
        
        # Initialize data module
        if cfg["dataset"].upper() == "MOVE4D":
            camera_subset_config = cfg.get("camera_subset_config")
            camera_experiment_name = cfg.get("camera_experiment_name")
            if camera_subset_config and not os.path.exists(camera_subset_config):
                print(f"\033[93mCamera config not found ({camera_subset_config}). Disabling camera subset.\033[0m")
                camera_subset_config = None
                camera_experiment_name = None
            data_module = MOVE4DDataModule(
                root_dir=cfg["data_root"], 
                batch_size=cfg["batch_size"], 
                num_workers=cfg["num_workers"] // 2,  # Reduce for testing
                n_frames=cfg["maxlen"],
                sample_stride=cfg["sample_stride"],
                data_stride_train=cfg["data_stride_train"],
                data_stride_test=cfg["data_stride_test"],
                split_file=cfg["split_file"],
                augmentation_config=cfg["augmentation"] if "augmentation" in cfg else None,
                anatomical_markers_group=cfg["anatomical_markers_group"],
                target_fps=cfg["fps"],
                input_type=cfg["input_type"],
                camera_subset_config=camera_subset_config,
                camera_experiment_name=camera_experiment_name,
            )
        else:
            data_module = AMASSDataModule(
                root_dir=cfg["data_root"],
                batch_size=cfg["batch_size"],
                num_workers=cfg["num_workers"] // 2,  # Reduce for testing
                n_frames=cfg["maxlen"],
                sample_stride=cfg["sample_stride"],
                data_stride_train=cfg["data_stride_train"],
                data_stride_test=cfg["data_stride_test"],
                split_file=cfg.get("subject_split_file"),
                augmentation_config=cfg.get("augmentation"),
                datasets_to_use=cfg.get("datasets_to_use"),
            )
        
        # Run testing only
        checkpoint_path = cfg.get("checkpoint_path")
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\033[92mRunning test-only mode with checkpoint: {checkpoint_path}\033[0m")
            test_results = run_robust_testing(checkpoint_path, cfg, data_module, use_single_gpu=True)
            if test_results:
                print(f"\033[92mTest-only mode completed successfully!\033[0m")
            else:
                print(f"\033[91mTest-only mode failed!\033[0m")
        else:
            print(f"\033[91mCheckpoint not found: {checkpoint_path}\033[0m")
            print("Please provide a valid checkpoint path using --checkpoint")
    else:
        # Run normal training + testing
        main(args.config)
        
        
# Example:
# python train_al.py --config configs/exps/move4d_transformer.yaml
