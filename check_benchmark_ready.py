#!/usr/bin/env python3
"""
Simple pre-flight check for the public benchmark release.
"""

from __future__ import annotations

import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def check_import(module_name: str, install_name: str | None = None) -> bool:
    try:
        importlib.import_module(module_name)
        print(f"[OK] import {module_name}")
        return True
    except Exception:
        target = install_name or module_name
        print(f"[FAIL] import {module_name}  -> install `{target}`")
        return False


def check_path(path: Path, description: str) -> bool:
    if path.exists():
        print(f"[OK] {description}: {path}")
        return True
    print(f"[WARN] missing {description}: {path}")
    return False


def main() -> int:
    print("pose2landmark-benchmark pre-flight check")
    print("=" * 48)

    required = True
    for module_name, install_name in (
        ("torch", "torch"),
        ("pytorch_lightning", "pytorch-lightning"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("yaml", "pyyaml"),
        ("h5py", "h5py"),
    ):
        required &= check_import(module_name, install_name)

    print("\nRepository files")
    required &= check_path(ROOT / "train_al.py", "training entrypoint")
    required &= check_path(ROOT / "configs" / "exps", "experiment configs")
    required &= check_path(ROOT / "lib" / "model" / "pose_baseline.py", "public baseline models")

    print("\nOptional data locations")
    check_path(ROOT / "data" / "move4d" / "MOVE4D", "MOVE4D root")
    check_path(ROOT / "data" / "amass", "AMASS root")
    check_path(ROOT / "data" / "amass" / "subjectSplit_curated.npy", "AMASS split file")

    print("\nOptional extras")
    check_import("wandb", "wandb")
    check_import("opensim", "opensim")

    if not required:
        print("\nCritical checks failed.")
        return 1

    print("\nCore checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
