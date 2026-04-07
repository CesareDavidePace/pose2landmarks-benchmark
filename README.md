# pose2landmark-benchmark

Benchmark code for 3D keypoint-to-anatomical landmark regression with multiple neural architecture families on MOVE4D and AMASS-style data.

This repository is the cleaned public project extracted from the original research workspace. It intentionally does **not** include:

- paper source files
- datasets
- local experiment logs
- W&B runs
- generated figures, tables, or ad-hoc notebooks

## Scope

The public release focuses on the reproducible benchmark core:

- training and evaluation for baseline architectures
- MOVE4D and AMASS data modules
- biomechanical losses and metrics
- robustness and error-propagation scripts
- OpenSim utilities for downstream kinematic analysis

Legacy DSTformer/GaitBERT code is intentionally excluded from this release.

## Repository Layout

```text
pose2landmark-benchmark/
├── configs/
│   ├── exps/              # experiment configs for MOVE4D and AMASS
│   └── split/             # subject splits for MOVE4D
├── lib/
│   ├── data/              # datamodules and dataset readers
│   ├── model/             # public baseline models
│   ├── isb/               # ISB joint-angle utilities
│   ├── opensim/           # OpenSim post-processing helpers
│   └── utils/             # shared utilities
├── scripts/
│   ├── test_all_baselines.py
│   ├── test_move4d_robustness.py
│   ├── test_move4d_error_propagation.py
│   └── README_ROBUSTNESS.md
├── train_al.py
├── check_benchmark_ready.py
└── environment.yml
```

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate pose2landmark
```

If you need a specific CUDA-enabled PyTorch build, replace the default `pytorch` install in `environment.yml` with the version appropriate for your machine.

## Expected Data Layout

### MOVE4D

```text
data/
└── move4d/
    └── MOVE4D/
        ├── SUBJECTS_CHARACTERISTICS.csv
        ├── VIDEO_STATISTICS.csv
        └── TDB_XXX_*/
```

Default MOVE4D configs assume `data/move4d/MOVE4D`.

### AMASS

```text
data/
└── amass/
    ├── subjectSplit_curated.npy
    ├── h5_dataset0_60_openpose/
    ├── h5_dataset1_60_openpose/
    └── ...
```

Default AMASS configs assume `data/amass` and `data/amass/subjectSplit_curated.npy`.

## Quick Start

Run the repository pre-flight check:

```bash
python check_benchmark_ready.py
```

Run a smoke test over all baseline forward passes:

```bash
python scripts/test_all_baselines.py
```

Train a MOVE4D model:

```bash
python train_al.py --config configs/exps/move4d_transformer.yaml
```

Train an AMASS model:

```bash
python train_al.py --config configs/exps/amass_lstm.yaml
```

Evaluate a trained checkpoint and export predictions:

```bash
python train_al.py \
  --config configs/exps/move4d_transformer.yaml \
  --test-only \
  --checkpoint path/to/checkpoint.ckpt
```

Predictions are written inside the checkpoint run directory under `predictions/`.

## Robustness And Error Propagation

Robustness and error-propagation analyses are available in:

- `scripts/test_move4d_robustness.py`
- `scripts/test_move4d_error_propagation.py`
- `scripts/README_ROBUSTNESS.md`

Typical outputs are written under:

- `results/robustness/`
- `results/error_prop/`

## Results Reproduction

This repository keeps the code paths required to reproduce:

- benchmark training across model families
- baseline test-time prediction export
- robustness analysis under structured corruption
- downstream biomechanical evaluation with ISB/OpenSim tooling

It does **not** ship original checkpoints or datasets. To reproduce paper-level numbers you need the same data splits, preprocessing, and trained models.

## Optional Dependencies

Some models or utilities require extra packages not needed for the default benchmark:

- `mamba-ssm` for Mamba-based models
- an S4 implementation for S4-based models
- OpenSim Python bindings for `lib/opensim/*`
- `smplx` for some visualization and mesh utilities

If those packages are missing, the core benchmark still works for the supported baseline subset.
