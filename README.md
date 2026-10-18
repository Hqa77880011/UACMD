# UACMD

Implementation of **Uncertainty-aware cross-modal distillation for robust multimodal speech affect recognition with missing modalities.**

## Installation

Requires Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3 and SciPy 1.15.3. Verification covers Python syntax, JSON/TOML and README CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

Obtain aligned text/audio/video benchmark features through [CMU MultimodalSDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK), [MELD](https://github.com/declare-lab/MELD) or the IEMOCAP release. NPZ fields are `x: (N,M,D)`, `y`, `group`, `split: train/val/test`, and optional `available: (N,M)` and `sentiment: (N,)`. Pool/pad benchmark features to fixed per-modality vectors with the same declared preprocessing for every method. Groups identify independent speakers, sessions or conversations. Set `classes` for the task and retain at least one available modality.

```bash
python -m uacd.stress --input data/features.npz --output data/missing50.npz --missing-rate 0.5
python -m uacd prepare --input data/missing50.npz --config config.json --output runs/full
```

## Training and inference

```bash
python -m uacd train --config config.json --output runs/full
python -m uacd run --config config.json --output runs/full
```

`config.json` records the method settings and seed `13`; the resolved configuration is saved as `runs/full/settings.json`. Keep separate output directories for different seeds and variants.

## Evaluation

```bash
python -m uacd evaluate --config config.json --output runs/full
```

## Ablations

Supported controls: `full`, `uniform`, `no-kd`, `no-feature`, `no-uncertainty`, `no-curriculum`.

Repeat the same data and configuration in a separate directory, for example:

```bash
python -m uacd prepare --input data/missing50.npz --config config.json --output runs/uniform --variant uniform
python -m uacd train --config config.json --output runs/uniform --variant uniform
python -m uacd run --config config.json --output runs/uniform --variant uniform
python -m uacd evaluate --config config.json --output runs/uniform --variant uniform
```

## Outputs

All main outputs are written below `runs/full/`.

`model.pt` stores training state; deployment uses the student. `predictions.npz` contains class/regression outputs, reliability weights and uncertainty. `metrics.csv` reports accuracy, weighted/Macro-F1, calibration and risk; supplied sentiment targets add MAE and Pearson correlation. `risk_coverage.csv` supplies retained-set classification behavior.

```bash
python -m uacd plot --config config.json --output runs/full
```
