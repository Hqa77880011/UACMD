# UACMD

Implementation of **Uncertainty-aware cross-modal distillation for robust multimodal speech affect recognition with missing modalities.**

## Install

Use Python 3.11 in an isolated environment. Run from the repository root:

```bash
python -m pip install -e .
```

## Data

Obtain aligned text/audio/video benchmark features through [CMU MultimodalSDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK), [MELD](https://github.com/declare-lab/MELD) or the IEMOCAP release. NPZ fields are `x: (N,M,D)`, `y`, `group`, `split: train/val/test`, and optional `available: (N,M)` and `sentiment: (N,)`. Pool/pad benchmark features to fixed per-modality vectors with the same declared preprocessing for every method. Groups identify independent speakers, sessions or conversations. Set `classes` for the task and retain at least one available modality.

## Workflow

```bash
python -m uacd.stress --input data/features.npz --output data/missing50.npz --missing-rate 0.5
python -m uacd prepare --input data/missing50.npz
python -m uacd train
python -m uacd run
python -m uacd evaluate
python -m uacd plot
```

Main commands accept `--config config.json` and `--output runs/default`. Each output directory records `settings.json`. Keep independent seeds, datasets and conditions in separate directories.

## Configuration

`width` sets aligned modality dimensions. `teacher_epochs` fits complete-view supervision before student training. `missing_rate,curriculum_epochs` define progressive masking. `corruption_rate,noise_scale` define observed degradation. `distillation_temperature,kd_weight,feature_weight` control teacher transfer. Quality/uncertainty temperatures and margins define reliability supervision. `regression_weight` enables the supplied sentiment target. Additional optimizer, batch-size and model-width defaults are in `config.json`.

## Outputs

`model.pt` stores training state; deployment uses the student. `predictions.npz` contains class/regression outputs, reliability weights and uncertainty. `metrics.csv` reports accuracy, weighted/Macro-F1, calibration and risk; supplied sentiment targets add MAE and Pearson correlation. `risk_coverage.csv` supplies retained-set classification behavior.

## Method and comparisons

Two-layer modality encoders project to a shared representation. Reliability logits combine bounded quality and positive uncertainty, then normalize over present modalities only. The complete-view teacher is frozen after its fitting stage. Student losses combine label CE, temperature-scaled response KL, normalized feature alignment and corruption-sensitive uncertainty. Missingness/corruption flags train the student but are not inferred from final labels.

Controls: `full`, `uniform`, `no-kd`, `no-feature`, `no-uncertainty`, `no-curriculum`. Choose `--variant` and use a separate training/output directory.

```bash
python -m uacd.compare --inputs "runs/*/metrics.csv" --reference full --metrics weighted_f1 macro_f1 accuracy --unit seed
```

Comparison averages rows within independent units before bootstrap intervals and paired Wilcoxon/Holm inference. For participant/client analyses choose those independent unit columns instead of counting recordings as replicates.

