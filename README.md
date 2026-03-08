# AttnLRP for DINOv2 – Gorilla Re-Identification

> Bachelor thesis project: Explainability for gorilla re-identification using Attention-Aware Layer-wise Relevance Propagation (LRP) on DINOv2 Vision Transformers.

---

## Overview

This project applies **AttnLRP** (Attention-Aware Layer-wise Relevance Propagation) to a **DINOv2** Vision Transformer (ViT-Giant) fine-tuned for **gorilla re-identification (Re-ID)**. The goal is to generate interpretable relevance maps that reveal which image patches the model uses when deciding whether two gorilla images show the same individual.

Key capabilities:
- **Hyperparameter sweep** – automated grid search over LRP gamma values to find the best explanation quality
- **Faithfulness evaluation** – MORF/LERF perturbation analysis to quantify how well relevance maps identify important image regions
- **Mask analysis** – measures how much relevance falls inside/outside segmentation masks (body parts)
- Supports both **fine-tuned** and **base (non-fine-tuned)** DINOv2 models

---

## Project Structure

```
attnlrp_dinov2/
├── src/bachelor_thesis/          # Main source package
│   ├── run_sweep.py              # Entry point: hyperparameter sweep
│   ├── run_faithfulness_eval.py  # Entry point: faithfulness evaluation
│   ├── run_mask_analysis.py      # Entry point: mask contribution analysis
│   ├── model_evaluation.py       # Entry point: baseline Re-ID accuracy
│   ├── generate_masks.py         # Segmentation mask generation (SAM2)
│   ├── lrp_helpers.py            # LRP composite creation & relevance computation
│   ├── basemodel.py              # TimmWrapper – model loading/wrapping
│   ├── dataset.py                # GorillaReIDDataset
│   ├── knn_helpers.py            # k-NN database & similarity lookup
│   ├── dino_patcher.py           # DINOv2 gradient-flow patches for LRP
│   ├── eval_helpers.py           # MORF/LERF faithfulness metrics
│   ├── sweep_helpers.py          # Sweep evaluation & candidate selection
│   ├── visualize.py              # Heatmap & curve plotting
│   ├── utils.py                  # Config loading (OmegaConf)
│   └── configs/
│       ├── base.yaml             # Shared configuration
│       └── experiment/
│           ├── finetuned.yaml    # Fine-tuned model + LRP/sweep params
│           └── non_finetuned.yaml
├── scripts/                      # SLURM job submission scripts
│   ├── run_sweep.sh
│   ├── run_faithfulness_eval.sh
│   ├── run_mask_analysis.sh
│   ├── run_model_eval.sh
│   └── run_mask_generation.sh
├── environment.yml               # Conda/Mamba environment
├── Dockerfile                    # Container definition
└── pyproject.toml                # Build & tool configuration
```

---

## Setup

### 1. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate research
```

> **Note:** The environment requires CUDA 12.1 and PyTorch ≥ 2.0. On macOS (Apple Silicon) the CUDA packages are automatically skipped.

### 2. Install the package

```bash
pip install -e .
```

### 3. Configure data paths

Edit `src/bachelor_thesis/configs/base.yaml` to point to your local data directories:

```yaml
data:
  dataset_dir: /path/to/gorilla/reid/dataset
  base_mask_dir: /path/to/segmentation/masks

knn:
  db_embeddings_dir: /path/to/knn/embeddings

lrp:
  db_relevances_dir: /path/to/relevance/db
```

For the fine-tuned model, also update the checkpoint path in `configs/experiment/finetuned.yaml`:

```yaml
model:
  checkpoint_path: /path/to/checkpoint.pth
```

---

## Usage

All entry-point scripts share the same interface:

```bash
python src/bachelor_thesis/<script>.py --config_name <finetuned|non_finetuned> [key=value ...]
```

The `--config_name` argument selects `configs/experiment/finetuned.yaml` or `configs/experiment/non_finetuned.yaml`. Values in the experiment config override those in `configs/base.yaml`. Any additional `key=value` arguments override individual config fields with the highest priority.

### Hyperparameter Sweep

Runs a 4-phase grid search over LRP gamma values to identify the best-performing configuration:

```bash
# Fine-tuned model with default parameters
python src/bachelor_thesis/run_sweep.py --config_name finetuned

# Override the LRP mode and number of queries per class
python src/bachelor_thesis/run_sweep.py --config_name finetuned \
    lrp.mode=proto_margin \
    sweep.queries_per_class=2
```

### Faithfulness Evaluation

Evaluates relevance maps using MORF (Most Relevant First) and LERF (Least Relevant First) patch perturbation:

```bash
python src/bachelor_thesis/run_faithfulness_eval.py --config_name finetuned

# Adjust evaluation granularity
python src/bachelor_thesis/run_faithfulness_eval.py --config_name finetuned \
    eval.patches_per_step=100
```

### Mask Analysis

Measures the fraction of relevance that falls inside segmentation masks:

```bash
python src/bachelor_thesis/run_mask_analysis.py --config_name finetuned
```

### Baseline Re-ID Accuracy

Evaluates plain k-NN Re-ID accuracy without LRP:

```bash
python src/bachelor_thesis/model_evaluation.py --config_name finetuned
```

### Running on a SLURM Cluster

Wrapper scripts under `scripts/` submit jobs to a SLURM cluster using Enroot containers:

```bash
sbatch scripts/run_sweep.sh finetuned
sbatch scripts/run_faithfulness_eval.sh finetuned lrp.mode=similarity
```

The first argument is always the config name; any subsequent arguments are forwarded as config overrides.

---

## Configuration

Configuration is managed with **OmegaConf** (YAML + dot-notation overrides). The hierarchy is:

```
base.yaml  ←  experiment/<config_name>.yaml  ←  CLI overrides
```

### Key configuration fields

| Key | Default | Description |
|---|---|---|
| `model.backbone` | `vit_giant_patch14_dinov2.lvd142m` | TIMM model identifier |
| `model.finetuned` | `true` / `false` | Whether to load a fine-tuned checkpoint |
| `model.checkpoint_path` | — | Path to `.pth` checkpoint (finetuned only) |
| `model.img_size` | `518` | Input image size (pixels) |
| `lrp.mode` | `soft_knn_margin_all` | LRP scoring mode (see below) |
| `lrp.conv_gamma` | mode-dependent | Gamma for convolutional layers |
| `lrp.lin_gamma` | mode-dependent | Gamma for linear layers |
| `lrp.temp` | mode-dependent | Temperature for soft-KNN modes |
| `lrp.topk` | mode-dependent | Top-K neighbours for topk modes |
| `knn.k` | `5` | Number of nearest neighbours |
| `knn.distance_metric` | `euclidean` | Distance metric for k-NN |
| `sweep.queries_per_class` | `4` | Number of query images per class during sweep |
| `eval.patches_per_step` | `50` | Patches removed per perturbation step |
| `data.analysis_split` | `test` | Dataset split used for analysis |

### LRP modes

| Mode | Description |
|---|---|
| `soft_knn_margin_all` | Soft-margin score over all k-NN neighbours |
| `soft_knn_margin_topk` | Same as above, but only considers the top-K most similar neighbours |
| `proto_margin` | Uses a class prototype as the positive anchor |
| `similarity` | Raw cosine similarity to the nearest neighbour |

Mode-specific parameters (`temp`, `topk`, `conv_gamma`, `lin_gamma`) are defined per mode in `lrp_params` inside the experiment YAML and are automatically resolved at runtime.

---

## Experiment Workflow

```
1. Hyperparameter Sweep (run_sweep.py)
   ├── Phase 1 – Grid sweep on train split
   ├── Phase 2 – Select top-10 candidates
   ├── Phase 3 – Evaluate top-10 on validation split
   └── Phase 4 – Accept/reject best candidate (generalisation threshold: 35 %)

2. Faithfulness Evaluation (run_faithfulness_eval.py)
   ├── Generate segmentation masks (SAM2, if not cached)
   ├── Compute LRP relevance maps
   ├── Perturb patches in MORF / LERF / Random order
   └── Plot accuracy-vs-perturbation curves → WandB

3. Mask Analysis (run_mask_analysis.py)
   ├── Load precomputed relevance maps
   └── Report positive/negative relevance fractions per different mask categories
```

---

## Requirements

- Python 3.10
- PyTorch ≥ 2.0 with CUDA 12.1
- See `environment.yml` for the full dependency list

Do not forget to set up wandb
