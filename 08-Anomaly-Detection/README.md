# 08-Anomaly-Detection

Anomaly detection in cyber/fraud/streaming contexts.

Last refresh: **2026-02-16 10:18:30**

## Prerequisites

- Python 3.10+ recommended
- Create a venv (see `docs/INSTALLATION.md`)
- Some notebooks require heavy dependencies (see `docs/OPTIONAL_HEAVY_DEPS.md`)

If a notebook downloads weights/models on first run, expect longer execution times.

## Contents

- `Autoencoders-Network-Security.ipynb` — Autoencoder workflows (denoising / reconstruction).
- `Gradient-Boosting-Insurance.ipynb` — Notebook project (see notebook for details).
- `Isolation-Forest-Cyber.ipynb` — Notebook project (see notebook for details).
- `KNN-Recommender-Systems.ipynb` — Notebook project (see notebook for details).
- `Local-Outlier-Factor-Fraud.ipynb` — Notebook project (see notebook for details).
- `Naive-Bayes-Spam-Filter.ipynb` — Notebook project (see notebook for details).
- `Streaming-Drift-Anomaly-Calibration.ipynb` — Notebook project (see notebook for details).

## How to run

### Interactive
```bash
jupyter notebook 08-Anomaly-Detection/<notebook>.ipynb
```

### Headless (embed outputs into the notebook file)
```bash
python -m jupyter nbconvert --to notebook --execute \
  08-Anomaly-Detection/<notebook>.ipynb --output <notebook>.ipynb --output-dir 08-Anomaly-Detection
```

## Expected outputs
- Printed metrics (accuracy/ROC-AUC/MSE/etc.)
- At least one plot or table for interpretation
- For heavy notebooks: model download logs (first run) + sample inference outputs

## Troubleshooting
- **SIGKILL / OOM** (especially diffusion on CPU): reduce steps, reduce image size, or run on a GPU machine.
- **Corrupted model cache** (Transformers/Diffusers): clear HuggingFace cache (`~/.cache/huggingface`).
- **Slow runs**: prefer tiny models for validation; then enable full runs intentionally.