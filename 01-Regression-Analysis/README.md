# 01-Regression-Analysis

Regression + deeper evaluation/diagnostics (industrial baselines).

Last refresh: **2026-02-16 10:18:30**

## Prerequisites

- Python 3.10+ recommended
- Create a venv (see `docs/INSTALLATION.md`)
- Some notebooks require heavy dependencies (see `docs/OPTIONAL_HEAVY_DEPS.md`)

If a notebook downloads weights/models on first run, expect longer execution times.

## Contents

- `Linear-Regression-Real-Estate.ipynb` — Notebook project (see notebook for details).
- `Logistic-Regression-Healthcare.ipynb` — Notebook project (see notebook for details).
- `Random-Forest-Finance.ipynb` — Notebook project (see notebook for details).

## How to run

### Interactive
```bash
jupyter notebook 01-Regression-Analysis/<notebook>.ipynb
```

### Headless (embed outputs into the notebook file)
```bash
python -m jupyter nbconvert --to notebook --execute \
  01-Regression-Analysis/<notebook>.ipynb --output <notebook>.ipynb --output-dir 01-Regression-Analysis
```

## Expected outputs
- Printed metrics (accuracy/ROC-AUC/MSE/etc.)
- At least one plot or table for interpretation
- For heavy notebooks: model download logs (first run) + sample inference outputs

## Troubleshooting
- **SIGKILL / OOM** (especially diffusion on CPU): reduce steps, reduce image size, or run on a GPU machine.
- **Corrupted model cache** (Transformers/Diffusers): clear HuggingFace cache (`~/.cache/huggingface`).
- **Slow runs**: prefer tiny models for validation; then enable full runs intentionally.