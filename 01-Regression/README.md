# 01-Regression

Foundations: supervised learning for continuous targets (regression).

Last refresh: **2026-02-16 16:05:59**

## Prerequisites

- Python 3.10+ recommended
- Create a venv (see `docs/INSTALLATION.md`)
- Some notebooks require heavy dependencies (see `docs/OPTIONAL_HEAVY_DEPS.md`)

If a notebook downloads weights/models on first run, expect longer execution times.

## Contents

### Notebooks (.ipynb)
- `Linear-Regression-Real-Estate.ipynb` — Notebook project (see notebook for details).
- `Logistic-Regression-Healthcare.ipynb` — Notebook project (see notebook for details).
- `Polynomial-Regression-Energy.ipynb` — Notebook project (see notebook for details).

### Python Tools (.py)
- `Linear-Regression-Real-Estate.py` — Automation tool for notebook management.
- `Logistic-Regression-Healthcare.py` — Automation tool for notebook management.
- `Polynomial-Regression-Energy.py` — Automation tool for notebook management.

## How to run

### Interactive (Jupyter)
```bash
jupyter notebook 01-Regression/<notebook>.ipynb
```

### Headless (embed outputs into the notebook file)
```bash
python -m jupyter nbconvert --to notebook --execute \
  01-Regression/<notebook>.ipynb --output <notebook>.ipynb --output-dir 01-Regression
```

### Running Python tools
```bash
cd 01-Regression
python <tool>.py
```

## Expected outputs
- Printed metrics (accuracy/ROC-AUC/MSE/etc.)
- At least one plot or table for interpretation
- For heavy notebooks: model download logs (first run) + sample inference outputs

## Troubleshooting
- **SIGKILL / OOM** (especially diffusion on CPU): reduce steps, reduce image size, or run on a GPU machine.
- **Corrupted model cache** (Transformers/Diffusers): clear HuggingFace cache (`~/.cache/huggingface`).
- **Slow runs**: prefer tiny models for validation; then enable full runs intentionally.
- **Import errors**: run `tools_fix_concatenated_imports.py` to fix concatenated imports.
- **JSON errors**: run `tools_fix_ipynb_json.py` to repair corrupted notebooks.