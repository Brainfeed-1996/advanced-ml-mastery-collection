# 02-Classification-Challenges

Challenge-style notebooks (harder variants, tuning, edge cases).

Last refresh: **2026-02-16 16:05:59**

## Prerequisites

- Python 3.10+ recommended
- Create a venv (see `docs/INSTALLATION.md`)
- Some notebooks require heavy dependencies (see `docs/OPTIONAL_HEAVY_DEPS.md`)

If a notebook downloads weights/models on first run, expect longer execution times.

## Contents

### Notebooks (.ipynb)
- `K-Means-Customer-Segmentation.ipynb` — K-means clustering for segmentation + silhouette-based K selection + profiling.
- `SVM-Image-Recognition.ipynb` — Support Vector Machine classification with tuning + diagnostics.
- `XGBoost-Customer-Churn.ipynb` — Gradient boosting for churn-style classification (XGBoost if available).

### Python Tools (.py)
- `K-Means-Customer-Segmentation.py` — Automation tool for notebook management.
- `SVM-Image-Recognition.py` — Automation tool for notebook management.
- `XGBoost-Customer-Churn.py` — Automation tool for notebook management.

## How to run

### Interactive (Jupyter)
```bash
jupyter notebook 02-Classification-Challenges/<notebook>.ipynb
```

### Headless (embed outputs into the notebook file)
```bash
python -m jupyter nbconvert --to notebook --execute \
  02-Classification-Challenges/<notebook>.ipynb --output <notebook>.ipynb --output-dir 02-Classification-Challenges
```

### Running Python tools
```bash
cd 02-Classification-Challenges
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