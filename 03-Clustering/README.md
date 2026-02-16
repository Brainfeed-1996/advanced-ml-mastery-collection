# 03-Clustering

Unsupervised segmentation & clustering evaluation.

Last refresh: **2026-02-16 16:05:59**

## Prerequisites

- Python 3.10+ recommended
- Create a venv (see `docs/INSTALLATION.md`)
- Some notebooks require heavy dependencies (see `docs/OPTIONAL_HEAVY_DEPS.md`)

If a notebook downloads weights/models on first run, expect longer execution times.

## Contents

### Notebooks (.ipynb)
- `DBSCAN-Anomaly-Detection.ipynb` — Notebook project (see notebook for details).
- `Hierarchical-Clustering-Genes.ipynb` — Notebook project (see notebook for details).
- `K-Means-Segmentation.ipynb` — K-means clustering for segmentation + silhouette-based K selection + profiling.

### Python Tools (.py)
- `DBSCAN-Anomaly-Detection.py` — Automation tool for notebook management.
- `Hierarchical-Clustering-Genes.py` — Automation tool for notebook management.
- `K-Means-Segmentation.py` — Automation tool for notebook management.

## How to run

### Interactive (Jupyter)
```bash
jupyter notebook 03-Clustering/<notebook>.ipynb
```

### Headless (embed outputs into the notebook file)
```bash
python -m jupyter nbconvert --to notebook --execute \
  03-Clustering/<notebook>.ipynb --output <notebook>.ipynb --output-dir 03-Clustering
```

### Running Python tools
```bash
cd 03-Clustering
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