# 03-Clustering

Unsupervised segmentation & clustering evaluation.

Last refresh: **2026-02-16 10:18:30**

## Prerequisites

- Python 3.10+ recommended
- Create a venv (see `docs/INSTALLATION.md`)
- Some notebooks require heavy dependencies (see `docs/OPTIONAL_HEAVY_DEPS.md`)

If a notebook downloads weights/models on first run, expect longer execution times.

## Contents

- `DBSCAN-Anomaly-Detection.ipynb` — Notebook project (see notebook for details).
- `Hierarchical-Clustering-Genes.ipynb` — Notebook project (see notebook for details).
- `K-Means-Segmentation.ipynb` — K-means clustering for segmentation + silhouette-based K selection + profiling.

## How to run

### Interactive
```bash
jupyter notebook 03-Clustering/<notebook>.ipynb
```

### Headless (embed outputs into the notebook file)
```bash
python -m jupyter nbconvert --to notebook --execute \
  03-Clustering/<notebook>.ipynb --output <notebook>.ipynb --output-dir 03-Clustering
```

## Expected outputs
- Printed metrics (accuracy/ROC-AUC/MSE/etc.)
- At least one plot or table for interpretation
- For heavy notebooks: model download logs (first run) + sample inference outputs

## Troubleshooting
- **SIGKILL / OOM** (especially diffusion on CPU): reduce steps, reduce image size, or run on a GPU machine.
- **Corrupted model cache** (Transformers/Diffusers): clear HuggingFace cache (`~/.cache/huggingface`).
- **Slow runs**: prefer tiny models for validation; then enable full runs intentionally.