# 03-Clustering


Auto-generated: 2026-02-16 01:25:44

## Contents

- `DBSCAN-Anomaly-Detection.ipynb` — Notebook project (see notebook for details).
- `Hierarchical-Clustering-Genes.ipynb` — Notebook project (see notebook for details).
- `K-Means-Segmentation.ipynb` — K-means clustering for segmentation + evaluation.

## How to run

```bash
jupyter notebook 03-Clustering/<notebook>.ipynb
```

Or execute headlessly (exports outputs into the same file):

```bash
python -m jupyter nbconvert --to notebook --execute \
  03-Clustering/<notebook>.ipynb --output <notebook>.ipynb --output-dir 03-Clustering
```

## Notes
- Some notebooks download models/weights on first run (Transformers/YOLO/Diffusers).
- For repeatable runs, pin dependencies and set seeds.