# 03-Clustering-Techniques


Auto-generated: 2026-02-16 09:50:14

## Contents

- `BERT-Sentiment-Analysis.ipynb` — Transformer-based NLP workflow (offline baseline + optional transformers).
- `LSTM-Stock-Prediction.ipynb` — Sequence forecasting/classification with recurrent nets.
- `PCA-Dimensionality-Reduction.ipynb` — Notebook project (see notebook for details).

## How to run

```bash
jupyter notebook 03-Clustering-Techniques/<notebook>.ipynb
```

Or execute headlessly (exports outputs into the same file):

```bash
python -m jupyter nbconvert --to notebook --execute \
  03-Clustering-Techniques/<notebook>.ipynb --output <notebook>.ipynb --output-dir 03-Clustering-Techniques
```

## Notes
- Some notebooks download models/weights on first run (Transformers/YOLO/Diffusers).
- For repeatable runs, pin dependencies and set seeds.