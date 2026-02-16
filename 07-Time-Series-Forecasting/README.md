# 07-Time-Series-Forecasting


Auto-generated: 2026-02-16 09:50:14

## Contents

- `Decision-Trees-Interpretability.ipynb` — Notebook project (see notebook for details).
- `ML-Model-Monitoring-Prometheus.ipynb` — Model monitoring patterns with Prometheus-style metrics.
- `Model-Quantization-TensorRT.ipynb` — Post-training quantization patterns.

## How to run

```bash
jupyter notebook 07-Time-Series-Forecasting/<notebook>.ipynb
```

Or execute headlessly (exports outputs into the same file):

```bash
python -m jupyter nbconvert --to notebook --execute \
  07-Time-Series-Forecasting/<notebook>.ipynb --output <notebook>.ipynb --output-dir 07-Time-Series-Forecasting
```

## Notes
- Some notebooks download models/weights on first run (Transformers/YOLO/Diffusers).
- For repeatable runs, pin dependencies and set seeds.