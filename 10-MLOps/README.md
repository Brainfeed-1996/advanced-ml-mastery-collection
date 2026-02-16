# 10-MLOps


Auto-generated: 2026-02-16 09:50:14

## Contents

- `Edge-Computing-TinyML.ipynb` — Notebook project (see notebook for details).
- `Model-Quantization-TensorRT.ipynb` — Post-training quantization patterns.
- `Prometheus-ML-Monitoring.ipynb` — Model monitoring patterns with Prometheus-style metrics.

## How to run

```bash
jupyter notebook 10-MLOps/<notebook>.ipynb
```

Or execute headlessly (exports outputs into the same file):

```bash
python -m jupyter nbconvert --to notebook --execute \
  10-MLOps/<notebook>.ipynb --output <notebook>.ipynb --output-dir 10-MLOps
```

## Notes
- Some notebooks download models/weights on first run (Transformers/YOLO/Diffusers).
- For repeatable runs, pin dependencies and set seeds.