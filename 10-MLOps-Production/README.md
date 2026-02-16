# 10-MLOps-Production


Auto-generated: 2026-02-16 01:25:44

## Contents

- `Federated-Learning-Privacy.ipynb` — Federated learning simulation and privacy notes.
- `ML-Edge-Computing-TinyML.ipynb` — Notebook project (see notebook for details).
- `Transformer-Attention-Mechanisms.ipynb` — Notebook project (see notebook for details).

## How to run

```bash
jupyter notebook 10-MLOps-Production/<notebook>.ipynb
```

Or execute headlessly (exports outputs into the same file):

```bash
python -m jupyter nbconvert --to notebook --execute \
  10-MLOps-Production/<notebook>.ipynb --output <notebook>.ipynb --output-dir 10-MLOps-Production
```

## Notes
- Some notebooks download models/weights on first run (Transformers/YOLO/Diffusers).
- For repeatable runs, pin dependencies and set seeds.