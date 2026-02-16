# 01-Regression


Auto-generated: 2026-02-16 09:50:14

## Contents

- `Linear-Regression-Real-Estate.ipynb` — Notebook project (see notebook for details).
- `Logistic-Regression-Healthcare.ipynb` — Notebook project (see notebook for details).
- `Polynomial-Regression-Energy.ipynb` — Notebook project (see notebook for details).

## How to run

```bash
jupyter notebook 01-Regression/<notebook>.ipynb
```

Or execute headlessly (exports outputs into the same file):

```bash
python -m jupyter nbconvert --to notebook --execute \
  01-Regression/<notebook>.ipynb --output <notebook>.ipynb --output-dir 01-Regression
```

## Notes
- Some notebooks download models/weights on first run (Transformers/YOLO/Diffusers).
- For repeatable runs, pin dependencies and set seeds.