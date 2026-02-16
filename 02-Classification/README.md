# 02-Classification


Auto-generated: 2026-02-16 01:25:44

## Contents

- `Random-Forest-Finance.ipynb` — Notebook project (see notebook for details).
- `SVM-Handwritten-Digits.ipynb` — Support Vector Machine classification with tuning.
- `XGBoost-Customer-Churn.ipynb` — Gradient boosting for classification/regression (XGBoost if available).

## How to run

```bash
jupyter notebook 02-Classification/<notebook>.ipynb
```

Or execute headlessly (exports outputs into the same file):

```bash
python -m jupyter nbconvert --to notebook --execute \
  02-Classification/<notebook>.ipynb --output <notebook>.ipynb --output-dir 02-Classification
```

## Notes
- Some notebooks download models/weights on first run (Transformers/YOLO/Diffusers).
- For repeatable runs, pin dependencies and set seeds.