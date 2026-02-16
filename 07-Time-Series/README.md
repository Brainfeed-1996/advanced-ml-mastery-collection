# 07-Time-Series


Auto-generated: 2026-02-16 01:25:44

## Contents

- `ARIMA-Sales-Forecasting.ipynb` — Notebook project (see notebook for details).
- `LSTM-Stock-Prediction.ipynb` — Sequence forecasting/classification with recurrent nets.
- `Prophet-Market-Trends.ipynb` — Notebook project (see notebook for details).
- `WalkForward-Backtesting-Baselines.ipynb` — Notebook project (see notebook for details).

## How to run

```bash
jupyter notebook 07-Time-Series/<notebook>.ipynb
```

Or execute headlessly (exports outputs into the same file):

```bash
python -m jupyter nbconvert --to notebook --execute \
  07-Time-Series/<notebook>.ipynb --output <notebook>.ipynb --output-dir 07-Time-Series
```

## Notes
- Some notebooks download models/weights on first run (Transformers/YOLO/Diffusers).
- For repeatable runs, pin dependencies and set seeds.