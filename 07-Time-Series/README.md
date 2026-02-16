# 07-Time-Series

Time-series forecasting baselines and deep models.

Last refresh: **2026-02-16 16:05:59**

## Prerequisites

- Python 3.10+ recommended
- Create a venv (see `docs/INSTALLATION.md`)
- Some notebooks require heavy dependencies (see `docs/OPTIONAL_HEAVY_DEPS.md`)

If a notebook downloads weights/models on first run, expect longer execution times.

## Contents

### Notebooks (.ipynb)
- `ARIMA-Sales-Forecasting.ipynb` — Notebook project (see notebook for details).
- `LSTM-Stock-Prediction.ipynb` — Sequence forecasting with LSTM + windowing + plots.
- `Prophet-Market-Trends.ipynb` — Notebook project (see notebook for details).
- `WalkForward-Backtesting-Baselines.ipynb` — Notebook project (see notebook for details).

### Python Tools (.py)
- `ARIMA-Sales-Forecasting.py` — Automation tool for notebook management.
- `LSTM-Stock-Prediction.py` — Automation tool for notebook management.
- `Prophet-Market-Trends.py` — Automation tool for notebook management.

## How to run

### Interactive (Jupyter)
```bash
jupyter notebook 07-Time-Series/<notebook>.ipynb
```

### Headless (embed outputs into the notebook file)
```bash
python -m jupyter nbconvert --to notebook --execute \
  07-Time-Series/<notebook>.ipynb --output <notebook>.ipynb --output-dir 07-Time-Series
```

### Running Python tools
```bash
cd 07-Time-Series
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