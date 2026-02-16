# 10-MLOps

MLOps patterns: monitoring, quantization, edge constraints.

Last refresh: **2026-02-16 16:05:59**

## Prerequisites

- Python 3.10+ recommended
- Create a venv (see `docs/INSTALLATION.md`)
- Some notebooks require heavy dependencies (see `docs/OPTIONAL_HEAVY_DEPS.md`)

If a notebook downloads weights/models on first run, expect longer execution times.

## Contents

### Notebooks (.ipynb)
- `Edge-Computing-TinyML.ipynb` — Notebook project (see notebook for details).
- `Model-Quantization-TensorRT.ipynb` — Post-training quantization patterns (TensorRT discussed; CPU fallback in notebook).
- `Prometheus-ML-Monitoring.ipynb` — Model monitoring patterns with Prometheus-like metrics.

### Python Tools (.py)
- `Edge-Computing-TinyML.py` — Automation tool for notebook management.
- `Model-Quantization-TensorRT.py` — Automation tool for notebook management.
- `Prometheus-ML-Monitoring.py` — Automation tool for notebook management.

## How to run

### Interactive (Jupyter)
```bash
jupyter notebook 10-MLOps/<notebook>.ipynb
```

### Headless (embed outputs into the notebook file)
```bash
python -m jupyter nbconvert --to notebook --execute \
  10-MLOps/<notebook>.ipynb --output <notebook>.ipynb --output-dir 10-MLOps
```

### Running Python tools
```bash
cd 10-MLOps
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