# 04-Natural-Language-Processing

More advanced NLP/DL workflows (training loops, RL intro).

Last refresh: **2026-02-16 10:18:30**

## Prerequisites

- Python 3.10+ recommended
- Create a venv (see `docs/INSTALLATION.md`)
- Some notebooks require heavy dependencies (see `docs/OPTIONAL_HEAVY_DEPS.md`)

If a notebook downloads weights/models on first run, expect longer execution times.

## Contents

- `CNN-Medical-Imaging.ipynb` — CNN training loop + evaluation + confusion matrix.
- `GPT-Fine-Tuning-Basics.ipynb` — Fine-tuning mechanics via a tiny causal LM + sampling.
- `RL-CartPole-Agent.ipynb` — Reinforcement learning agent training (gymnasium).

## How to run

### Interactive
```bash
jupyter notebook 04-Natural-Language-Processing/<notebook>.ipynb
```

### Headless (embed outputs into the notebook file)
```bash
python -m jupyter nbconvert --to notebook --execute \
  04-Natural-Language-Processing/<notebook>.ipynb --output <notebook>.ipynb --output-dir 04-Natural-Language-Processing
```

## Expected outputs
- Printed metrics (accuracy/ROC-AUC/MSE/etc.)
- At least one plot or table for interpretation
- For heavy notebooks: model download logs (first run) + sample inference outputs

## Troubleshooting
- **SIGKILL / OOM** (especially diffusion on CPU): reduce steps, reduce image size, or run on a GPU machine.
- **Corrupted model cache** (Transformers/Diffusers): clear HuggingFace cache (`~/.cache/huggingface`).
- **Slow runs**: prefer tiny models for validation; then enable full runs intentionally.