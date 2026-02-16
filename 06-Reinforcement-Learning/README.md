# 06-Reinforcement-Learning

Reinforcement learning (policy gradients, Q-learning, DQN patterns).

Last refresh: **2026-02-16 10:18:30**

## Prerequisites

- Python 3.10+ recommended
- Create a venv (see `docs/INSTALLATION.md`)
- Some notebooks require heavy dependencies (see `docs/OPTIONAL_HEAVY_DEPS.md`)

If a notebook downloads weights/models on first run, expect longer execution times.

## Contents

- `Deep-Q-Network-Atari.ipynb` — Deep Q-Network (DQN) training pattern (resource heavy).
- `Feature-Engineering-Pipeline.ipynb` — Notebook project (see notebook for details).
- `Hyperparameter-Optimization-Optuna.ipynb` — Hyperparameter optimization with Optuna (study + best params).
- `Prophet-Market-Trends.ipynb` — Notebook project (see notebook for details).
- `Q-Learning-Maze-Solver.ipynb` — Notebook project (see notebook for details).
- `RL-CartPole-Agent.ipynb` — Reinforcement learning agent training (gymnasium).

## How to run

### Interactive
```bash
jupyter notebook 06-Reinforcement-Learning/<notebook>.ipynb
```

### Headless (embed outputs into the notebook file)
```bash
python -m jupyter nbconvert --to notebook --execute \
  06-Reinforcement-Learning/<notebook>.ipynb --output <notebook>.ipynb --output-dir 06-Reinforcement-Learning
```

## Expected outputs
- Printed metrics (accuracy/ROC-AUC/MSE/etc.)
- At least one plot or table for interpretation
- For heavy notebooks: model download logs (first run) + sample inference outputs

## Troubleshooting
- **SIGKILL / OOM** (especially diffusion on CPU): reduce steps, reduce image size, or run on a GPU machine.
- **Corrupted model cache** (Transformers/Diffusers): clear HuggingFace cache (`~/.cache/huggingface`).
- **Slow runs**: prefer tiny models for validation; then enable full runs intentionally.