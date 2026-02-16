# 06-Reinforcement-Learning


Auto-generated: 2026-02-16 09:50:14

## Contents

- `Deep-Q-Network-Atari.ipynb` — Deep RL atari-style pipeline (heavy).
- `Feature-Engineering-Pipeline.ipynb` — Notebook project (see notebook for details).
- `Hyperparameter-Optimization-Optuna.ipynb` — Hyperparameter optimization with Optuna.
- `Prophet-Market-Trends.ipynb` — Notebook project (see notebook for details).
- `Q-Learning-Maze-Solver.ipynb` — Notebook project (see notebook for details).
- `RL-CartPole-Agent.ipynb` — Reinforcement learning agent training.

## How to run

```bash
jupyter notebook 06-Reinforcement-Learning/<notebook>.ipynb
```

Or execute headlessly (exports outputs into the same file):

```bash
python -m jupyter nbconvert --to notebook --execute \
  06-Reinforcement-Learning/<notebook>.ipynb --output <notebook>.ipynb --output-dir 06-Reinforcement-Learning
```

## Notes
- Some notebooks download models/weights on first run (Transformers/YOLO/Diffusers).
- For repeatable runs, pin dependencies and set seeds.