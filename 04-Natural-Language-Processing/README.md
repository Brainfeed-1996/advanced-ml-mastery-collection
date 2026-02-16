# 04-Natural-Language-Processing


Auto-generated: 2026-02-16 09:50:14

## Contents

- `CNN-Medical-Imaging.ipynb` — Convolutional neural network example pipeline.
- `GPT-Fine-Tuning-Basics.ipynb` — Language model / fine-tuning mechanics (offline-friendly).
- `RL-CartPole-Agent.ipynb` — Reinforcement learning agent training.

## How to run

```bash
jupyter notebook 04-Natural-Language-Processing/<notebook>.ipynb
```

Or execute headlessly (exports outputs into the same file):

```bash
python -m jupyter nbconvert --to notebook --execute \
  04-Natural-Language-Processing/<notebook>.ipynb --output <notebook>.ipynb --output-dir 04-Natural-Language-Processing
```

## Notes
- Some notebooks download models/weights on first run (Transformers/YOLO/Diffusers).
- For repeatable runs, pin dependencies and set seeds.