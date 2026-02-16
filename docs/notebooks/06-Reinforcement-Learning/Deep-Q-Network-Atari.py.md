# DQN - Atari Game Playing

**Folder:** `06-Reinforcement-Learning`

**Last updated:** 2026-02-16 16:29:34

## Purpose

Train a Deep Q-Network to play Atari games

## Key Features

- Atari game environment setup
- DQN architecture implementation
- Experience replay
- Target network
- Training and evaluation

## How to Use

Run the notebook to train a DQN agent. This is resource-intensive.

## Expected Outputs

Trained DQN agent, game play videos, performance metrics

## Difficulty Level

- **Advanced**

## Prerequisites

- Python 3.10+
- Virtual environment (see docs/INSTALLATION.md)
- Required dependencies (see docs/OPTIONAL_HEAVY_DEPS.md)

## Running the Notebook

### Interactive
```bash
jupyter notebook 06-Reinforcement-Learning/Deep-Q-Network-Atari.ipynb
```

### Headless (with outputs)
```bash
python -m jupyter nbconvert --to notebook --execute \
  06-Reinforcement-Learning/Deep-Q-Network-Atari.ipynb --output Deep-Q-Network-Atari.ipynb --output-dir 06-Reinforcement-Learning
```

## Troubleshooting

- **Missing outputs**: Run the notebook in Jupyter or use nbconvert
- **Memory issues**: Reduce batch size or use smaller models
- **Slow execution**: Use CPU-safe defaults or reduce complexity
- **Dependency errors**: Install required packages from requirements.txt
