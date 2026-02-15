"""Batch execute remaining notebooks (previously valid but without outputs).

Goal: force outputs into all notebooks.
This script is intentionally resilient: it continues on failures and prints a summary.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def pycmd(*args: str) -> list[str]:
    return [sys.executable, *args]


root = Path(__file__).resolve().parent

# Notebooks reported by validate_notebooks.py as "VALID WITHOUT OUTPUTS" (40 total)
# (kept as explicit list for reproducibility)
targets = [
    r"02-Classification-Challenges/K-Means-Customer-Segmentation.ipynb",
    r"02-Classification-Challenges/SVM-Image-Recognition.ipynb",
    r"02-Classification-Challenges/XGBoost-Customer-Churn.ipynb",
    r"03-Clustering-Techniques/BERT-Sentiment-Analysis.ipynb",
    r"03-Clustering-Techniques/LSTM-Stock-Prediction.ipynb",
    r"04-Natural-Language-Processing/CNN-Medical-Imaging.ipynb",
    r"04-Natural-Language-Processing/GPT-Fine-Tuning-Basics.ipynb",
    r"04-Natural-Language-Processing/RL-CartPole-Agent.ipynb",
    r"04-NLP/BERT-Sentiment-Analysis.ipynb",
    r"04-NLP/GPT-Fine-Tuning-Basics.ipynb",
    r"04-NLP/Named-Entity-Recognition-Spacy.ipynb",
    r"05-Computer-Vision/Autoencoders-Denoising.ipynb",
    r"05-Computer-Vision/CNN-Medical-Imaging.ipynb",
    r"05-Computer-Vision/GAN-Synthetic-Data-Gen.ipynb",
    r"05-Computer-Vision/Isolation-Forest-Cybersecurity.ipynb",
    r"05-Computer-Vision/ResNet-Transfer-Learning.ipynb",
    r"05-Computer-Vision/YOLOv8-Object-Detection.ipynb",
    r"06-Reinforcement-Learning/Deep-Q-Network-Atari.ipynb",
    r"06-Reinforcement-Learning/Feature-Engineering-Pipeline.ipynb",
    r"06-Reinforcement-Learning/Hyperparameter-Optimization-Optuna.ipynb",
    r"06-Reinforcement-Learning/Prophet-Market-Trends.ipynb",
    r"06-Reinforcement-Learning/Q-Learning-Maze-Solver.ipynb",
    r"06-Reinforcement-Learning/RL-CartPole-Agent.ipynb",
    r"07-Time-Series/LSTM-Stock-Prediction.ipynb",
    r"07-Time-Series/Prophet-Market-Trends.ipynb",
    r"07-Time-Series-Forecasting/Decision-Trees-Interpretability.ipynb",
    r"07-Time-Series-Forecasting/ML-Model-Monitoring-Prometheus.ipynb",
    r"07-Time-Series-Forecasting/Model-Quantization-TensorRT.ipynb",
    r"09-Generative-AI/GAN-Face-Generation.ipynb",
    r"09-Generative-AI/Named-Entity-Recognition-Spacy.ipynb",
    r"09-Generative-AI/Object-Detection-YOLOv8.ipynb",
    r"09-Generative-AI/Stable-Diffusion-Prompt-Eng.ipynb",
    r"09-Generative-AI/Transfer-Learning-ResNet.ipynb",
    r"09-Generative-AI/Variational-Autoencoders-MNIST.ipynb",
    r"10-MLOps/Edge-Computing-TinyML.ipynb",
    r"10-MLOps/Model-Quantization-TensorRT.ipynb",
    r"10-MLOps/Prometheus-ML-Monitoring.ipynb",
    r"10-MLOps-Production/Federated-Learning-Privacy.ipynb",
    r"10-MLOps-Production/ML-Edge-Computing-TinyML.ipynb",
    r"10-MLOps-Production/Transformer-Attention-Mechanisms.ipynb",
]

ok, fail, skipped = [], [], []

for rel in targets:
    nb = root / rel
    if not nb.exists():
        print(f"SKIP missing: {rel}")
        skipped.append(rel)
        continue

    print(f"EXEC: {rel}")
    try:
        # Use a higher timeout: some notebooks download models or take longer.
        run(
            pycmd(
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                str(nb),
                "--ExecutePreprocessor.kernel_name=python3",
                "--ExecutePreprocessor.timeout=1800",
                "--ExecutePreprocessor.startup_timeout=300",
                "--output",
                nb.name,
                "--output-dir",
                str(nb.parent),
            )
        )
        print(f"OK: {rel}")
        ok.append(rel)
    except Exception as e:
        print(f"FAIL: {rel} -> {e}")
        fail.append(rel)

print("\n=== DONE batch 5 ===")
print(f"OK: {len(ok)}")
print(f"FAIL: {len(fail)}")
print(f"SKIP: {len(skipped)}")
if fail:
    print("\nFAILED LIST:")
    for x in fail:
        print(f"- {x}")
