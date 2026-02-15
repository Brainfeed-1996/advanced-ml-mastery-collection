"""Execute more notebooks to add outputs - batch 2"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

def run(cmd):
    subprocess.check_call(cmd)

def pycmd(*args):
    return [sys.executable, *args]

root = Path(__file__).resolve().parent

# Notebooks "CPU-safe" - skip transformers/RL/YOLO/diffusion heavy ones
targets = [
    # More regression/classification
    "01-Regression-Analysis/Random-Forest-Finance.ipynb",
    "02-Classification/Random-Forest-Finance.ipynb",
    "02-Classification/SVM-Handwritten-Digits.ipynb",
    "02-Classification/XGBoost-Customer-Churn.ipynb",
    # Clustering
    "03-Clustering/DBSCAN-Anomaly-Detection.ipynb",
    "03-Clustering/Hierarchical-Clustering-Genes.ipynb",
    "03-Clustering/K-Means-Segmentation.ipynb",
    # Time series (statsmodels)
    "07-Time-Series/ARIMA-Sales-Forecasting.ipynb",
    # Anomaly detection
    "08-Anomaly-Detection/Gradient-Boosting-Insurance.ipynb",
    "08-Anomaly-Detection/Isolation-Forest-Cyber.ipynb",
    "08-Anomaly-Detection/Naive-Bayes-Spam-Filter.ipynb",
]

for rel in targets:
    nb = root / rel
    if not nb.exists():
        print(f"SKIP missing: {rel}")
        continue
    print(f"EXEC: {rel}")
    try:
        run(pycmd(
            "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute", str(nb),
            "--ExecutePreprocessor.kernel_name=python3",
            "--ExecutePreprocessor.timeout=300",
            "--output", nb.name,
            "--output-dir", str(nb.parent),
        ))
        print(f"OK: {rel}")
    except Exception as e:
        print(f"FAIL: {rel} -> {e}")

print("DONE batch 2")
