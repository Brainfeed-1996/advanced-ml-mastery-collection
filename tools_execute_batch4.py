"""Batch execute more notebooks to reach 100% outputs."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

def run(cmd):
    subprocess.check_call(cmd)

def pycmd(*args):
    return [sys.executable, *args]

root = Path(__file__).resolve().parent

# More notebooks - CPU-safe ones
targets = [
    "01-Regression-Analysis/Random-Forest-Finance.ipynb",
    "02-Classification/SVM-Handwritten-Digits.ipynb",
    "02-Classification/XGBoost-Customer-Churn.ipynb",
    "03-Clustering/DBSCAN-Anomaly-Detection.ipynb",
    "03-Clustering/Hierarchical-Clustering-Genes.ipynb",
    "03-Clustering/K-Means-Segmentation.ipynb",
    "04-NLP/TFIDF-ErrorAnalysis-Classifier.ipynb",
    "07-Time-Series/WalkForward-Backtesting-Baselines.ipynb",
    "08-Anomaly-Detection/Autoencoders-Network-Security.ipynb",
    "08-Anomaly-Detection/Gradient-Boosting-Insurance.ipynb",
    "08-Anomaly-Detection/Isolation-Forest-Cyber.ipynb",
    "08-Anomaly-Detection/KNN-Recommender-Systems.ipynb",
    "08-Anomaly-Detection/Local-Outlier-Factor-Fraud.ipynb",
    "08-Anomaly-Detection/Naive-Bayes-Spam-Filter.ipynb",
    "08-Anomaly-Detection/Streaming-Drift-Anomaly-Calibration.ipynb",
    "09-Generative-AI/CharNGram-LanguageModel.ipynb",
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
            "--ExecutePreprocessor.startup_timeout=180",
            "--output", nb.name,
            "--output-dir", str(nb.parent),
        ))
        print(f"OK: {rel}")
    except Exception as e:
        print(f"FAIL: {rel} -> {e}")

print("DONE batch 4")
