"""Execute batch 3 - More notebooks with outputs"""
import subprocess, sys
from pathlib import Path

def run(cmd):
    subprocess.check_call(cmd)

def pycmd(*args):
    return [sys.executable, *args]

root = Path(__file__).resolve().parent

targets = [
    # Clustering
    "03-Clustering/DBSCAN-Anomaly-Detection.ipynb",
    "03-Clustering/Hierarchical-Clustering-Genes.ipynb",
    "03-Clustering/K-Means-Segmentation.ipynb",
    # Time series
    "07-Time-Series/ARIMA-Sales-Forecasting.ipynb",
    "07-Time-Series/WalkForward-Backtesting-Baselines.ipynb",
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

print("DONE batch 3")
