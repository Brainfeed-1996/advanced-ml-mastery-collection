from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def pycmd(*args: str) -> list[str]:
    return [sys.executable, *args]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="notebooks-py311")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--mode", choices=["light"], default="light")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    docs = root / "docs" / "executed"
    docs.mkdir(parents=True, exist_ok=True)

    # Light/CPU notebooks list (avoid deep learning, transformers, RL heavy, diffusion, yolo)
    notebooks = [
        # Regression
        "01-Regression/Linear-Regression-Real-Estate.ipynb",
        "01-Regression/Logistic-Regression-Healthcare.ipynb",
        "01-Regression/Polynomial-Regression-Energy.ipynb",
        "01-Regression-Analysis/Linear-Regression-Real-Estate.ipynb",
        "01-Regression-Analysis/Logistic-Regression-Healthcare.ipynb",
        "01-Regression-Analysis/Random-Forest-Finance.ipynb",
        # Classification
        "02-Classification/Random-Forest-Finance.ipynb",
        "02-Classification/SVM-Handwritten-Digits.ipynb",
        "02-Classification/XGBoost-Customer-Churn.ipynb",
        "02-Classification-Challenges/K-Means-Customer-Segmentation.ipynb",
        "02-Classification-Challenges/SVM-Image-Recognition.ipynb",
        "02-Classification-Challenges/XGBoost-Customer-Churn.ipynb",
        # Clustering
        "03-Clustering/DBSCAN-Anomaly-Detection.ipynb",
        "03-Clustering/Hierarchical-Clustering-Genes.ipynb",
        "03-Clustering/K-Means-Segmentation.ipynb",
        "03-Clustering-Techniques/PCA-Dimensionality-Reduction.ipynb",
        # Time series (statsmodels)
        "07-Time-Series/ARIMA-Sales-Forecasting.ipynb",
        # Anomaly/ML misc
        "08-Anomaly-Detection/Gradient-Boosting-Insurance.ipynb",
        "08-Anomaly-Detection/Isolation-Forest-Cyber.ipynb",
        "08-Anomaly-Detection/KNN-Recommender-Systems.ipynb",
        "08-Anomaly-Detection/Local-Outlier-Factor-Fraud.ipynb",
        "08-Anomaly-Detection/Naive-Bayes-Spam-Filter.ipynb",
    ]

    for rel in notebooks:
        nb = root / rel
        if not nb.exists():
            print(f"SKIP missing: {rel}")
            continue
        print(f"EXEC: {rel}")
        # Execute in-place: overwrite notebook with outputs
        run(
            pycmd(
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                str(nb),
                f"--ExecutePreprocessor.kernel_name={args.kernel}",
                f"--ExecutePreprocessor.timeout={args.timeout}",
                "--output",
                nb.name,
                "--output-dir",
                str(nb.parent),
            )
        )

        # Export to HTML (stored in docs) for extra visibility / artefacts
        out_html = docs / (nb.stem + ".html")
        run(
            pycmd(
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "html",
                str(nb),
                "--output",
                out_html.name,
                "--output-dir",
                str(docs),
            )
        )

    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
