# Notebook Execution Index (Outputs + Artefacts)

This repository contains executed Jupyter notebooks (`.ipynb`) with saved outputs so GitHub renders the full results.

Additionally, we export HTML artefacts in `docs/executed/` for a clean, shareable rendering.

## Executed notebooks

| Notebook | HTML artefact |
|---|---|
| `01-Regression/Linear-Regression-Real-Estate.ipynb` | `docs/executed/Linear-Regression-Real-Estate.html` |
| `01-Regression/Logistic-Regression-Healthcare.ipynb` | `docs/executed/Logistic-Regression-Healthcare.html` |
| `01-Regression/Polynomial-Regression-Energy.ipynb` | `docs/executed/Polynomial-Regression-Energy.html` |
| `02-Classification/Random-Forest-Finance.ipynb` | `docs/executed/Random-Forest-Finance.html` |
| `02-Classification/SVM-Handwritten-Digits.ipynb` | `docs/executed/SVM-Handwritten-Digits.html` |
| `02-Classification/XGBoost-Customer-Churn.ipynb` | `docs/executed/XGBoost-Customer-Churn.html` |
| `03-Clustering/K-Means-Segmentation.ipynb` | `docs/executed/K-Means-Segmentation.html` |
| `03-Clustering/DBSCAN-Anomaly-Detection.ipynb` | `docs/executed/DBSCAN-Anomaly-Detection.html` |
| `03-Clustering/Hierarchical-Clustering-Genes.ipynb` | `docs/executed/Hierarchical-Clustering-Genes.html` |
| `03-Clustering-Techniques/PCA-Dimensionality-Reduction.ipynb` | `docs/executed/PCA-Dimensionality-Reduction.html` |
| `07-Time-Series/ARIMA-Sales-Forecasting.ipynb` | `docs/executed/ARIMA-Sales-Forecasting.html` |
| `08-Anomaly-Detection/Naive-Bayes-Spam-Filter.ipynb` | `docs/executed/Naive-Bayes-Spam-Filter.html` |
| `08-Anomaly-Detection/Gradient-Boosting-Insurance.ipynb` | `docs/executed/Gradient-Boosting-Insurance.html` |
| `08-Anomaly-Detection/Local-Outlier-Factor-Fraud.ipynb` | `docs/executed/Local-Outlier-Factor-Fraud.html` |
| `08-Anomaly-Detection/KNN-Recommender-Systems.ipynb` | `docs/executed/KNN-Recommender-Systems.html` |
| `08-Anomaly-Detection/Isolation-Forest-Cyber.ipynb` | `docs/executed/Isolation-Forest-Cyber.html` |
| `08-Anomaly-Detection/Autoencoders-Network-Security.ipynb` | `docs/executed/Autoencoders-Network-Security.html` |

## How execution works

- Kernel: `notebooks-py311`
- Execution tool: `jupyter nbconvert --execute`
- Policy:
  - notebooks are executed **in-place** to preserve saved outputs
  - HTML exports go to `docs/executed/`

## Notes

- Some notebooks (CV / Diffusion / large Transformers / RL Atari) are heavier to execute reliably on CPU-only environments.
  The goal is still to make **every notebook** produce visible outputs on GitHub; heavy ones may be adapted to CPU-safe settings
  while keeping the same advanced concepts.
