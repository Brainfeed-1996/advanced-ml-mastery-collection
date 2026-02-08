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
| `03-Clustering-Techniques/PCA-Dimensionality-Reduction.ipynb` | `docs/executed/PCA-Dimensionality-Reduction.html` |

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
