# 08 — Anomaly Detection

Industrial-grade anomaly detection notebooks (cyber, fraud, insurance, and streaming drift).

## Notebooks

- `Isolation-Forest-Cyber.ipynb` — classical unsupervised baseline for cybersecurity-style signals
- `Local-Outlier-Factor-Fraud.ipynb` — density-based local anomaly scoring for fraud patterns
- `Gradient-Boosting-Insurance.ipynb` — supervised proxy (rare-event / anomaly-like) modeling with boosting
- `Autoencoders-Network-Security.ipynb` — reconstruction-error approach for high-dimensional security features
- `Streaming-Drift-Anomaly-Calibration.ipynb` — streaming calibration + drift-aware monitoring patterns
- `Naive-Bayes-Spam-Filter.ipynb` — lightweight baseline (text as anomaly-ish classification)

## Notes

- The repository includes **executed notebooks with outputs** when feasible (CPU-safe). Some cells may be intentionally lightweight to keep runtime reasonable.
- If you want deterministic runs, consider setting a fixed `random_state` everywhere + pin exact package versions.
