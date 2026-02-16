# DBSCAN - Anomaly Detection

**Folder:** `03-Clustering`

**Last updated:** 2026-02-16 16:29:34

## Purpose

Detect anomalies using density-based clustering (DBSCAN)

## Key Features

- Anomaly detection in high-dimensional data
- DBSCAN parameter tuning
- Outlier detection and visualization
- Comparison with other clustering methods
- Real-world anomaly detection patterns

## How to Use

Run the notebook to detect anomalies. Adjust epsilon and min_samples parameters.

## Expected Outputs

Anomaly scores, outlier detection, visualization of normal vs anomalous data

## Difficulty Level

- **Advanced**

## Prerequisites

- Python 3.10+
- Virtual environment (see docs/INSTALLATION.md)
- Required dependencies (see docs/OPTIONAL_HEAVY_DEPS.md)

## Running the Notebook

### Interactive
```bash
jupyter notebook 03-Clustering/DBSCAN-Anomaly-Detection.ipynb
```

### Headless (with outputs)
```bash
python -m jupyter nbconvert --to notebook --execute \
  03-Clustering/DBSCAN-Anomaly-Detection.ipynb --output DBSCAN-Anomaly-Detection.ipynb --output-dir 03-Clustering
```

## Troubleshooting

- **Missing outputs**: Run the notebook in Jupyter or use nbconvert
- **Memory issues**: Reduce batch size or use smaller models
- **Slow execution**: Use CPU-safe defaults or reduce complexity
- **Dependency errors**: Install required packages from requirements.txt
