# K-Means - Customer Segmentation

**Folder:** `02-Classification-Challenges`

**Last updated:** 2026-02-16 16:29:34

## Purpose

Segment customers into groups using K-Means clustering

## Key Features

- Customer behavior clustering
- Elbow method for K selection
- Silhouette analysis
- Cluster profiling and interpretation
- Visualization of customer segments

## How to Use

Run the notebook to segment customers. Adjust K based on business needs.

## Expected Outputs

Customer segments, cluster profiles, visualization plots

## Difficulty Level

- **Intermediate**

## Prerequisites

- Python 3.10+
- Virtual environment (see docs/INSTALLATION.md)
- Required dependencies (see docs/OPTIONAL_HEAVY_DEPS.md)

## Running the Notebook

### Interactive
```bash
jupyter notebook 02-Classification-Challenges/K-Means-Customer-Segmentation.ipynb
```

### Headless (with outputs)
```bash
python -m jupyter nbconvert --to notebook --execute \
  02-Classification-Challenges/K-Means-Customer-Segmentation.ipynb --output K-Means-Customer-Segmentation.ipynb --output-dir 02-Classification-Challenges
```

## Troubleshooting

- **Missing outputs**: Run the notebook in Jupyter or use nbconvert
- **Memory issues**: Reduce batch size or use smaller models
- **Slow execution**: Use CPU-safe defaults or reduce complexity
- **Dependency errors**: Install required packages from requirements.txt
