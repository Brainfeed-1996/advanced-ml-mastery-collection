# Hierarchical Clustering - Gene Expression

**Folder:** `03-Clustering`

**Last updated:** 2026-02-16 16:29:34

## Purpose

Cluster genes using hierarchical clustering for biological analysis

## Key Features

- Gene expression data preprocessing
- Hierarchical clustering with different linkage methods
- Dendrogram visualization and interpretation
- Cluster validation metrics
- Biological pathway analysis

## How to Use

Run the notebook to cluster genes. Interpret clusters in biological context.

## Expected Outputs

Gene clusters, dendrogram, cluster validation metrics

## Difficulty Level

- **Advanced**

## Prerequisites

- Python 3.10+
- Virtual environment (see docs/INSTALLATION.md)
- Required dependencies (see docs/OPTIONAL_HEAVY_DEPS.md)

## Running the Notebook

### Interactive
```bash
jupyter notebook 03-Clustering/Hierarchical-Clustering-Genes.ipynb
```

### Headless (with outputs)
```bash
python -m jupyter nbconvert --to notebook --execute \
  03-Clustering/Hierarchical-Clustering-Genes.ipynb --output Hierarchical-Clustering-Genes.ipynb --output-dir 03-Clustering
```

## Troubleshooting

- **Missing outputs**: Run the notebook in Jupyter or use nbconvert
- **Memory issues**: Reduce batch size or use smaller models
- **Slow execution**: Use CPU-safe defaults or reduce complexity
- **Dependency errors**: Install required packages from requirements.txt
