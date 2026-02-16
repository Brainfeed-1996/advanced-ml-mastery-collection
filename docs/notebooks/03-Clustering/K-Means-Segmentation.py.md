# K-Means - Image Segmentation

**Folder:** `03-Clustering`

**Last updated:** 2026-02-16 16:29:34

## Purpose

Segment images using K-Means clustering

## Key Features

- Image color space conversion
- K-Means for color segmentation
- Segmentation quality evaluation
- Comparison with other segmentation methods
- Application to real images

## How to Use

Run the notebook to segment images. Adjust K for different segmentation levels.

## Expected Outputs

Segmented images, segmentation metrics, comparison plots

## Difficulty Level

- **Intermediate**

## Prerequisites

- Python 3.10+
- Virtual environment (see docs/INSTALLATION.md)
- Required dependencies (see docs/OPTIONAL_HEAVY_DEPS.md)

## Running the Notebook

### Interactive
```bash
jupyter notebook 03-Clustering/K-Means-Segmentation.ipynb
```

### Headless (with outputs)
```bash
python -m jupyter nbconvert --to notebook --execute \
  03-Clustering/K-Means-Segmentation.ipynb --output K-Means-Segmentation.ipynb --output-dir 03-Clustering
```

## Troubleshooting

- **Missing outputs**: Run the notebook in Jupyter or use nbconvert
- **Memory issues**: Reduce batch size or use smaller models
- **Slow execution**: Use CPU-safe defaults or reduce complexity
- **Dependency errors**: Install required packages from requirements.txt
