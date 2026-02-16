# YOLOv8 - Object Detection

**Folder:** `05-Computer-Vision`

**Last updated:** 2026-02-16 16:29:34

## Purpose

Real-time object detection using YOLOv8

## Key Features

- YOLOv8 model loading and inference
- Object detection pipeline
- Real-time detection setup
- Model evaluation metrics
- Application to video streams

## How to Use

Run the notebook to detect objects. Use webcam for real-time detection.

## Expected Outputs

Detected objects, bounding boxes, confidence scores, real-time detection

## Difficulty Level

- **Advanced**

## Prerequisites

- Python 3.10+
- Virtual environment (see docs/INSTALLATION.md)
- Required dependencies (see docs/OPTIONAL_HEAVY_DEPS.md)

## Running the Notebook

### Interactive
```bash
jupyter notebook 05-Computer-Vision/YOLOv8-Object-Detection.ipynb
```

### Headless (with outputs)
```bash
python -m jupyter nbconvert --to notebook --execute \
  05-Computer-Vision/YOLOv8-Object-Detection.ipynb --output YOLOv8-Object-Detection.ipynb --output-dir 05-Computer-Vision
```

## Troubleshooting

- **Missing outputs**: Run the notebook in Jupyter or use nbconvert
- **Memory issues**: Reduce batch size or use smaller models
- **Slow execution**: Use CPU-safe defaults or reduce complexity
- **Dependency errors**: Install required packages from requirements.txt
