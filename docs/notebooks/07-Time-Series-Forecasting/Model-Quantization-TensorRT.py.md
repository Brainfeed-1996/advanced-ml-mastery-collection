# Model Quantization - TensorRT

**Folder:** `07-Time-Series-Forecasting`

**Last updated:** 2026-02-16 16:29:34

## Purpose

Quantize ML models for deployment using TensorRT

## Key Features

- Model quantization techniques
- TensorRT optimization
- Inference speed improvement
- Model accuracy preservation
- Deployment preparation

## How to Use

Run the notebook to quantize models. Use quantized models for deployment.

## Expected Outputs

Quantized models, speed benchmarks, accuracy comparison

## Difficulty Level

- **Advanced**

## Prerequisites

- Python 3.10+
- Virtual environment (see docs/INSTALLATION.md)
- Required dependencies (see docs/OPTIONAL_HEAVY_DEPS.md)

## Running the Notebook

### Interactive
```bash
jupyter notebook 07-Time-Series-Forecasting/Model-Quantization-TensorRT.ipynb
```

### Headless (with outputs)
```bash
python -m jupyter nbconvert --to notebook --execute \
  07-Time-Series-Forecasting/Model-Quantization-TensorRT.ipynb --output Model-Quantization-TensorRT.ipynb --output-dir 07-Time-Series-Forecasting
```

## Troubleshooting

- **Missing outputs**: Run the notebook in Jupyter or use nbconvert
- **Memory issues**: Reduce batch size or use smaller models
- **Slow execution**: Use CPU-safe defaults or reduce complexity
- **Dependency errors**: Install required packages from requirements.txt
