# 09-Generative-AI

Generative AI: GAN/VAE, object detection, stable diffusion prompt engineering.

Last refresh: **2026-02-16 16:05:59**

## Prerequisites

- Python 3.10+ recommended
- Create a venv (see `docs/INSTALLATION.md`)
- Some notebooks require heavy dependencies (see `docs/OPTIONAL_HEAVY_DEPS.md`)

If a notebook downloads weights/models on first run, expect longer execution times.

## Contents

### Notebooks (.ipynb)
- `CharNGram-LanguageModel.ipynb` — Notebook project (see notebook for details).
- `GAN-Face-Generation.ipynb` — Notebook project (see notebook for details).
- `Named-Entity-Recognition-Spacy.ipynb` — Named Entity Recognition using spaCy + visualization.
- `Object-Detection-YOLOv8.ipynb` — Object detection with YOLOv8 (downloads weights on first run).
- `Stable-Diffusion-Prompt-Eng.ipynb` — Stable Diffusion prompt engineering (CPU-safe default + optional full model).
- `Transfer-Learning-ResNet.ipynb` — Transfer learning with pretrained ResNet (torchvision weights).
- `Variational-Autoencoders-MNIST.ipynb` — VAE training + latent sampling + reconstructions.

### Python Tools (.py)
- `GAN-Face-Generation.py` — Automation tool for notebook management.
- `Named-Entity-Recognition-Spacy.py` — Automation tool for notebook management.
- `Object-Detection-YOLOv8.py` — Automation tool for notebook management.
- `Stable-Diffusion-Prompt-Eng.py` — Automation tool for notebook management.
- `Transfer-Learning-ResNet.py` — Automation tool for notebook management.
- `Variational-Autoencoders-MNIST.py` — Automation tool for notebook management.

## How to run

### Interactive (Jupyter)
```bash
jupyter notebook 09-Generative-AI/<notebook>.ipynb
```

### Headless (embed outputs into the notebook file)
```bash
python -m jupyter nbconvert --to notebook --execute \
  09-Generative-AI/<notebook>.ipynb --output <notebook>.ipynb --output-dir 09-Generative-AI
```

### Running Python tools
```bash
cd 09-Generative-AI
python <tool>.py
```

## Expected outputs
- Printed metrics (accuracy/ROC-AUC/MSE/etc.)
- At least one plot or table for interpretation
- For heavy notebooks: model download logs (first run) + sample inference outputs

## Troubleshooting
- **SIGKILL / OOM** (especially diffusion on CPU): reduce steps, reduce image size, or run on a GPU machine.
- **Corrupted model cache** (Transformers/Diffusers): clear HuggingFace cache (`~/.cache/huggingface`).
- **Slow runs**: prefer tiny models for validation; then enable full runs intentionally.
- **Import errors**: run `tools_fix_concatenated_imports.py` to fix concatenated imports.
- **JSON errors**: run `tools_fix_ipynb_json.py` to repair corrupted notebooks.