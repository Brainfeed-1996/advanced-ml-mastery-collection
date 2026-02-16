# 09-Generative-AI


Auto-generated: 2026-02-16 09:50:14

## Contents

- `CharNGram-LanguageModel.ipynb` — Notebook project (see notebook for details).
- `GAN-Face-Generation.ipynb` — Notebook project (see notebook for details).
- `Named-Entity-Recognition-Spacy.ipynb` — NER pipeline and evaluation.
- `Object-Detection-YOLOv8.ipynb` — Object detection pipeline with YOLO (weights download).
- `Stable-Diffusion-Prompt-Eng.ipynb` — Text-to-image generation with diffusion (model download).
- `Transfer-Learning-ResNet.ipynb` — Transfer learning with ResNet (weights download).
- `Variational-Autoencoders-MNIST.ipynb` — Notebook project (see notebook for details).

## How to run

```bash
jupyter notebook 09-Generative-AI/<notebook>.ipynb
```

Or execute headlessly (exports outputs into the same file):

```bash
python -m jupyter nbconvert --to notebook --execute \
  09-Generative-AI/<notebook>.ipynb --output <notebook>.ipynb --output-dir 09-Generative-AI
```

## Notes
- Some notebooks download models/weights on first run (Transformers/YOLO/Diffusers).
- For repeatable runs, pin dependencies and set seeds.