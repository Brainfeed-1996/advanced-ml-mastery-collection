# 05-Computer-Vision


Auto-generated: 2026-02-16 01:25:44

## Contents

- `Autoencoders-Denoising.ipynb` — Notebook project (see notebook for details).
- `CNN-Medical-Imaging.ipynb` — Convolutional neural network example pipeline.
- `GAN-Synthetic-Data-Gen.ipynb` — Notebook project (see notebook for details).
- `Isolation-Forest-Cybersecurity.ipynb` — Notebook project (see notebook for details).
- `ResNet-Transfer-Learning.ipynb` — Transfer learning with ResNet (weights download).
- `YOLOv8-Object-Detection.ipynb` — Object detection pipeline with YOLO (weights download).

## How to run

```bash
jupyter notebook 05-Computer-Vision/<notebook>.ipynb
```

Or execute headlessly (exports outputs into the same file):

```bash
python -m jupyter nbconvert --to notebook --execute \
  05-Computer-Vision/<notebook>.ipynb --output <notebook>.ipynb --output-dir 05-Computer-Vision
```

## Notes
- Some notebooks download models/weights on first run (Transformers/YOLO/Diffusers).
- For repeatable runs, pin dependencies and set seeds.