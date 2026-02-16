# Optional / Heavy Dependencies

Some notebooks are designed for **real executions** that may download weights/models.

## Heavier notebooks

- Transformers (BERT sentiment)
- YOLOv8 object detection
- Stable Diffusion prompt engineering
- Transfer learning (ResNet pretrained weights)

## Recommended approach

1) Create a dedicated venv
2) Install base requirements
3) Add heavy deps as needed

Example:

```bash
python -m venv ml-env
# Windows:
.\ml-env\Scripts\Activate.ps1

pip install -r requirements.txt

# Heavy deps (examples)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate
pip install diffusers safetensors
pip install ultralytics
```

## Offline mode

If you want to prevent downloads:

```bash
set TRANSFORMERS_OFFLINE=1
set HF_HUB_OFFLINE=1
```

Notes:
- First run can take time due to model downloads.
- CI runners may time out unless cached.