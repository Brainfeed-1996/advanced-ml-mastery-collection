"""Generate/refresh README.md files across the repository.

- Root README stays high-level.
- Each top-level module folder gets a README with:
  - purpose
  - list of notebooks + short description
  - how to run (jupyter / nbconvert)

This is deliberately deterministic and can be re-run.

Run:
  python tools_generate_readmes.py
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
STAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Minimal curated descriptions by filename keywords
DESC = {
    "K-Means": "K-means clustering for segmentation + evaluation.",
    "SVM": "Support Vector Machine classification with tuning.",
    "XGBoost": "Gradient boosting for classification/regression (XGBoost if available).",
    "BERT": "Transformer-based NLP workflow (offline baseline + optional transformers).",
    "GPT": "Language model / fine-tuning mechanics (offline-friendly).",
    "Named-Entity": "NER pipeline and evaluation.",
    "CNN": "Convolutional neural network example pipeline.",
    "LSTM": "Sequence forecasting/classification with recurrent nets.",
    "YOLO": "Object detection pipeline with YOLO (weights download).",
    "Stable-Diffusion": "Text-to-image generation with diffusion (model download).",
    "ResNet": "Transfer learning with ResNet (weights download).",
    "VAE": "Variational autoencoder training workflow.",
    "Prometheus": "Model monitoring patterns with Prometheus-style metrics.",
    "Quantization": "Post-training quantization patterns.",
    "Federated": "Federated learning simulation and privacy notes.",
    "Optuna": "Hyperparameter optimization with Optuna.",
    "CartPole": "Reinforcement learning agent training.",
    "Atari": "Deep RL atari-style pipeline (heavy).",
}


def describe(name: str) -> str:
    for k, v in DESC.items():
        if k in name:
            return v
    return "Notebook project (see notebook for details)."


def folder_readme(folder: Path) -> str:
    nbs = sorted(folder.glob("*.ipynb"))
    lines = [f"# {folder.name}\n", "", f"Auto-generated: {STAMP}", "", "## Contents", ""]
    if not nbs:
        lines.append("(no notebooks found)")
    else:
        for nb in nbs:
            lines.append(f"- `{nb.name}` — {describe(nb.stem)}")
    lines += [
        "",
        "## How to run",
        "",
        "```bash",
        f"jupyter notebook {folder.name}/<notebook>.ipynb",
        "```",
        "",
        "Or execute headlessly (exports outputs into the same file):",
        "",
        "```bash",
        "python -m jupyter nbconvert --to notebook --execute \\",
        f"  {folder.name}/<notebook>.ipynb --output <notebook>.ipynb --output-dir {folder.name}",
        "```",
        "",
        "## Notes",
        "- Some notebooks download models/weights on first run (Transformers/YOLO/Diffusers).",
        "- For repeatable runs, pin dependencies and set seeds.",
    ]
    return "\n".join(lines).replace("\r\n", "\n")


def main() -> None:
    # Top-level content folders (skip core/docs/.github)
    for child in sorted(ROOT.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        if child.name in {"core", "docs", ".github"}:
            continue
        if not any(child.glob("*.ipynb")):
            continue
        (child / "README.md").write_text(folder_readme(child), encoding="utf-8")


if __name__ == "__main__":
    main()
