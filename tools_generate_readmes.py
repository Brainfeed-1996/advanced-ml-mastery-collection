"""Generate/refresh README.md files across the repository (industrial-grade).

Goals
- Root README stays high-level.
- Each top-level module folder gets a README with:
  - purpose & learning objectives
  - prerequisites & heavy deps guidance
  - how to run (Jupyter + nbconvert)
  - what you should see as outputs
  - troubleshooting (downloads/cache/CPU vs GPU)

Run:
  python tools_generate_readmes.py
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
STAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

FOLDER_INTROS = {
    "01-Regression": "Foundations: supervised learning for continuous targets (regression).",
    "01-Regression-Analysis": "Regression + deeper evaluation/diagnostics (industrial baselines).",
    "02-Classification": "Core classification algorithms + evaluation patterns.",
    "02-Classification-Challenges": "Challenge-style notebooks (harder variants, tuning, edge cases).",
    "03-Clustering": "Unsupervised segmentation & clustering evaluation.",
    "03-Clustering-Techniques": "Additional techniques / deep-learning flavored clustering projects.",
    "04-NLP": "Natural language processing workflows (offline baselines + optional Transformers).",
    "04-Natural-Language-Processing": "More advanced NLP/DL workflows (training loops, RL intro).",
    "05-Computer-Vision": "Computer vision workflows (CNN, transfer learning, detection).",
    "06-Reinforcement-Learning": "Reinforcement learning (policy gradients, Q-learning, DQN patterns).",
    "07-Time-Series": "Time-series forecasting baselines and deep models.",
    "07-Time-Series-Forecasting": "Production forecasting topics: monitoring, interpretability, quantization.",
    "08-Anomaly-Detection": "Anomaly detection in cyber/fraud/streaming contexts.",
    "09-Generative-AI": "Generative AI: GAN/VAE, object detection, stable diffusion prompt engineering.",
    "10-MLOps": "MLOps patterns: monitoring, quantization, edge constraints.",
    "10-MLOps-Production": "Production-grade MLOps: federated learning, transformers internals, edge ML.",
}

# Minimal curated descriptions by filename keywords
DESC = {
    "K-Means": "K-means clustering for segmentation + silhouette-based K selection + profiling.",
    "SVM": "Support Vector Machine classification with tuning + diagnostics.",
    "XGBoost": "Gradient boosting for churn-style classification (XGBoost if available).",
    "BERT": "Sentiment pipeline with offline baseline + Transformers inference.",
    "GPT": "Fine-tuning mechanics via a tiny causal LM + sampling.",
    "Named-Entity": "Named Entity Recognition using spaCy + visualization.",
    "CNN": "CNN training loop + evaluation + confusion matrix.",
    "LSTM": "Sequence forecasting with LSTM + windowing + plots.",
    "YOLO": "Object detection with YOLOv8 (downloads weights on first run).",
    "Stable-Diffusion": "Stable Diffusion prompt engineering (CPU-safe default + optional full model).",
    "ResNet": "Transfer learning with pretrained ResNet (torchvision weights).",
    "Variational-Autoencoders": "VAE training + latent sampling + reconstructions.",
    "Autoencoders": "Autoencoder workflows (denoising / reconstruction).",
    "Prometheus": "Model monitoring patterns with Prometheus-like metrics.",
    "Quantization": "Post-training quantization patterns (TensorRT discussed; CPU fallback in notebook).",
    "Federated": "Federated learning simulation + privacy considerations.",
    "Optuna": "Hyperparameter optimization with Optuna (study + best params).",
    "CartPole": "Reinforcement learning agent training (gymnasium).",
    "Atari": "Deep Q-Network (DQN) training pattern (resource heavy).",
}


def describe(name: str) -> str:
    for k, v in DESC.items():
        if k in name:
            return v
    return "Notebook project (see notebook for details)."


def folder_readme(folder: Path) -> str:
    nbs = sorted(folder.glob("*.ipynb"))

    intro = FOLDER_INTROS.get(folder.name, "Notebook collection.")

    lines: list[str] = []
    lines += [f"# {folder.name}", "", intro, "", f"Last refresh: **{STAMP}**", ""]

    lines += [
        "## Prerequisites",
        "",
        "- Python 3.10+ recommended",
        "- Create a venv (see `docs/INSTALLATION.md`)",
        "- Some notebooks require heavy dependencies (see `docs/OPTIONAL_HEAVY_DEPS.md`)",
        "",
        "If a notebook downloads weights/models on first run, expect longer execution times.",
        "",
    ]

    lines += ["## Contents", ""]
    if not nbs:
        lines.append("(no notebooks found)")
    else:
        for nb in nbs:
            lines.append(f"- `{nb.name}` — {describe(nb.stem)}")

    lines += [
        "",
        "## How to run",
        "",
        "### Interactive",
        "```bash",
        f"jupyter notebook {folder.name}/<notebook>.ipynb",
        "```",
        "",
        "### Headless (embed outputs into the notebook file)",
        "```bash",
        "python -m jupyter nbconvert --to notebook --execute \\",
        f"  {folder.name}/<notebook>.ipynb --output <notebook>.ipynb --output-dir {folder.name}",
        "```",
        "",
        "## Expected outputs",
        "- Printed metrics (accuracy/ROC-AUC/MSE/etc.)",
        "- At least one plot or table for interpretation",
        "- For heavy notebooks: model download logs (first run) + sample inference outputs",
        "",
        "## Troubleshooting",
        "- **SIGKILL / OOM** (especially diffusion on CPU): reduce steps, reduce image size, or run on a GPU machine.",
        "- **Corrupted model cache** (Transformers/Diffusers): clear HuggingFace cache (`~/.cache/huggingface`).",
        "- **Slow runs**: prefer tiny models for validation; then enable full runs intentionally.",
    ]

    return "\n".join(lines).replace("\r\n", "\n")


def main() -> None:
    # Top-level content folders (skip core/docs/.github)
    for child in sorted(ROOT.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        if child.name in {"core", "docs", ".github", "__pycache__"}:
            continue
        if not any(child.glob("*.ipynb")):
            continue
        (child / "README.md").write_text(folder_readme(child), encoding="utf-8")


if __name__ == "__main__":
    main()
