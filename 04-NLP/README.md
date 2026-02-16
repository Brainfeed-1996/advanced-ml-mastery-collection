# 04-NLP

Natural language processing workflows (offline baselines + optional Transformers).

Last refresh: **2026-02-16 16:05:59**

## Prerequisites

- Python 3.10+ recommended
- Create a venv (see `docs/INSTALLATION.md`)
- Some notebooks require heavy dependencies (see `docs/OPTIONAL_HEAVY_DEPS.md`)

If a notebook downloads weights/models on first run, expect longer execution times.

## Contents

### Notebooks (.ipynb)
- `BERT-Sentiment-Analysis.ipynb` — Sentiment pipeline with offline baseline + Transformers inference.
- `GPT-Fine-Tuning-Basics.ipynb` — Fine-tuning mechanics via a tiny causal LM + sampling.
- `Named-Entity-Recognition-Spacy.ipynb` — Named Entity Recognition using spaCy + visualization.
- `TFIDF-ErrorAnalysis-Classifier.ipynb` — Notebook project (see notebook for details).

### Python Tools (.py)
- `BERT-Sentiment-Analysis.py` — Automation tool for notebook management.
- `GPT-Fine-Tuning-Basics.py` — Automation tool for notebook management.
- `Named-Entity-Recognition-Spacy.py` — Automation tool for notebook management.

## How to run

### Interactive (Jupyter)
```bash
jupyter notebook 04-NLP/<notebook>.ipynb
```

### Headless (embed outputs into the notebook file)
```bash
python -m jupyter nbconvert --to notebook --execute \
  04-NLP/<notebook>.ipynb --output <notebook>.ipynb --output-dir 04-NLP
```

### Running Python tools
```bash
cd 04-NLP
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