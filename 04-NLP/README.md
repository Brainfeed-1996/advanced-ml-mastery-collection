# 04-NLP


Auto-generated: 2026-02-16 01:25:44

## Contents

- `BERT-Sentiment-Analysis.ipynb` — Transformer-based NLP workflow (offline baseline + optional transformers).
- `GPT-Fine-Tuning-Basics.ipynb` — Language model / fine-tuning mechanics (offline-friendly).
- `Named-Entity-Recognition-Spacy.ipynb` — NER pipeline and evaluation.
- `TFIDF-ErrorAnalysis-Classifier.ipynb` — Notebook project (see notebook for details).

## How to run

```bash
jupyter notebook 04-NLP/<notebook>.ipynb
```

Or execute headlessly (exports outputs into the same file):

```bash
python -m jupyter nbconvert --to notebook --execute \
  04-NLP/<notebook>.ipynb --output <notebook>.ipynb --output-dir 04-NLP
```

## Notes
- Some notebooks download models/weights on first run (Transformers/YOLO/Diffusers).
- For repeatable runs, pin dependencies and set seeds.