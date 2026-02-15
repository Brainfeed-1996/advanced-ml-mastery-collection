# Installation

## Quick Start

```bash
git clone https://github.com/Brainfeed-1996/advanced-ml-mastery-collection.git
cd advanced-ml-mastery-collection
pip install -r requirements.txt
jupyter notebook
```

## Virtual Environment

### Linux/Mac

```bash
python -m venv ml-env
source ml-env/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv ml-env
.\ml-env\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dependencies (high-level)

- Core: `numpy`, `pandas`, `scikit-learn`
- Deep learning: `tensorflow` and/or `torch`
- NLP: `transformers`, `spacy`, `nltk`
- CV: `opencv-python`, `torchvision`
- Viz: `matplotlib`, `seaborn`, `plotly`
- Time series: `prophet`, `statsmodels`
- MLOps: `optuna`, `mlflow`

Exact pins live in `requirements.txt`.