from __future__ import annotations

from pathlib import Path

import nbformat


def replace_cell_source(nb_path: Path, contains: str, new_source: str) -> bool:
    nb = nbformat.read(nb_path, as_version=4)
    changed = False
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source") or ""
        if contains in src:
            cell["source"] = new_source
            changed = True
            break
    if changed:
        nbformat.write(nb, nb_path)
    return changed


def main() -> int:
    root = Path(__file__).resolve().parent

    # Patch: make dataset offline-first BUT keep same column names as California housing notebooks
    offline_housing_cell = """import numpy as np\nimport pandas as pd\nfrom sklearn.datasets import make_regression\n\n# Offline-first synthetic dataset with the same schema as California Housing\n# so downstream feature-engineering cells keep working.\n\nFEATURES = [\n    'MedInc',\n    'HouseAge',\n    'AveRooms',\n    'AveBedrms',\n    'Population',\n    'AveOccup',\n    'Latitude',\n    'Longitude',\n]\n\nX, y = make_regression(\n    n_samples=6000,\n    n_features=len(FEATURES),\n    n_informative=6,\n    noise=18.0,\n    random_state=SEED,\n)\n\ndf = pd.DataFrame(X, columns=FEATURES)\n# Keep positive-only columns for ratios to behave (rooms/bedrooms/occupancy)\ndf['HouseAge'] = np.abs(df['HouseAge'])\ndf['AveRooms'] = np.abs(df['AveRooms']) + 1.0\ndf['AveBedrms'] = np.abs(df['AveBedrms']) + 0.5\ndf['Population'] = (np.abs(df['Population']) * 1000).astype(float) + 100.0\ndf['AveOccup'] = np.abs(df['AveOccup']) + 0.5\n\n# Lat/Long in plausible ranges\ndf['Latitude'] = 32 + (np.abs(df['Latitude']) % 10)\ndf['Longitude'] = -124 + (np.abs(df['Longitude']) % 10)\n\ndf['target'] = y\n\ndisplay(df.head())\nprint(df.shape)\ndf.describe().T\n"""

    patches = {
        "01-Regression/Linear-Regression-Real-Estate.ipynb": (
            "Offline-first dataset",
            offline_housing_cell,
        ),
        "01-Regression-Analysis/Linear-Regression-Real-Estate.ipynb": (
            "fetch_california_housing",
            offline_housing_cell,
        ),
    }

    changed = 0
    for rel, (needle, new_src) in patches.items():
        p = root / rel
        if not p.exists():
            print(f"SKIP missing: {rel}")
            continue
        if replace_cell_source(p, needle, new_src):
            print(f"PATCHED: {rel}")
            changed += 1
        else:
            print(f"NO MATCH: {rel}")

    print(f"patched notebooks: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
