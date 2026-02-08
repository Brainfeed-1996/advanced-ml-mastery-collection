from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    nb_path = (root / args.notebook).resolve()
    nb = nbformat.read(nb_path, as_version=4)

    repl = """from sklearn.datasets import make_blobs

class IndustrialDataGenerator:
    def __init__(self, n_samples: int = 2000, n_features: int = 10, centers: int = 4, cluster_std: float = 1.0):
        self.n_samples = n_samples
        self.n_features = n_features
        self.centers = centers
        self.cluster_std = cluster_std

    def generate(self):
        logger.info("Generating clustering dataset (offline-first).")
        X, y = make_blobs(
            n_samples=self.n_samples,
            n_features=self.n_features,
            centers=self.centers,
            cluster_std=self.cluster_std,
            random_state=SEED,
        )
        df = pd.DataFrame(X, columns=[f"Dim_{i}" for i in range(self.n_features)])
        return df, y


data_gen = IndustrialDataGenerator()
X_df, y = data_gen.generate()
X_df.head()
"""

    changed = False
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source") or ""
        src_s = "".join(src) if isinstance(src, list) else str(src)
        if "IndustrialDataGenerator" in src_s and "make_blobs" in src_s:
            cell["source"] = repl
            changed = True
            break

    if not changed:
        raise SystemExit("no matching generator cell")

    nbformat.write(nb, nb_path)
    print("rewrote generator cell")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
