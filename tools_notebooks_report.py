from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    nb_paths = sorted(root.glob("**/*.ipynb"))
    sizes = sorted(((p.stat().st_size, p) for p in nb_paths), reverse=True)

    print(f"notebooks: {len(nb_paths)}")
    print("largest:")
    for sz, p in sizes[:15]:
        print(f"- {sz/1024/1024:.2f} MB  {p.relative_to(root)}")

    # quick JSON parse validation
    bad = []
    for _, p in sizes:
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            bad.append((str(p.relative_to(root)), str(e)))

    if bad:
        print("\nINVALID JSON notebooks:")
        for name, err in bad[:30]:
            print(f"- {name}: {err}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
