from __future__ import annotations

import json
import re
from pathlib import Path

TRAILING_COMMA_RE = re.compile(r",(\s*[\]}])")


def remove_trailing_commas(text: str) -> str:
    # Iteratively remove trailing commas before ] or }
    prev = None
    cur = text
    for _ in range(50):
        if cur == prev:
            break
        prev = cur
        cur = TRAILING_COMMA_RE.sub(r"\1", cur)
    return cur


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    nb_paths = sorted(root.glob("**/*.ipynb"))
    fixed = 0
    still_bad: list[tuple[str, str]] = []

    for p in nb_paths:
        raw = p.read_text(encoding="utf-8", errors="strict")
        try:
            json.loads(raw)
            continue
        except Exception as e:
            # try repair
            repaired = remove_trailing_commas(raw)
            try:
                json.loads(repaired)
            except Exception as e2:
                still_bad.append((str(p.relative_to(root)), str(e2)))
                continue

            p.write_text(repaired, encoding="utf-8")
            fixed += 1

    print(f"notebooks total: {len(nb_paths)}")
    print(f"fixed invalid json: {fixed}")
    if still_bad:
        print(f"still invalid: {len(still_bad)}")
        for name, err in still_bad[:50]:
            print(f"- {name}: {err}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
