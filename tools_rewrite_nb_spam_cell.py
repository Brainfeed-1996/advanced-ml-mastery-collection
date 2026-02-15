from __future__ import annotations

from pathlib import Path

import nbformat


def main() -> int:
    root = Path(__file__).resolve().parent
    nb_path = root / "08-Anomaly-Detection" / "Naive-Bayes-Spam-Filter.ipynb"
    nb = nbformat.read(nb_path, as_version=4)

    new_src = """import pandas as pd
import numpy as np

# Simple offline dataset + augmentation (deterministic)
data = [
    ('ham', 'Hey, are we still on for lunch tomorrow?'),
    ('ham', 'Please review the attached report before 5pm.'),
    ('ham', 'Can you call me when you have a minute?'),
    ('ham', 'Happy birthday! Hope you have a great day.'),
    ('ham', 'Your appointment is confirmed for Monday at 10:00.'),
    ('spam', 'WIN a brand new phone! Click http://bit.ly/free now!'),
    ('spam', 'Congratulations! You have been selected for a $1000 gift card. Reply YES'),
    ('spam', 'URGENT: Your account is suspended. Verify details at www.secure-login.example'),
    ('spam', 'Lowest rate loans available. Apply now and get approved today!'),
    ('spam', 'You have 1 new voicemail. Listen here: http://tinyurl.com/vm'),
]

ham_templates = [
    'Hi {name}, can you send the files?',
    "Let's meet at {time} at {place}.",
    'Thanks for your help on {topic}.',
]
spam_templates = [
    'Claim your prize of {amount}! Visit {url}',
    'Limited offer: {pct}% OFF. Buy now: {url}',
    'Act now to receive {amount} cash advance. Apply at {url}',
]

names = ['Alex','Sam','Jordan','Taylor','Morgan','Chris']
times = ['9am','noon','6pm','tomorrow','next week']
places = ['cafe','office','station','downtown']
topics = ['the presentation','the invoice','the project','the meeting']
amounts = ['$500','$1000','$2000']
urls = ['http://bit.ly/deal','http://tinyurl.com/claim','http://example.com/win']
pcts = [30,40,50,70]

rng = np.random.default_rng(SEED)
for _ in range(300):
    t = rng.choice(ham_templates)
    msg = t.format(name=rng.choice(names), time=rng.choice(times), place=rng.choice(places), topic=rng.choice(topics))
    data.append(('ham', msg))
for _ in range(300):
    t = rng.choice(spam_templates)
    msg = t.format(amount=rng.choice(amounts), url=rng.choice(urls), pct=rng.choice(pcts))
    data.append(('spam', msg))

df = pd.DataFrame(data, columns=['label','text'])
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
df['y'] = (df['label'] == 'spam').astype(int)

display(df.head())
print(df['label'].value_counts())
"""

    changed = False
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source") or ""
        src_s = "".join(src) if isinstance(src, list) else str(src)
        if "ham_templates" in src_s and "spam_templates" in src_s:
            cell["source"] = new_src
            changed = True
            break

    if not changed:
        raise SystemExit("target cell not found")

    nbformat.write(nb, nb_path)
    print("patched cell")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
