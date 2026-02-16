"""Rebuild + execute the remaining notebooks reported as NO OUTPUTS.

This uses the industrial builders from tools_rebuild_no_outputs_industrial.py for supported filenames.
For unsupported filenames, it will only execute (no overwrite).

Run:
  python tools_run_remaining_no_outputs.py
"""

from __future__ import annotations

import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent

mod = importlib.import_module("tools_rebuild_no_outputs_industrial")
write_nb = getattr(mod, "write_nb")
exec_nb = getattr(mod, "exec_nb")
builder_for = getattr(mod, "builder_for")

TARGETS = [
    r"04-Natural-Language-Processing/RL-CartPole-Agent.ipynb",
    r"04-NLP/BERT-Sentiment-Analysis.ipynb",
    r"04-NLP/GPT-Fine-Tuning-Basics.ipynb",
    r"04-NLP/Named-Entity-Recognition-Spacy.ipynb",
    r"05-Computer-Vision/Autoencoders-Denoising.ipynb",
    r"05-Computer-Vision/CNN-Medical-Imaging.ipynb",
    r"05-Computer-Vision/GAN-Synthetic-Data-Gen.ipynb",
    r"05-Computer-Vision/Isolation-Forest-Cybersecurity.ipynb",
    r"05-Computer-Vision/ResNet-Transfer-Learning.ipynb",
    r"05-Computer-Vision/YOLOv8-Object-Detection.ipynb",
    r"06-Reinforcement-Learning/RL-CartPole-Agent.ipynb",
    r"09-Generative-AI/Object-Detection-YOLOv8.ipynb",
    r"09-Generative-AI/Stable-Diffusion-Prompt-Eng.ipynb",
]

ok, fail, overwritten = 0, 0, 0

for rel in TARGETS:
    p = ROOT / rel
    fn = builder_for(rel)
    if fn is not None:
        write_nb(p, fn())
        overwritten += 1
    print("EXEC:", rel)
    try:
        exec_nb(p, timeout=3600)
        ok += 1
    except Exception as e:
        fail += 1
        print("FAIL:", rel)
        print(e)

print("\nSUMMARY")
print("ok", ok)
print("fail", fail)
print("overwritten", overwritten)
