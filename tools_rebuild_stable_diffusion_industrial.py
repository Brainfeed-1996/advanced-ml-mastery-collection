"""Rewrite + execute Stable Diffusion notebook in an industrial, CPU-safe way.

Default uses tiny pipeline to avoid SIGKILL on CPU.
Optional full model run: set SD_FULL=1.

Run:
  python tools_rebuild_stable_diffusion_industrial.py
"""

from __future__ import annotations

from pathlib import Path
import nbformat as nbf

import tools_rebuild_no_outputs_industrial as base


def main() -> None:
    cells = [
        base.md(
            "Stable Diffusion — Prompt Engineering (industrial)",
            "Default: tiny SD pipeline for CPU safety. Optional: SD v1-5 full download/run (may SIGKILL on CPU).\n\n"
            "Usage:\n"
            "- Default (CPU-safe): run as-is\n"
            "- Full model attempt: set env var `SD_FULL=1` before executing\n",
        )
    ]

    cells.append(
        nbf.v4.new_code_cell(
            """import os
import torch
from diffusers import StableDiffusionPipeline

FULL = os.getenv('SD_FULL', '0') == '1'
model_id = 'runwayml/stable-diffusion-v1-5' if FULL else 'hf-internal-testing/tiny-stable-diffusion-pipe'

print('FULL=', FULL)
print('model_id=', model_id)

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to('cpu')

prompt = 'a cyberpunk city skyline at night, neon lights, cinematic'
steps = 15 if FULL else 5
image = pipe(prompt, num_inference_steps=steps).images[0]
image.size
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.imshow(image)
plt.axis('off')
plt.show()
"""
        )
    )

    cells.append(nbf.v4.new_code_cell(f"print('DONE {base.STAMP}')"))

    path = Path("09-Generative-AI/Stable-Diffusion-Prompt-Eng.ipynb")
    base.write_nb(path, cells)

    # Execute with allow_errors behavior handled by CLI caller; here we just run standard execute.
    base.exec_nb(path, timeout=7200)


if __name__ == "__main__":
    main()
