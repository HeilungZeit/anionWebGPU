"""Выходы учителя на чистом входе — заранее, для `train.py --teacher-cache`.

    python teacher_cache.py data/init/2x_AnimeJaNai_V2_Compact_36k.onnx train128 teacher-compact.npy

Цель обучения с `--teacher 1` — учитель на эталоне, уменьшенном 2×2
(`clean_input`). AnimeJaNai V2 Compact (64nf/16nc) в 6.5 раза тяжелее SUC —
считать его на каждом шаге значит учить вдвое дольше. Выход хранится как
uint8 BGR той же формы, что `gt.npy` (округление до уровня — много меньше
ошибки ученика).
"""

import sys
from pathlib import Path

import numpy as np
import torch

from train import DATA, clean_input, onnx_compact, to_tensor

BATCH = 64


def main() -> None:
    model_path, data, name = sys.argv[1:]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    teacher = onnx_compact(Path(model_path)).to(device).eval()
    gt_all = np.load(DATA / data / "gt.npy", mmap_mode="r")
    out = np.lib.format.open_memmap(DATA / data / name, "w+", np.uint8, gt_all.shape)
    with torch.no_grad():
        for start in range(0, len(gt_all), BATCH):
            gt = to_tensor(gt_all[start : start + BATCH], device)
            rgb = teacher(clean_input(gt)).clamp(0, 1).permute(0, 2, 3, 1)
            out[start : start + len(gt)] = (rgb * 255 + 0.5).byte().cpu().numpy()[..., ::-1]
            if start // BATCH % 50 == 0:
                print(f"{start}/{len(gt_all)}", flush=True)
    out.flush()
    print(f"готово: {DATA / data / name}")


if __name__ == "__main__":
    main()
