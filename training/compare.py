"""Сравнение моделей на настоящих кадрах Kodik.

    python compare.py out.png model_a.onnx model_b.pt [...]

Модели — как у `pilot_eval.py`: ONNX SUC или чекпойнт `.pt` любой архитектуры.

Для каждой серии из `data/kodik/` берёт фиксированные кадры и фрагмент
256×144 из середины, прогоняет все модели ×2 и складывает мозаику: строка —
фрагмент, столбцы — вход (растянут ближайшим соседом) и модели по порядку.
Эталона у кадров Kodik нет, так что это сравнение на глаз.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from train import load_any, to_tensor

DATA = Path(__file__).parent / "data"
FRAMES = ("08", "20", "32")
CROP = (256, 144)
# Контекст вокруг фрагмента: край свёрток не должен попасть в мозаику, а поле
# зрения у сетей половинной сетки (Э19) — до ~50 px.
PAD = 32


def run(model: nn.Module, bgr: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        out = model(to_tensor(bgr, torch.device("cpu"))[None]).clamp(0, 1)[0]
    rgb = (out.permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)
    return np.ascontiguousarray(rgb[..., ::-1])


def main() -> None:
    out_path, *model_paths = sys.argv[1:]
    models = [load_any(path) for path in map(Path, model_paths)]

    rows = []
    for series in sorted((DATA / "kodik").iterdir()):
        for name in FRAMES:
            img = cv2.imread(str(series / f"{name}.png"))
            if img is None:  # одиночные кадры вроде `dara1/` — не в мозаику
                continue
            h, w = img.shape[:2]
            cw, ch = CROP
            y, x = (h - ch) // 2, (w - cw) // 2
            patch = img[y - PAD : y + ch + PAD, x - PAD : x + cw + PAD]
            tiles = [cv2.resize(img[y : y + ch, x : x + cw], (cw * 2, ch * 2), interpolation=cv2.INTER_NEAREST)]
            for model in models:
                tiles.append(run(model, patch)[2 * PAD : -2 * PAD, 2 * PAD : -2 * PAD])
            rows.append(np.hstack(tiles))
    cv2.imwrite(out_path, np.vstack(rows))


if __name__ == "__main__":
    main()
