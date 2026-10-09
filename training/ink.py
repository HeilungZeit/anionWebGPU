"""Плотность штриха: насколько модель светлит тёмные линии и буквы.

    python ink.py base.onnx model.pt [model2.pt ...]
    python ink.py --val base.onnx model.pt [...]

Эталона у кадров Kodik нет, поэтому сравнение идёт с базовой моделью
(AnimeJaNai V2 SUC): штрих — пиксели её выхода, которые темнее своего
окружения (гаусс σ = 4 px) больше чем на INK уровней. По ним для каждой модели
печатается средняя разница яркости с базовой (плюс — светлее) и доля
потерянного контраста штриха: (Δ яркости) / (глубина штриха у базовой).
Кадры — `data/kodik/*/*.png`, все.

С `--val` — сверка с эталоном на отложенных кадрах `data/val/`: штрих берётся
у эталона, печатается то же для базовой модели, перечисленных и учителя
(базовая на чистом входе — цель обучения с `--teacher 1`).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from train import Compact, clean_input, load_onnx, to_tensor

DATA = Path(__file__).parent / "data"
INK = 20


def load(path: Path) -> Compact:
    model = Compact()
    if path.suffix == ".pt":
        model.load_state_dict(torch.load(path, map_location="cpu"))
    else:
        load_onnx(model, path)
    return model.eval()


def luma(model: Compact, bgr: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        out = model.to(device)(to_tensor(bgr, device)[None]).clamp(0, 1)[0]
    r, g, b = (out * 255).cpu().numpy()
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def to_luma(t: torch.Tensor) -> np.ndarray:
    r, g, b = (t.clamp(0, 1) * 255).cpu().numpy()
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def validate(base_path: Path, paths: list[Path], device: torch.device) -> None:
    models = {"база": load(base_path).to(device)} | {f"{p.parent.name}/{p.stem}": load(p).to(device) for p in paths}
    rows: dict[str, list[list[float]]] = {name: [] for name in [*models, "учитель"]}
    for lq_path in sorted((DATA / "val" / "lq").glob("*.png")):
        lq = to_tensor(cv2.imread(str(lq_path)), device)[None]
        gt = to_tensor(cv2.imread(str(DATA / "val" / "gt" / lq_path.name)), device)[None]
        ref = to_luma(gt[0])
        depth = cv2.GaussianBlur(ref, (0, 0), 4) - ref
        mask = depth > INK
        mask[:8], mask[-8:], mask[:, :8], mask[:, -8:] = False, False, False, False
        if mask.sum() < 100:
            continue
        with torch.no_grad():
            outs = {name: model(lq)[0] for name, model in models.items()}
            outs["учитель"] = models["база"](clean_input(gt))[0]
        for name, out in outs.items():
            delta = to_luma(out) - ref
            rows[name].append([delta[mask].mean(), delta[mask].mean() / depth[mask].mean()])
    for name, values in rows.items():
        d, c = np.mean(values, axis=0)
        print(f"{name:40s} штрих к эталону {d:+5.2f} ({c:+.1%})")


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    args = sys.argv[1:]
    if args[0] == "--val":
        base_path, *paths = map(Path, args[1:])
        validate(base_path, paths, device)
        return
    base_path, *paths = map(Path, args)
    base = load(base_path)
    models = [load(p) for p in paths]

    frames = sorted((DATA / "kodik").glob("*/*.png"))
    stats: dict[str, list[list[float]]] = {}
    for path in frames:
        bgr = cv2.imread(str(path))
        ref = luma(base, bgr, device)
        depth = cv2.GaussianBlur(ref, (0, 0), 4) - ref
        mask = depth > INK
        if mask.sum() < 100:
            continue
        for p, model in zip(paths, models):
            delta = luma(model, bgr, device) - ref
            stats.setdefault(path.parent.name, []).append(
                [delta[mask].mean(), delta[mask].mean() / depth[mask].mean()]
            )
    for series, rows in stats.items():
        rows = np.array(rows).reshape(-1, len(models), 2).mean(axis=0)
        cells = "  ".join(f"{p.parent.name}/{p.stem}: {d:+5.2f} ({c:+.1%})" for p, (d, c) in zip(paths, rows))
        print(f"{series:12s} {cells}")
    every = np.array([r for rows in stats.values() for r in rows]).reshape(-1, len(models), 2).mean(axis=0)
    print(f"{'все':12s} " + "  ".join(f"{p.parent.name}/{p.stem}: {d:+5.2f} ({c:+.1%})" for p, (d, c) in zip(paths, every)))


if __name__ == "__main__":
    main()
