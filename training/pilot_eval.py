"""Сводка моделей «Деталей» (Э19): отложенные кадры с эталоном и кадры Kodik.

    python pilot_eval.py имя=путь [имя=путь ...]

Путь — ONNX SUC, чекпойнт `.pt` (архитектура из `arch.txt` рядом) или
`mix<α>:<a>:<b>` — поэлементная смесь весов α·a + (1 − α)·b (как Kodik mix75
в Э17). Первая модель — база для «штрих светлее базы».

`data/val/` (эталон есть): PSNR к эталону и к учителю (JaNai V2 на чистом
входе), плотность штриха к эталону (как `ink.py --val`).

`data/kodik/*/*.png` (эталона нет), яркость выхода ×2; маски берутся из
входа, растянутого бикубикой, — одни и те же пиксели у всех моделей:
- шум    — СКО ВЧ (выход − гаусс σ1.5) на плоских участках 16×16 (размах
           сглаженного входа ≤ 6 уровней): меньше — чище заливка;
- детали — то же на участках со слабой текстурой (СКО входа 2–10 уровней,
           без контуров): больше — фон меньше похож на акварель (Э17);
- блоки  — сетка 8×8 энкода после ×2: средний слабый перепад (< 8 уровней)
           соседей на границе x ≡ 15 (mod 16) против x ≡ 3, 7, 11 — той же
           чётности и той же фазы 4×4, чтобы рисунок 2×2 и сетка pixelshuffle
           сюда не попадали; 1.0 — блоков нет;
- 2×2    — слабые перепады между парами пикселей выхода против перепадов
           внутри пары (skip ближайшим соседом ×2): 1.0 — пар не видно;
- штрих  — средняя Δ яркости на штрихе базы (темнее окружения на 20+):
           плюс — модель светлит линии и буквы (Э13).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from train import DATA, Compact, clean_input, load_any, load_onnx, psnr, to_tensor

TEACHER = DATA / "init" / "2x_AnimeJaNai_V2_SuperUltraCompact_100k.onnx"
INK = 20


def load(spec: str) -> torch.nn.Module:
    if spec.startswith("mix"):
        alpha, a, b = spec.removeprefix("mix").split(":")
        model_a, model_b = load_any(Path(a)), load_any(Path(b))
        state = {k: float(alpha) * v + (1 - float(alpha)) * model_b.state_dict()[k] for k, v in model_a.state_dict().items()}
        model_a.load_state_dict(state)
        return model_a
    return load_any(Path(spec))


def luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def run(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    """Выход в уровнях 8 бит, RGB (H, W, 3)."""
    with torch.no_grad():
        out = model(x).clamp(0, 1)[0]
    return out.permute(1, 2, 0).cpu().numpy() * 255


def phase_steps(y: np.ndarray) -> np.ndarray:
    """Суммы и счёт слабых (< 8) перепадов соседей по фазе границы mod 16, по
    обеим осям: [фаза, (сумма, число)]."""
    out = np.zeros((16, 2))
    for d in (np.abs(np.diff(y, axis=1)), np.abs(np.diff(y, axis=0)).T):
        weak = d < 8
        for k in range(16):
            out[k] += [d[:, k::16][weak[:, k::16]].sum(), weak[:, k::16].sum()]
    return out


def grid_scores(steps: np.ndarray) -> tuple[float, float]:
    """(блоки 8×8 энкода, рисунок 2×2) из `phase_steps`, сложенных по кадрам."""
    v = steps[:, 0] / steps[:, 1]
    return float(v[15] / v[[3, 7, 11]].mean()), float(v[1::2].mean() / v[0::2].mean())


def tile_masks(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Плоские и слабо текстурные участки 16×16 выхода — по входу ×2."""
    h, w = bgr.shape[:2]
    up = luma(cv2.resize(bgr[..., ::-1].astype(np.float32), (w * 2, h * 2), interpolation=cv2.INTER_CUBIC))
    smooth = cv2.GaussianBlur(up, (0, 0), 2)
    edges = np.abs(cv2.Sobel(smooth, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
    th, tw = h * 2 // 16, w * 2 // 16

    def tiles(a: np.ndarray) -> np.ndarray:
        return a[: th * 16, : tw * 16].reshape(th, 16, tw, 16).swapaxes(1, 2)

    spread = tiles(smooth).max(axis=(2, 3)) - tiles(smooth).min(axis=(2, 3))
    std = tiles(up).std(axis=(2, 3))
    edge = tiles(edges).max(axis=(2, 3))
    flat = spread <= 6
    texture = (std >= 2) & (std <= 10) & (edge < 60)
    flat[0], flat[-1], flat[:, 0], flat[:, -1] = False, False, False, False
    texture[0], texture[-1], texture[:, 0], texture[:, -1] = False, False, False, False
    return flat, texture


def hf_std(y: np.ndarray, mask: np.ndarray) -> list[float]:
    hf = y - cv2.GaussianBlur(y, (0, 0), 1.5)
    th, tw = mask.shape
    tiles = hf[: th * 16, : tw * 16].reshape(th, 16, tw, 16).swapaxes(1, 2)
    return list(tiles[mask].std(axis=(1, 2)))


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    named = [arg.split("=", 1) for arg in sys.argv[1:]]
    models = {name: load(spec).to(device).eval() for name, spec in named}
    teacher = Compact()
    load_onnx(teacher, TEACHER)
    teacher = teacher.to(device).eval()

    val: dict[str, list[list[float]]] = {name: [] for name in models}
    for lq_path in sorted((DATA / "val" / "lq").glob("*.png")):
        lq = to_tensor(cv2.imread(str(lq_path)), device)[None]
        gt = to_tensor(cv2.imread(str(DATA / "val" / "gt" / lq_path.name)), device)[None]
        with torch.no_grad():
            target = teacher(clean_input(gt)).clamp(0, 1)
        ref = luma(gt[0].permute(1, 2, 0).cpu().numpy() * 255)
        depth = cv2.GaussianBlur(ref, (0, 0), 4) - ref
        ink = depth > INK
        ink[:8], ink[-8:], ink[:, :8], ink[:, -8:] = False, False, False, False
        for name, model in models.items():
            with torch.no_grad():
                out = model(lq).clamp(0, 1)
            delta = luma(out[0].permute(1, 2, 0).cpu().numpy() * 255) - ref
            ink_c = delta[ink].mean() / depth[ink].mean() if ink.sum() >= 100 else np.nan
            val[name].append([psnr(out, gt), psnr(out, target), ink_c])

    kodik: dict[str, dict[str, list[float]]] = {name: {"noise": [], "detail": [], "ink": []} for name in models}
    steps = {name: np.zeros((16, 2)) for name in models}
    base_name = next(iter(models))
    for path in sorted((DATA / "kodik").glob("*/*.png")):
        bgr = cv2.imread(str(path))
        if bgr.shape[0] % 2 or bgr.shape[1] % 2:
            continue
        flat, texture = tile_masks(bgr)
        x = to_tensor(bgr, device)[None]
        outs = {name: luma(run(model, x)) for name, model in models.items()}
        base = outs[base_name]
        depth = cv2.GaussianBlur(base, (0, 0), 4) - base
        ink = depth > INK
        for name, y in outs.items():
            row = kodik[name]
            row["noise"] += hf_std(y, flat)
            row["detail"] += hf_std(y, texture)
            steps[name] += phase_steps(y)
            if ink.sum() >= 100:
                row["ink"].append(float((y - base)[ink].mean()))

    print(f"{'модель':14s} {'к эталону':>10s} {'к учителю':>10s} {'штрих/эт.':>10s}"
          f" {'шум':>6s} {'детали':>7s} {'блоки':>6s} {'2×2':>5s} {'штрих/база':>11s}")
    for name in models:
        g, t, c = np.nanmean(val[name], axis=0)
        k = kodik[name]
        blocks, pairs = grid_scores(steps[name])
        print(f"{name:14s} {g:9.2f}  {t:9.2f}  {c:+9.1%} {np.mean(k['noise']):6.2f} {np.mean(k['detail']):7.2f}"
              f" {blocks:6.3f} {pairs:5.2f} {np.mean(k['ink']):+10.2f}")


if __name__ == "__main__":
    main()
