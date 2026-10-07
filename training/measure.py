"""Метрики кадров для подбора порчи: Kodik против испорченных эталонов.

    python measure.py <папка с png> [<папка> ...]

Для каждой папки печатает медианы по кадрам:
- res@s   — PSNR кадра против его же копии, ужатой в s раз и растянутой
            обратно (Lanczos). Высокий PSNR при малом s значит, что деталей
            мельче этого масштаба в кадре нет: он растянут из меньшего.
- hf      — доля энергии лапласиана в яркости (резкость контуров).
- block   — во сколько раз перепад яркости на границах блоков 8×8 больше,
            чем внутри них; 1.0 — блочности нет.
- flat    — доля плоских участков 16×16 (заливка и градиенты), на которых
            виден бандинг.
- band    — среднее число ступенек яркости на плоских участках: при
            бандинге гладкий градиент рвётся на полосы.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

SCALES = (0.5, 0.67, 0.75, 0.85)


def luma(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def resample_psnr(y: np.ndarray, s: float) -> float:
    h, w = y.shape
    small = cv2.resize(y, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LANCZOS4)
    # Края кадра Lanczos портит сам по себе — не считаем их.
    return psnr(y[8:-8, 8:-8], back[8:-8, 8:-8])


def hf_ratio(y: np.ndarray) -> float:
    lap = cv2.Laplacian(y, cv2.CV_32F)
    return float(np.mean(np.abs(lap)) / (np.mean(np.abs(y - y.mean())) + 1e-6))


def blockiness(y: np.ndarray) -> float:
    dx = np.abs(np.diff(y, axis=1))
    dy = np.abs(np.diff(y, axis=0))
    on = np.concatenate([dx[:, 7::8].ravel(), dy[7::8, :].ravel()])
    mask_x = np.ones(dx.shape[1], bool)
    mask_x[7::8] = False
    mask_y = np.ones(dy.shape[0], bool)
    mask_y[7::8] = False
    off = np.concatenate([dx[:, mask_x].ravel(), dy[mask_y, :].ravel()])
    # Сравниваем только слабые перепады: блочность видна на гладком, а
    # контуры рисовки сравнение бы утопили.
    on, off = on[on < 8], off[off < 8]
    return float(on.mean() / (off.mean() + 1e-6))


def banding(y: np.ndarray) -> tuple[float, float]:
    h, w = y.shape
    tiles = y[: h // 16 * 16, : w // 16 * 16].reshape(h // 16, 16, w // 16, 16).swapaxes(1, 2)
    tiles = tiles.reshape(-1, 16, 16)
    spread = tiles.max(axis=(1, 2)) - tiles.min(axis=(1, 2))
    flat = tiles[(spread > 0) & (spread <= 6)]
    if len(flat) == 0:
        return 0.0, 0.0
    # Ступенька — соседние пиксели, разошедшиеся ровно на уровень; при гладком
    # градиенте после сжатия таких мало, при бандинге — длинные границы полос.
    steps = (np.abs(np.diff(np.round(flat), axis=2)) == 1).mean(axis=(1, 2))
    return len(flat) / len(tiles), float(steps.mean())


def measure(folder: Path) -> dict[str, float]:
    rows = []
    for path in sorted(folder.glob("*.png")):
        y = luma(path)
        row = {f"res@{s}": resample_psnr(y, s) for s in SCALES}
        row["hf"] = hf_ratio(y)
        row["block"] = blockiness(y)
        row["flat"], row["band"] = banding(y)
        rows.append(row)
    return {k: float(np.median([r[k] for r in rows])) for k in rows[0]}


def main() -> None:
    folders = [Path(p) for p in sys.argv[1:]]
    width = max(len(f.name) for f in folders)
    header = None
    for folder in folders:
        stats = measure(folder)
        if header is None:
            header = list(stats)
            print(" " * width, *(f"{k:>9}" for k in header))
        print(f"{folder.name:<{width}}", *(f"{stats[k]:9.3f}" for k in header))


if __name__ == "__main__":
    main()
