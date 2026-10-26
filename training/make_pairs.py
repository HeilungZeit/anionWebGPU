"""Пары для обучения ×2: эталон APISR и он же, уменьшенный вдвое и испорченный.

    python make_pairs.py [--patch 96] [--crops 3] [--out train]

`--patch` — сторона входного фрагмента, `--out` — каталог в `data/`. Сетям
с широким полем зрения (Э19: до 50 px) фрагменты 96 малы — им `--patch 128
--crops 2 --out train128` (порча та же: зерно генератора — номер кадра).

Порча идёт через настоящий x264 (~0.1 с на кадр), поэтому пары готовятся
заранее, а не в загрузчике. На каждый кадр — VARIANTS вариантов порчи, из
каждого — CROPS случайных фрагментов. Всё складывается в два массива uint8
на диске (`data/train/lq.npy`, `gt.npy`), обучение читает их через memmap.

Последние VAL кадров (по имени) не участвуют: из них делаются целые кадры
для проверки, `data/val/{lq,gt}/`.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from degrade import random_degrade

DATA = Path(__file__).parent / "data"
SCALE = 2
LQ_PATCH = 96
VARIANTS = 2
CROPS = 3
VAL = 100


def load(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    # Вход должен делиться на SCALE, а после уменьшения — быть чётным для x264.
    step = SCALE * 2
    return img[: img.shape[0] // step * step, : img.shape[1] // step * step]


def train_job(args: tuple[int, Path], patch: int = LQ_PATCH, crops: int = CROPS) -> tuple[np.ndarray, np.ndarray]:
    index, path = args
    rng = np.random.default_rng(index)
    img = load(path)
    lqs, gts = [], []
    for _ in range(VARIANTS):
        lq, gt = random_degrade(img, rng, SCALE)
        h, w = lq.shape[:2]
        for _ in range(crops):
            y = int(rng.integers(0, h - patch + 1))
            x = int(rng.integers(0, w - patch + 1))
            lqs.append(lq[y : y + patch, x : x + patch])
            gts.append(gt[y * SCALE : (y + patch) * SCALE, x * SCALE : (x + patch) * SCALE])
    return np.stack(lqs), np.stack(gts)


def val_job(args: tuple[int, Path]) -> None:
    index, path = args
    lq, gt = random_degrade(load(path), np.random.default_rng(10_000 + index), SCALE)
    cv2.imwrite(str(DATA / "val" / "lq" / path.name), lq)
    cv2.imwrite(str(DATA / "val" / "gt" / path.name), gt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", type=int, default=LQ_PATCH)
    parser.add_argument("--crops", type=int, default=CROPS)
    parser.add_argument("--out", default="train")
    args = parser.parse_args()
    patch, crops, out = args.patch, args.crops, DATA / args.out

    sources = sorted((DATA / "apisr" / "APISR_Dataset").glob("*.png"))
    train, val = sources[:-VAL], sources[-VAL:]

    for sub in ("val/lq", "val/gt"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    count = len(train) * VARIANTS * crops
    gt_patch = patch * SCALE
    lq_out = np.lib.format.open_memmap(out / "lq.npy", "w+", np.uint8, (count, patch, patch, 3))
    gt_out = np.lib.format.open_memmap(out / "gt.npy", "w+", np.uint8, (count, gt_patch, gt_patch, 3))

    with ProcessPoolExecutor() as pool:
        list(pool.map(val_job, enumerate(val)))
        cursor = 0
        job = partial(train_job, patch=patch, crops=crops)
        for done, (lqs, gts) in enumerate(pool.map(job, enumerate(train), chunksize=4)):
            lq_out[cursor : cursor + len(lqs)] = lqs
            gt_out[cursor : cursor + len(gts)] = gts
            cursor += len(lqs)
            if done % 250 == 0:
                print(f"{done}/{len(train)}", flush=True)

    lq_out.flush()
    gt_out.flush()
    print(f"готово: {cursor} пар {patch}→{gt_patch} в {out.name}/, проверка — {len(val)} кадров")


if __name__ == "__main__":
    main()
