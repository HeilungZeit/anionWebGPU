"""Пары для обучения ×2: эталон APISR и он же, уменьшенный вдвое и испорченный.

    python make_pairs.py

Порча идёт через настоящий x264 (~0.1 с на кадр), поэтому пары готовятся
заранее, а не в загрузчике. На каждый кадр — VARIANTS вариантов порчи, из
каждого — CROPS случайных фрагментов. Всё складывается в два массива uint8
на диске (`data/train/lq.npy`, `gt.npy`), обучение читает их через memmap.

Последние VAL кадров (по имени) не участвуют: из них делаются целые кадры
для проверки, `data/val/{lq,gt}/`.
"""

from concurrent.futures import ProcessPoolExecutor
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


def train_job(args: tuple[int, Path]) -> tuple[np.ndarray, np.ndarray]:
    index, path = args
    rng = np.random.default_rng(index)
    img = load(path)
    lqs, gts = [], []
    for _ in range(VARIANTS):
        lq, gt = random_degrade(img, rng, SCALE)
        h, w = lq.shape[:2]
        for _ in range(CROPS):
            y = int(rng.integers(0, h - LQ_PATCH + 1))
            x = int(rng.integers(0, w - LQ_PATCH + 1))
            lqs.append(lq[y : y + LQ_PATCH, x : x + LQ_PATCH])
            gts.append(gt[y * SCALE : (y + LQ_PATCH) * SCALE, x * SCALE : (x + LQ_PATCH) * SCALE])
    return np.stack(lqs), np.stack(gts)


def val_job(args: tuple[int, Path]) -> None:
    index, path = args
    lq, gt = random_degrade(load(path), np.random.default_rng(10_000 + index), SCALE)
    cv2.imwrite(str(DATA / "val" / "lq" / path.name), lq)
    cv2.imwrite(str(DATA / "val" / "gt" / path.name), gt)


def main() -> None:
    sources = sorted((DATA / "apisr" / "APISR_Dataset").glob("*.png"))
    train, val = sources[:-VAL], sources[-VAL:]

    for sub in ("val/lq", "val/gt", "train"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)

    count = len(train) * VARIANTS * CROPS
    gt_patch = LQ_PATCH * SCALE
    lq_out = np.lib.format.open_memmap(
        DATA / "train" / "lq.npy", "w+", np.uint8, (count, LQ_PATCH, LQ_PATCH, 3)
    )
    gt_out = np.lib.format.open_memmap(
        DATA / "train" / "gt.npy", "w+", np.uint8, (count, gt_patch, gt_patch, 3)
    )

    with ProcessPoolExecutor() as pool:
        list(pool.map(val_job, enumerate(val)))
        cursor = 0
        for done, (lqs, gts) in enumerate(pool.map(train_job, enumerate(train), chunksize=4)):
            lq_out[cursor : cursor + len(lqs)] = lqs
            gt_out[cursor : cursor + len(gts)] = gts
            cursor += len(lqs)
            if done % 250 == 0:
                print(f"{done}/{len(train)}", flush=True)

    lq_out.flush()
    gt_out.flush()
    print(f"готово: {cursor} пар {LQ_PATCH}→{gt_patch}, проверка — {len(val)} кадров")


if __name__ == "__main__":
    main()
