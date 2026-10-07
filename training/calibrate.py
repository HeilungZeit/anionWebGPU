"""Сетка порчи APISR против кадров Kodik по метрикам `measure.py`.

    python calibrate.py

Портит выборку эталонов (`data/apisr_sample`) на сетке CONFIGS, пишет
испорченные кадры в `data/calib/<узел>/` и печатает их метрики рядом с
метриками серий Kodik (`data/kodik/*`). Ближайший узел сетки к каждой серии —
в конце.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from degrade import degrade
from measure import measure

DATA = Path(__file__).parent / "data"
# name → (soft, crf, speed, pre_crf)
CONFIGS = {
    f"s{soft}_c{crf}_v{speed}_p{pre}": (soft, crf, speed, pre)
    for soft in (0.85, 0.6, 0.45)
    for crf in (26, 32, 38)
    for speed in (1.5, 4.0)
    for pre in (None, 24)
}
KEYS = ("res@0.5", "res@0.75", "hf", "block", "band")


def run(job: tuple[Path, Path, tuple]) -> None:
    src, out, config = job
    if out.exists():
        return
    img = cv2.imread(str(src))
    # Кодек требует чётных сторон; редкие кадры APISR шире 16:9.
    img = img[: img.shape[0] // 2 * 2, : img.shape[1] // 2 * 2]
    lq, _ = degrade(img, *config)
    cv2.imwrite(str(out), lq)


def main() -> None:
    sources = sorted((DATA / "apisr_sample").glob("*.png"))
    jobs = []
    for name, config in CONFIGS.items():
        folder = DATA / "calib" / name
        folder.mkdir(parents=True, exist_ok=True)
        jobs += [(src, folder / src.name, config) for src in sources]
    with ProcessPoolExecutor() as pool:
        list(pool.map(run, jobs, chunksize=8))

    rows = {f.name: measure(f) for f in sorted((DATA / "kodik").iterdir())}
    kodik = list(rows)
    rows |= {name: measure(DATA / "calib" / name) for name in CONFIGS}

    print(f"{'':<24}", *(f"{k:>9}" for k in KEYS))
    for name, st in rows.items():
        print(f"{name:<24}", *(f"{st[k]:9.3f}" for k in KEYS))

    # Масштаб метрик разный — нормируем на разброс по всем строкам.
    table = np.array([[st[k] for k in KEYS] for st in rows.values()])
    scale = table.std(axis=0) + 1e-6
    names = list(rows)
    print("\nближайшая порча:")
    for i, name in enumerate(kodik):
        dist = np.linalg.norm((table[len(kodik):] - table[i]) / scale, axis=1)
        best = np.argsort(dist)[:3]
        print(f"  {name:<12}", ", ".join(f"{names[len(kodik) + j]} ({dist[j]:.2f})" for j in best))


if __name__ == "__main__":
    main()
