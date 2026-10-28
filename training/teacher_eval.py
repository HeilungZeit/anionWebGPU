"""Сравнение кандидатов в учителя «Деталей» (Э20): чужие модели ×2 из сети
против своих на тех же кадрах, что у `pilot_eval.py`.

    python teacher_eval.py run [--step=N] имя=модель [имя=модель ...]
    python teacher_eval.py report имя [имя ...]
    python teacher_eval.py combos лист [подпись=]выражение [...]

Модель — ONNX SUC или чекпойнт `.pt` (как у `pilot_eval.py`), `.pth` или
`.safetensors` любой архитектуры, которую знает spandrel (RCAN, DAT, PLKSR,
ESRGAN…), или `ort:<путь.onnx>` — произвольный граф через onnxruntime (так
идут AnimeJaNai V3.1 на SPAN: в spandrel их нет, только ONNX fp16).

`run` прогоняет модель по `data/val/` (эталон есть) и всем кадрам
`data/kodik/` и пишет в `data/teachers/out/<имя>/`: `metrics.json` и
фрагменты для мозаики (те же кадры и середина, что у `compare.py`). Модели
считаются по одной — тяжёлые (DAT2, RCAN) идут минутами на кадр, а результат
каждой сохраняется сразу. `--step=N` — каждый N-й кадр (и все кадры
мозаики): для DAT2, у которого кадр 720p — три минуты; `report` сводит
модели по общим кадрам.

`report` печатает сводку и собирает мозаики `data/teachers/mosaic/`: по
картинке на фрагмент, сетка подписанных плиток — вход (ближайший сосед) и
модели в порядке аргументов. Первая модель — база для «штрих/база».

`combos` — листы вариантов составной цели (`data/teachers/combos/<лист>/`)
на тех же фрагментах: `u48-2+apisr-rrdb@0.75` — низ U48-2 и верх APISR выше
гаусса σ0.75; `*k` — сила верха; несколько `+…@σ` применяются слева направо.
`…^u48-2` — на штрихе (мягкая маска линий модели после `^`) пиксели этой
модели: линии берутся у неё, остальное — у выражения слева.

Метрики Kodik — как в `pilot_eval.py` (шум, детали, блоки, 2×2), но маски
штриха берутся из входа (бикубика ×2, темнее окружения на 20+), а не из выхода
базы: модели считаются порознь. «Штрих/база» — разница средней яркости
выхода на этих пикселях против первой модели: плюс — светлее базы.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from pilot_eval import INK, grid_scores, hf_std, luma, phase_steps, tile_masks
from train import DATA, load_any, to_tensor

OUT = DATA / "teachers" / "out"
MOSAIC = DATA / "teachers" / "mosaic"
# Фрагменты — как у `compare.py`: кадры серий и середина 256×144 с запасом.
FRAMES = ("08", "20", "32")
CROP = (256, 144)
PAD = 32
# DAT2 целым кадром 720p не влезает: внимание по окнам, но память — на весь
# кадр. Плитки с запасом PAD с каждой стороны, склейка без перекрытия.
TILE = 256


class Runner:
    """uint8 BGR (H, W, 3) → float RGB (2H, 2W, 3) в уровнях 8 бит, 0..255."""

    def __init__(self, spec: str, device: torch.device):
        self.device = device
        self.ort = None
        self.tile = 0
        if spec.startswith("ort:"):
            import onnxruntime as ort

            # 4 потока, не все ядра: прогоны идут часами и греют ноутбук.
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            self.ort = ort.InferenceSession(spec[4:], options, providers=["CPUExecutionProvider"])
            inp = self.ort.get_inputs()[0]
            self.ort_name = inp.name
            self.ort_type = np.float16 if "float16" in inp.type else np.float32
            return
        path = Path(spec)
        if path.suffix in (".pth", ".safetensors"):
            import spandrel

            desc = spandrel.ModelLoader().load_from_file(str(path))
            if desc.scale != 2:
                raise SystemExit(f"{path}: масштаб ×{desc.scale}, нужен ×2")
            self.model = desc.model.to(device).eval()
            self.tile = TILE if desc.architecture.name == "DAT" else 0
        else:
            self.model = load_any(path).to(device).eval()

    def net(self, rgb: np.ndarray) -> np.ndarray:
        """float RGB (h, w, 3) 0..1 → то же ×2."""
        if self.ort is not None:
            x = rgb.transpose(2, 0, 1)[None].astype(self.ort_type)
            return self.ort.run(None, {self.ort_name: x})[0][0].astype(np.float32).transpose(1, 2, 0)
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None].to(self.device)
        with torch.no_grad():
            return self.model(x)[0].permute(1, 2, 0).cpu().numpy()

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        rgb = bgr[..., ::-1].astype(np.float32) / 255
        if not self.tile:
            out = self.net(rgb)
        else:
            h, w = rgb.shape[:2]
            padded = np.pad(rgb, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="reflect")
            out = np.zeros((h * 2, w * 2, 3), np.float32)
            for y in range(0, h, self.tile):
                for x in range(0, w, self.tile):
                    th, tw = min(self.tile, h - y), min(self.tile, w - x)
                    tile = self.net(padded[y : y + th + 2 * PAD, x : x + tw + 2 * PAD])
                    out[2 * y : 2 * (y + th), 2 * x : 2 * (x + tw)] = tile[2 * PAD : 2 * (PAD + th), 2 * PAD : 2 * (PAD + tw)]
        return np.clip(out, 0, 1) * 255


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Как `train.psnr`: без каймы 8 px."""
    mse = np.mean((a[8:-8, 8:-8] / 255 - b[8:-8, 8:-8] / 255) ** 2)
    return float(10 * np.log10(1 / mse))


def ink_mask(bgr: np.ndarray) -> np.ndarray:
    """Штрих по входу ×2 бикубикой: темнее окружения (гаусс σ4) на `INK`+."""
    h, w = bgr.shape[:2]
    up = luma(cv2.resize(bgr[..., ::-1].astype(np.float32), (w * 2, h * 2), interpolation=cv2.INTER_CUBIC))
    mask = cv2.GaussianBlur(up, (0, 0), 4) - up > INK
    mask[:8], mask[-8:], mask[:, :8], mask[:, -8:] = False, False, False, False
    return mask


def run(name: str, spec: str, step: int) -> None:
    import time

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = Runner(spec, device)
    out_dir = OUT / name
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    started = time.time()

    val = {}
    for lq_path in sorted((DATA / "val" / "lq").glob("*.png"))[::step]:
        gt = cv2.imread(str(DATA / "val" / "gt" / lq_path.name))[..., ::-1].astype(np.float32)
        out = model(cv2.imread(str(lq_path)))
        ref = luma(gt)
        depth = cv2.GaussianBlur(ref, (0, 0), 4) - ref
        ink = depth > INK
        ink[:8], ink[-8:], ink[:, :8], ink[:, -8:] = False, False, False, False
        ink_c = float((luma(out) - ref)[ink].mean() / depth[ink].mean()) if ink.sum() >= 100 else None
        val[lq_path.stem] = [psnr(out, gt), ink_c]
    print(f"{name}: val {time.time() - started:.0f} с", flush=True)

    # По кадру: суммы и счёт СКО плиток (шум, детали), перепады по фазам,
    # средняя яркость на штрихе — `report` складывает их по общим кадрам.
    kodik = {}
    paths = sorted((DATA / "kodik").glob("*/*.png"))
    paths = [p for i, p in enumerate(paths) if i % step == 0 or p.stem in FRAMES]
    for i, path in enumerate(paths):
        bgr = cv2.imread(str(path))
        out = model(bgr)
        y = luma(out)
        flat, texture = tile_masks(bgr)
        noise, detail = hf_std(y, flat), hf_std(y, texture)
        ink = ink_mask(bgr)
        kodik[f"{path.parent.name}-{path.stem}"] = {
            "noise": [float(np.sum(noise)), len(noise)],
            "detail": [float(np.sum(detail)), len(detail)],
            "steps": phase_steps(y).tolist(),
            "ink": float(y[ink].mean()) if ink.sum() >= 100 else None,
        }
        if path.stem in FRAMES:
            h, w = bgr.shape[:2]
            cw, ch = CROP
            cy, cx = (h - ch) // 2, (w - cw) // 2
            crop = out[2 * cy : 2 * (cy + ch), 2 * cx : 2 * (cx + cw)]
            cv2.imwrite(str(out_dir / "crops" / f"{path.parent.name}-{path.stem}.png"), (crop + 0.5).astype(np.uint8)[..., ::-1])
        if i % 25 == 0:
            print(f"{name}: kodik {i}/{len(paths)}, {time.time() - started:.0f} с", flush=True)

    seconds = (time.time() - started) / (len(paths) + len(val))
    metrics = {"spec": spec, "seconds_per_frame": seconds, "val": val, "kodik": kodik}
    (out_dir / "metrics.json").write_text(json.dumps(metrics))
    print(f"{name}: готово за {time.time() - started:.0f} с", flush=True)


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (8 + 11 * len(text), 24), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def report(names: list[str]) -> None:
    metrics = {name: json.loads((OUT / name / "metrics.json").read_text()) for name in names}
    val_keys = sorted(set.intersection(*(set(m["val"]) for m in metrics.values())))
    kodik_keys = sorted(set.intersection(*(set(m["kodik"]) for m in metrics.values())))
    print(f"кадров: val {len(val_keys)}, Kodik {len(kodik_keys)}")
    base = metrics[names[0]]["kodik"]
    print(f"{'модель':16s} {'к эталону':>10s} {'штрих/эт.':>10s} {'шум':>6s} {'детали':>7s}"
          f" {'блоки':>6s} {'2×2':>5s} {'штрих/база':>11s} {'с/кадр':>7s}")
    for name, m in metrics.items():
        v = np.array([m["val"][k] for k in val_keys], dtype=float)
        k = [m["kodik"][key] for key in kodik_keys]
        noise = sum(f["noise"][0] for f in k) / sum(f["noise"][1] for f in k)
        detail = sum(f["detail"][0] for f in k) / sum(f["detail"][1] for f in k)
        blocks, pairs = grid_scores(sum(np.array(f["steps"]) for f in k))
        ink = np.nanmean([np.nan if f["ink"] is None else f["ink"] - base[key]["ink"] for f, key in zip(k, kodik_keys)])
        print(f"{name:16s} {np.mean(v[:, 0]):9.2f}  {np.nanmean(v[:, 1]):+9.1%} {noise:6.2f} {detail:7.2f}"
              f" {blocks:6.3f} {pairs:5.2f} {ink:+10.2f} {m['seconds_per_frame']:7.2f}")

    for crop_path in sorted((OUT / names[0] / "crops").glob("*.png")):
        tiles = [label(cv2.imread(str(OUT / name / "crops" / crop_path.name)), name) for name in names]
        sheet(MOSAIC / crop_path.name, crop_path.stem, tiles)
    print(f"мозаики: {MOSAIC}")


def sheet(path: Path, crop: str, tiles: list[np.ndarray], cols: int = 4) -> None:
    """Сетка плиток, первая — вход фрагмента `crop` (`серия-кадр`)."""
    series, frame = crop.split("-")
    src = cv2.imread(str(DATA / "kodik" / series / f"{frame}.png"))
    h, w = src.shape[:2]
    cw, ch = CROP
    cy, cx = (h - ch) // 2, (w - cw) // 2
    tiles = [label(cv2.resize(src[cy : cy + ch, cx : cx + cw], (cw * 2, ch * 2), interpolation=cv2.INTER_NEAREST), "input"), *tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]))


def combine(expr: str, crop: str) -> np.ndarray:
    """Фрагмент по выражению: `модель` или `база+модель@σ[*k]+…` — слева
    направо полоса выше σ заменяется верхом следующей модели:
    `lo(r) + k·(m − lo(m))`, lo — гаусс σ в пикселях выхода. `…^модель` —
    штрих этой модели поверх (`ink_overlay`)."""
    expr, *overlays = expr.split("^")
    first, *terms = expr.split("+")
    out = cv2.imread(str(OUT / first / "crops" / f"{crop}.png")).astype(np.float32)
    for term in terms:
        name, _, band = term.partition("@")
        sigma, _, k = band.partition("*")
        m = cv2.imread(str(OUT / name / "crops" / f"{crop}.png")).astype(np.float32)
        s = float(sigma)
        out = cv2.GaussianBlur(out, (0, 0), s) + float(k or 1) * (m - cv2.GaussianBlur(m, (0, 0), s))
    for name in overlays:
        out = ink_overlay(out, cv2.imread(str(OUT / name / "crops" / f"{crop}.png")).astype(np.float32))
    return (np.clip(out, 0, 255) + 0.5).astype(np.uint8)


def ink_overlay(out: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Линии `m` поверх `out`: маска — темнее гаусса σ4 на 8–20 уровней
    (мягкий порог) у `m` и хотя бы на 4–12 у `out` — линии, которых в
    `out` нет (U48-2 дорисовывает их на дымке), не переносятся. Маска
    расширена на пиксель и сглажена: край линии и её сглаживание тоже от `m`."""

    def depth(img: np.ndarray) -> np.ndarray:
        y = luma(img[..., ::-1])
        return cv2.GaussianBlur(y, (0, 0), 4) - y

    mask = np.minimum(np.clip((depth(m) - 8) / 12, 0, 1), np.clip((depth(out) - 4) / 8, 0, 1))
    mask = cv2.GaussianBlur(cv2.dilate(mask, np.ones((3, 3), np.uint8)), (0, 0), 1)[..., None]
    return mask * m + (1 - mask) * out


def combos(name: str, columns: list[str]) -> None:
    """Лист вариантов цели: по картинке на фрагмент, плитки — `подпись=выражение`."""
    named = [c.split("=", 1) if "=" in c else (c, c) for c in columns]
    first = named[0][1].split("+")[0]
    for crop_path in sorted((OUT / first / "crops").glob("*.png")):
        tiles = [label(combine(expr, crop_path.stem), title) for title, expr in named]
        sheet(DATA / "teachers" / "combos" / name / crop_path.name, crop_path.stem, tiles)
    print(f"листы: {DATA / 'teachers' / 'combos' / name}")


def main() -> None:
    cmd, *args = sys.argv[1:]
    if cmd == "run":
        step = 1
        if args[0].startswith("--step="):
            step = int(args.pop(0).removeprefix("--step="))
        for name, spec in (arg.split("=", 1) for arg in args):
            run(name, spec, step)
    elif cmd == "report":
        report(args)
    elif cmd == "combos":
        combos(args[0], args[1:])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
