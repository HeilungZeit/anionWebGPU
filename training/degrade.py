"""Порча кадра «как у Kodik».

Цепочка: смягчение (ужатие в `soft` раз и растяжение обратно — так выглядят
пересчёты разрешения у релиз-групп до Kodik) → короткий клип с панорамой →
libx264 с настройками самого Kodik → декодированный средний кадр.

Настройки x264 сняты с SEI серий Kodik (одинаковы у всех озвучек на
1000 кбит/с): ref=2, deblock=1:0:0, me=hex, subme=6, psy_rd=1.00:0.00,
trellis=1, 8x8dct, bframes=3, b_adapt=1, weightp=1, keyint=90, aq=1:1.00,
mbtree. Kodik жмёт в два прохода в битрейт, а здесь — CRF: качество
одиночного клипа нельзя привязать к битрейту целой серии, зато CRF подбирается
по метрикам кадров (`calibrate.py`).

Клип нужен потому, что почти все кадры Kodik — P и B, а у них свои
артефакты: «плывущие» блоки и смазанная заливка. Одиночный I-кадр их не даёт.
"""

import subprocess

import cv2
import numpy as np

X264_PARAMS = ":".join(
    [
        "ref=2",
        "deblock=1,0",
        "me=hex",
        "subme=6",
        "psy-rd=1.00,0.00",
        "trellis=1",
        "8x8dct=1",
        "bframes=3",
        "b-adapt=1",
        "weightp=1",
        "keyint=90",
        "min-keyint=9",
        "aq-mode=1",
        "aq-strength=1.0",
        "rc-lookahead=30",
    ]
)

CLIP_LEN = 7
"""Длина клипа: I P B B B P …, средний кадр — B, как большинство у Kodik."""


def soften(img: np.ndarray, soft: float, down: int = cv2.INTER_AREA) -> np.ndarray:
    """`down` — ресемплер группы: AniLibria уменьшает 1080p→720p Lanczos'ом
    (проверено на паре 720p/1080p с CDN), отсюда лишняя резкость и ореолы."""
    if soft >= 1.0:
        return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (round(w * soft), round(h * soft)), interpolation=down)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def pan_clip(img: np.ndarray, speed: float) -> list[np.ndarray]:
    """Кадры панорамы со сдвигом `speed` пикселей на кадр по горизонтали."""
    h, w = img.shape[:2]
    frames = []
    for k in range(CLIP_LEN):
        m = np.float32([[1, 0, speed * (k - CLIP_LEN // 2)], [0, 1, 0]])
        frames.append(cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT))
    return frames


# BT.709, ограниченный диапазон: так браузер показывает HD-видео без меток
# цвета, а у серий Kodik меток нет. Перевод — свой, а не swscale: тот с
# настройками по умолчанию темнил кадр на 1.3 уровня и сдвигал цвет (B и R
# на −1.8) даже без сжатия, и сеть выучила обратный сдвиг.
KR, KB = 0.2126, 0.0722
KG = 1 - KR - KB


def bgr_to_yuv420(bgr: np.ndarray) -> bytes:
    b, g, r = np.moveaxis(bgr.astype(np.float32) / 255, -1, 0)
    y = KR * r + KG * g + KB * b
    cb = (b - y) / (2 * (1 - KB))
    cr = (r - y) / (2 * (1 - KR))
    h, w = y.shape
    # Цветность — среднее по квадрату 2×2.
    cb = cb.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    cr = cr.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    planes = [16 + 219 * y, 128 + 224 * cb, 128 + 224 * cr]
    return b"".join(np.clip(np.round(p), 0, 255).astype(np.uint8).tobytes() for p in planes)


def yuv420_to_bgr(raw: bytes, w: int, h: int) -> list[np.ndarray]:
    data = np.frombuffer(raw, np.uint8)
    size = w * h * 3 // 2
    frames = []
    for start in range(0, len(data), size):
        frame = data[start : start + size].astype(np.float32)
        y = (frame[: w * h].reshape(h, w) - 16) / 219
        cb = (frame[w * h : w * h * 5 // 4].reshape(h // 2, w // 2) - 128) / 224
        cr = (frame[w * h * 5 // 4 :].reshape(h // 2, w // 2) - 128) / 224
        cb = cv2.resize(cb, (w, h), interpolation=cv2.INTER_LINEAR)
        cr = cv2.resize(cr, (w, h), interpolation=cv2.INTER_LINEAR)
        r = y + 2 * (1 - KR) * cr
        b = y + 2 * (1 - KB) * cb
        g = (y - KR * r - KB * b) / KG
        bgr = np.stack([b, g, r], axis=-1)
        frames.append(np.clip(np.round(bgr * 255), 0, 255).astype(np.uint8))
    return frames


def x264_roundtrip(frames: list[np.ndarray], crf: float, params: str = X264_PARAMS) -> list[np.ndarray]:
    h, w = frames[0].shape[:2]
    encode = subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-r", "24000/1001", "-i", "-",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            "-x264-params", params,
            "-f", "h264", "-",
        ],
        input=b"".join(bgr_to_yuv420(f) for f in frames),
        capture_output=True,
        check=True,
    )
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "h264", "-i", "-", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        input=encode.stdout,
        capture_output=True,
        check=True,
    )
    return yuv420_to_bgr(decode.stdout, w, h)


UPSTREAM_PARAMS = "deblock=-1,-1:psy-rd=1.00,0.15:bframes=3:keyint=240"
"""Энкод релиз-группы до Kodik — по образцу AniLibria (Main, deblock -1:-1)."""


def degrade(
    img: np.ndarray,
    soft: float,
    crf: float,
    speed: float = 1.5,
    pre_crf: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (испорченный кадр, эталон к нему) — эталон сдвинут так же.

    `pre_crf` — сжатие источника до Kodik: релиз-группа отдаёт уже сжатое
    видео, и Kodik жмёт его повторно.
    """
    clip = pan_clip(img, speed)
    frames = [soften(f, soft) for f in clip]
    if pre_crf is not None:
        frames = x264_roundtrip(frames, pre_crf, UPSTREAM_PARAMS)
    out = x264_roundtrip(frames, crf)
    mid = CLIP_LEN // 2
    return out[mid], clip[mid]


def darken(img: np.ndarray, gamma: float, gain: float) -> np.ndarray:
    return np.clip(255 * gain * (img / 255.0) ** gamma, 0, 255).astype(np.uint8)


def sharpen(img: np.ndarray, amount: float, radius: float) -> np.ndarray:
    """Нерезкая маска релиз-группы: тёмные и светлые ореолы вдоль контуров."""
    blur = cv2.GaussianBlur(img.astype(np.float32), (0, 0), radius)
    return np.clip(img + amount * (img - blur), 0, 255).astype(np.uint8)


def grain(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, sigma, img.shape[:2])[..., None]
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def random_degrade(
    img: np.ndarray, rng: np.random.Generator, scale: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Случайная порча из диапазонов, подобранных по пяти сериям Kodik.

    Эталон тоже меняется (затемнение), поэтому возвращается пара
    (испорченный кадр, эталон). При `scale` > 1 кадр сперва уменьшается в
    `scale` раз и портится уже в этом масштабе — это вход сети, эталон остаётся
    в исходном. Порча задана в пикселях входа, как у Kodik в пикселях 720p.
    Диапазоны и откуда они взялись — в docs/PLAN.md, этап Э11.
    """
    # Тёмные сцены: у Kodik средняя яркость 60–97 против 124 у APISR, а
    # блочность Kodik (1.10–1.27) получается только на тёмном.
    if rng.random() < 0.5:
        img = darken(img, rng.uniform(1.2, 2.4), rng.uniform(0.4, 0.9))
    gt = img
    if scale > 1:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)

    clip = pan_clip(img, rng.choice([0.0, rng.uniform(0.5, 4.0)]))
    soft = rng.uniform(0.4, 0.9)
    down = cv2.INTER_LANCZOS4 if rng.random() < 0.5 else cv2.INTER_AREA
    frames = [soften(f, soft, down) for f in clip]
    if rng.random() < 0.25:
        amount, radius = rng.uniform(0.3, 1.0), rng.uniform(1.0, 2.5)
        frames = [sharpen(f, amount, radius) for f in frames]
    if rng.random() < 0.35:
        sigma = rng.uniform(0.8, 4.0)
        frames = [grain(f, sigma, rng) for f in frames]
    if rng.random() < 0.6:
        frames = x264_roundtrip(frames, rng.uniform(16, 26), UPSTREAM_PARAMS)
    # Не все серии Kodik перекодирует: энкод релиз-группы (AniLibria
    # 2.5 Мбит/с, 2x2) он отдаёт как есть, и зерно в нём выживает.
    if rng.random() < 0.3:
        out = x264_roundtrip(frames, rng.uniform(14, 24), UPSTREAM_PARAMS)
    else:
        out = x264_roundtrip(frames, rng.uniform(24, 40))
    # Средний кадр панорамы не сдвинут, так что эталон к нему — сам `gt`.
    return out[CLIP_LEN // 2], gt
