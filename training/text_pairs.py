"""Пары с надписями (Э22): учителя на мелком тексте мягкие, и сеть его мажет.

    python text_pairs.py [--count 4000] [--val 200] [--out train128-text]
    python text_pairs.py --sheet лист.png [--out train128-text]

Основа — пары `train128` и цель из кэша учителя. На эталон кладётся чёткий
текст, вход собирается заново: чистый фон (эталон, уменьшенный 2×2) плюс
слой надписи, уменьшенный и размытый отдельно, и та же порча, что у
остальных пар (`degrade_input`). Цель — кэш снаружи надписи и чёткий эталон
на ней (маска с запасом): учителей пересчитывать не нужно.

Две ветки:
- надписи озвучки — кириллица и латиница поверх кадра, размыты ещё до
  Kodik (σ слоя 0.3–1.0 px входа), 10–28 px: мельче и мутнее буквы на входе
  уже не читаются, и сеть училась бы их выдумывать;
- японский текст в кадре — кана и кандзи как часть рисунка (σ 0–0.6),
  10–36 px, иногда вертикально.

Проверочная часть — на шрифтах, которых нет в обучении (Verdana и
ヒラギノ丸ゴ), с маской надписи: `val_lq.npy`, `val_gt.npy`, `val_mask.npy`.
"""

import argparse
import glob
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from degrade import degrade_input

DATA = Path(__file__).parent / "data"
BASE = "train128"
CACHE = "teacher-c-sd3-a075k5-ulq.npy"
# Пул поменьше: машина одна, остальное в это время не запускаем.
WORKERS = 3

SUPPLEMENTAL = "/System/Library/Fonts/Supplemental/"
LATIN_TRAIN = [
    "Arial.ttf", "Arial Bold.ttf", "Arial Black.ttf", "Arial Narrow Bold.ttf", "Arial Rounded Bold.ttf",
    "Georgia.ttf", "Georgia Bold.ttf", "Comic Sans MS.ttf", "Comic Sans MS Bold.ttf", "Impact.ttf",
    "Times New Roman.ttf", "Times New Roman Bold.ttf", "Trebuchet MS.ttf", "Trebuchet MS Bold.ttf",
    "Tahoma Bold.ttf", "Courier New Bold.ttf",
]
LATIN_VAL = ["Verdana.ttf", "Verdana Bold.ttf"]
JP_TRAIN = [f"ヒラギノ角ゴシック W{w}" for w in (2, 3, 4, 6, 8, 9)] + ["ヒラギノ明朝 ProN", "Songti", "STHeiti Medium"]
JP_VAL = ["ヒラギノ丸ゴ ProN W4"]

RU_WORDS = """
Птичий мудрец|Таверна|Аптека|Закрыто|Открыто|Выход|Вход|Опасно|Не входить|Гильдия
искателей|Королевская академия|Школа магии|Кафе «Луна»|Книжная лавка|Станция|Библиотека
Кузница|Рынок|Храм|Замок|Деревня Листа|Северные врата|Подземелье|Уровень 5|Ранг S
Серия 12|Эпизод 7|Глава 3|Продолжение следует|Конец|Спустя три года|На следующий день
Внимание!|Срочно|Объявление|Розыск|Награда 1000 золотых|Меню|Обед|Скидки 50%|Сегодня
Перевод: Дара|Озвучка|Субтитры|Тайминг|Голоса|Звук|Релиз|Студия|Перевод и озвучка
AniLibria|AniDub|Dream Cast|SHIZA Project|StudioBand|AniMedia|JAM|Persona99|AnimeVost
Добро пожаловать|Спасибо за просмотр|Не забудьте подписаться|Тренировочный зал|Склад
Сила|Ловкость|Интеллект|Здоровье|Мана|Опыт|Задание выполнено|Новый навык|Уровень повышен
Письмо от отца|Дневник|Секретно|Совершенно секретно|Отчёт|Карта мира|Восточный лес
""".split("|")
RU_WORDS = [w.strip() for line in RU_WORDS for w in line.split("\n") if w.strip()]
EN_WORDS = ["OPEN", "CLOSED", "EXIT", "Cafe", "Book Store", "Station", "HOTEL", "SALE", "Level Up",
            "Episode 4", "To be continued", "Guild", "Shop", "Bar", "NO ENTRY", "Welcome"]
CYR = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"

JP_WORDS = """
第一話|第十二話|次回予告|終わり|つづく|東京|学校|魔法学園|勇者|魔王|冒険者ギルド|喫茶店
本屋|駅前|出口|入口|危険|立入禁止|営業中|準備中|閉店|お知らせ|ありがとう|さようなら
こんにちは|おはよう|大丈夫|行くぞ|待って|本当に|秘密|約束|友達|家族|未来|世界|運命
图书馆|図書館|病院|警察|神社|お寺|商店街|公園|屋上|教室|職員室|生徒会室|体育館
ラーメン|コーヒー|ケーキ|アニメ|ゲーム|テスト|レベル|スキル|ステータス|ダンジョン
株式会社|一年三組|二〇二四年|春夏秋冬|東西南北|上下左右|月火水木金土日|風林火山
""".split("|")
JP_WORDS = [w.strip() for line in JP_WORDS for w in line.split("\n") if w.strip()]
KANA = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん" \
       "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンー"
KANJI = "一二三四五六七八九十百千万円年月日時分人子女男父母先生学校名前国語本気電車道店駅出入口中外上下" \
        "右左大小山川田町村森林花草空雨雪風天海島橋石金銀火水木土東西南北春夏秋冬朝昼夜今週毎回話会社長" \
        "明暗白黒赤青黄色音楽歌声力心思考言語読書見聞行来帰食飲買売開閉始終止通使作休働遊走飛泳待持取送" \
        "家族友愛恋夢希望勇者魔王剣盾城門寺神社病院医薬警察危険禁止営業準備閉店新古高安強弱早遅近遠多少"


def font_path(name: str) -> str:
    """Имена японских файлов на диске в NFD — ищем по нормализованному."""
    if name.endswith(".ttf"):
        return SUPPLEMENTAL + name
    for p in glob.glob("/System/Library/Fonts/*.tt*") + glob.glob(SUPPLEMENTAL + "*.tt*"):
        if unicodedata.normalize("NFC", Path(p).stem) == name:
            return p
    raise FileNotFoundError(name)


def load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path(name), size)


def covered(font: ImageFont.FreeTypeFont, text: str) -> bool:
    """Нет ли в шрифте «тофу» вместо знаков: маска пустого места (U+E000)."""
    missing = bytes(font.getmask(""))
    return all(ch == " " or bytes(font.getmask(ch)) != missing for ch in text)


def pick_text(rng: np.random.Generator, jp: bool) -> str:
    if jp:
        r = rng.random()
        if r < 0.5:
            return str(rng.choice(JP_WORDS))
        pool = KANA if r < 0.7 else KANJI if r < 0.9 else KANA + KANJI
        return "".join(rng.choice(list(pool), int(rng.integers(2, 8))))
    r = rng.random()
    if r < 0.6:
        return str(rng.choice(RU_WORDS))
    if r < 0.75:
        return str(rng.choice(EN_WORDS))
    return "".join(rng.choice(list(CYR), int(rng.integers(3, 10))))


def contrast_color(rng: np.random.Generator, bg_luma: float) -> tuple[int, int, int]:
    """Цвет надписи, заметный на фоне: светлый на тёмном и наоборот."""
    while True:
        c = rng.integers(0, 256, 3)
        if abs(float(c @ [0.299, 0.587, 0.114]) - bg_luma) > 70:
            return tuple(int(v) for v in c)


def render_layer(rng: np.random.Generator, size: int, bg_luma: float, jp: bool,
                 fonts: list[str]) -> np.ndarray:
    """RGBA-слой надписи `size`×`size` (разрешение эталона), float 0..1."""
    while True:
        name = str(rng.choice(fonts))
        px = int(2 * (rng.uniform(10, 36) if jp else rng.uniform(10, 28)))
        font = load_font(name, px)
        text = pick_text(rng, jp)
        if covered(font, text):
            break
    vertical = jp and rng.random() < 0.3
    style = rng.choice(["outline", "plain", "shadow", "glow"], p=[0.35, 0.25, 0.2, 0.2])
    fill = contrast_color(rng, bg_luma)
    dark = sum(fill) / 3 > 128
    edge = (int(rng.integers(0, 40)),) * 3 if dark else (int(rng.integers(215, 256)),) * 3
    stroke = int(rng.integers(2, 7)) if style == "outline" else 0

    canvas = 4 * size
    layer = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx, cy = canvas // 2, canvas // 2
    if vertical:
        for k, ch in enumerate(text):
            draw.text((cx, cy + (k - len(text) / 2) * px * 1.05), ch, font=font, anchor="mm",
                      fill=fill + (255,), stroke_width=stroke, stroke_fill=edge + (255,))
    else:
        draw.text((cx, cy), text, font=font, anchor="mm",
                  fill=fill + (255,), stroke_width=stroke, stroke_fill=edge + (255,))
    if style in ("shadow", "glow"):
        alpha = layer.getchannel("A")
        if style == "shadow":
            off = int(rng.integers(2, 6))
            halo = alpha.transform(alpha.size, Image.AFFINE, (1, 0, -off, 0, 1, -off))
            halo = halo.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 2.5))))
        else:
            halo = alpha.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(float(rng.uniform(2, 5))))
        under = Image.new("RGBA", layer.size, edge + (0,))
        under.putalpha(halo)
        layer = Image.alpha_composite(under, layer)
    if rng.random() < 0.3:
        layer = layer.rotate(float(rng.uniform(-8, 8)), resample=Image.BICUBIC)

    # Надпись сдвигается так, что попадает в кадр хотя бы частью.
    box = layer.getbbox() or (0, 0, canvas, canvas)
    tw, th = box[2] - box[0], box[3] - box[1]
    x = int(rng.integers(-tw // 3, max(size - 2 * tw // 3, -tw // 3 + 1)))
    y = int(rng.integers(-th // 3, max(size - 2 * th // 3, -th // 3 + 1)))
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.alpha_composite(layer.crop(box), (max(x, 0), max(y, 0)), (max(-x, 0), max(-y, 0)))
    rgba = np.asarray(out, np.float32) / 255
    if rng.random() < 0.2:
        rgba[..., 3] *= rng.uniform(0.6, 0.9)
    return rgba


def make_pair(index: int, base: int, val: bool, gt_all: np.ndarray, target_all: np.ndarray):
    """(вход, эталон, цель, маска) для пары `base` основы."""
    rng = np.random.default_rng(1_000_000 * (1 + val) + index)
    gt = gt_all[base].astype(np.float32) / 255          # BGR
    target = target_all[base].astype(np.float32) / 255
    size = gt.shape[0]
    jp = rng.random() < 0.45
    luma = float(gt.mean(axis=(0, 1)) @ [0.114, 0.587, 0.299]) * 255
    fonts = (JP_VAL if jp else LATIN_VAL) if val else (JP_TRAIN if jp else LATIN_TRAIN)
    while True:
        rgba = render_layer(rng, size, luma, jp, fonts)
        if (rgba[..., 3] > 0.5).mean() > 0.03:
            break
    color = rgba[..., 2::-1]                             # RGB → BGR
    a = rgba[..., 3:]
    gt_text = gt * (1 - a) + color * a

    # Вход: чистый фон 2×2 и слой, уменьшенный и размытый отдельно.
    prem = cv2.resize(np.concatenate([color * a, a], -1), (size // 2, size // 2), interpolation=cv2.INTER_AREA)
    sigma = rng.uniform(0, 0.6) if jp else rng.uniform(0.3, 1.0)
    if sigma > 0.05:
        prem = cv2.GaussianBlur(prem, (0, 0), sigma)
    bg = cv2.resize(gt, (size // 2, size // 2), interpolation=cv2.INTER_AREA)
    clean = bg * (1 - prem[..., 3:]) + prem[..., :3]
    lq = degrade_input(np.clip(np.round(clean * 255), 0, 255).astype(np.uint8), rng)

    # Маска с запасом ~4 px выхода, мягкий край.
    mask = cv2.dilate((a[..., 0] > 0.01).astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    mask = np.clip(cv2.GaussianBlur(mask, (0, 0), 1.5) * 1.5, 0, 1)[..., None]
    target = target * (1 - mask) + gt_text * mask
    q = lambda x: np.clip(np.round(x * 255), 0, 255).astype(np.uint8)
    return lq, q(gt_text), q(target), q(mask[..., 0])


def job(args, gt_path: Path, target_path: Path):
    index, base, val = args
    gt_all = np.load(gt_path, mmap_mode="r")
    target_all = np.load(target_path, mmap_mode="r")
    return make_pair(index, base, val, gt_all, target_all)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=4000)
    p.add_argument("--val", type=int, default=200)
    p.add_argument("--out", default="train128-text")
    p.add_argument("--sheet", type=Path, help="только лист примеров (вход / эталон / цель), без набора")
    args = p.parse_args()
    out = DATA / args.out
    gt_path, target_path = DATA / BASE / "gt.npy", DATA / BASE / CACHE
    total = len(np.load(gt_path, mmap_mode="r"))
    bases = np.random.default_rng(7).permutation(total)
    run = partial(job, gt_path=gt_path, target_path=target_path)

    if args.sheet:
        rows = []
        for k in range(12):
            lq, gt, target, _ = run((k, int(bases[k]), k >= 9))
            up = cv2.resize(lq, gt.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            rows.append(np.concatenate([np.pad(t, ((2, 2), (2, 2), (0, 0)), constant_values=255) for t in (up, gt, target)], 1))
        sheet = np.concatenate([np.concatenate(rows[i:i + 4], 0) for i in range(0, 12, 4)], 1)
        cv2.imwrite(str(args.sheet), sheet)
        print(f"лист: {args.sheet} (столбцы: вход ×2 ближайшим, эталон с текстом, цель; последний ряд третьего столбца — проверочные шрифты)")
        return

    out.mkdir(parents=True, exist_ok=True)
    gt_size = np.load(gt_path, mmap_mode="r").shape[1]
    for prefix, count, offset, val in (("", args.count, 0, False), ("val_", args.val, args.count, True)):
        lq_out = np.lib.format.open_memmap(out / f"{prefix}lq.npy", "w+", np.uint8, (count, gt_size // 2, gt_size // 2, 3))
        gt_out = np.lib.format.open_memmap(out / f"{prefix}gt.npy", "w+", np.uint8, (count, gt_size, gt_size, 3))
        tg_out = np.lib.format.open_memmap(out / f"{prefix}target.npy", "w+", np.uint8, (count, gt_size, gt_size, 3))
        mk_out = np.lib.format.open_memmap(out / f"{prefix}mask.npy", "w+", np.uint8, (count, gt_size, gt_size))
        tasks = [(k, int(bases[offset + k]), val) for k in range(count)]
        with ProcessPoolExecutor(WORKERS) as pool:
            for k, (lq, gt, target, mask) in enumerate(pool.map(run, tasks, chunksize=16)):
                lq_out[k], gt_out[k], tg_out[k], mk_out[k] = lq, gt, target, mask
                if k % 500 == 0:
                    print(f"{prefix or 'train_'}{k}/{count}", flush=True)
        for a in (lq_out, gt_out, tg_out, mk_out):
            a.flush()
    print(f"готово: {args.count} пар + {args.val} проверочных в {out.name}/ (основа {BASE}, цель {CACHE})")


if __name__ == "__main__":
    main()
