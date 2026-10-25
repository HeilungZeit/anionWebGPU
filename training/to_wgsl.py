"""Веса CompactU (Э19) для ядра «Деталей»: `model.json` + `weights.bin`.

    python to_wgsl.py runs/<имя>/best.pt out/

Раскладка блоба — как у `compact-to-wgsl.py convert` из anion-dl: по слою
веса `[out][tap][inGroup]` vec4<f16> (хвост входных каналов — нули), затем
смещения и наклоны PReLU, добитые до кратности четырём. В описании —
`arch: SRVGGNetCompactU` и `unshuffle: 2`: первый слой читает кадр,
разложенный pixel-unshuffle 2×2 (канал c·4 + dy·2 + dx), последний отдаёт
3·(2·scale)² каналов под pixelshuffle ×4. SUC конвертируется в anion-dl.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from torch import nn

from train import CompactU, load_any

PACK = 4


def groups(channels: int) -> int:
    return -(-channels // PACK)


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    model = load_any(src)
    if not isinstance(model, CompactU):
        raise SystemExit("это не CompactU: SUC конвертирует compact-to-wgsl.py из anion-dl")
    convs = [m for m in model.body if isinstance(m, nn.Conv2d)]
    prelus = [m for m in model.body if isinstance(m, nn.PReLU)]

    blob = bytearray()
    layers = []
    for index, conv in enumerate(convs):
        weight = conv.weight.detach().numpy().astype(np.float32)
        out_c, in_c = weight.shape[:2]
        in_g = groups(in_c)
        w = np.zeros((out_c, 9, in_g * PACK), dtype=np.float32)
        w[:, :, :in_c] = weight.reshape(out_c, in_c, 9).transpose(0, 2, 1)
        offset = len(blob)
        blob += w.astype(np.float16).tobytes()
        padded = groups(out_c) * PACK
        bias = np.zeros(padded, dtype=np.float32)
        bias[:out_c] = conv.bias.detach().numpy()
        blob += bias.astype(np.float16).tobytes()
        prelu = index < len(prelus)
        if prelu:
            slope = np.zeros(padded, dtype=np.float32)
            slope[:out_c] = prelus[index].weight.detach().numpy()
            blob += slope.astype(np.float16).tobytes()
        layers.append({
            "name": f"body.{index * 2}",
            "offset": offset,
            "in": in_c,
            "out": out_c,
            "inGroups": in_g,
            "outGroups": groups(out_c),
            "prelu": prelu,
        })

    meta = {
        "arch": "SRVGGNetCompactU",
        "license": "",
        "numFeat": convs[0].out_channels,
        "numConv": len(convs) - 2,
        "scale": model.scale,
        "unshuffle": 2,
        "pack": PACK,
        "dtype": "float16",
        "source": f"{src.parent.name}/{src.name}",
        "sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": len(blob),
        "layers": layers,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights.bin").write_bytes(bytes(blob))
    (out / "model.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"{out}: {len(layers)} слоёв, {len(blob)} Б")


if __name__ == "__main__":
    main()
