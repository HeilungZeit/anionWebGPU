"""Сети Anime4K (GLSL mpv) в PyTorch — по разбору `conversion/cnn.py`.

Нужны для целей обучения (Э23: U48-3 на входе после Restore CNNM) и
офлайн-сравнений. Математика та же, что в шейдере: свёртки mat4 по сдвигам
texOff с краем clamp, CReLU, сложение с MAIN; у ×2 — Depth-to-Space плюс
билинейно растянутый MAIN. Четвёртый канал MAIN — 1, как у видео в WebGPU.

    from anime4k import Anime4K
    restore = Anime4K("Anime4K_Restore_CNN_M.glsl").to(device)
    out = restore(rgb)  # (B, 3, H, W) в 0..1
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

GLSL = Path(__file__).parent.parent / "conversion" / "glsl" / "anime4k"
sys.path.insert(0, str(GLSL.parent.parent))
from cnn import parse  # noqa: E402


def shift(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """texOff(dx, dy): out[y][x] = x[y + dy][x + dx], край — clamp."""
    if dx == 0 and dy == 0:
        return x
    p = max(abs(dx), abs(dy))
    h, w = x.shape[-2:]
    return F.pad(x, (p, p, p, p), mode="replicate")[..., p + dy : p + dy + h, p + dx : p + dx + w]


class Anime4K(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.convs, self.d2s = parse(str(GLSL / name))
        # mat4(…) в GLSL — по столбцам: M·v = Σ_j w[4j + i]·v_j.
        mats = [torch.tensor([float(v) for v in w.split(",")]).view(4, 4).T
                for conv in self.convs for _, _, _, w in conv.terms]
        self.register_buffer("mats", torch.stack(mats))
        self.register_buffer("bias", torch.stack([torch.tensor([float(v) for v in c.bias.split(",")]) for c in self.convs]))

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        main = torch.cat([rgb, torch.ones_like(rgb[:, :1])], 1)
        tex = {"MAIN": main}
        k = 0
        for n, conv in enumerate(self.convs):
            out = self.bias[n].view(1, 4, 1, 1).expand(rgb.shape[0], 4, *rgb.shape[-2:]).clone()
            for i, dx, dy, _ in conv.terms:
                src = conv.sources[i]
                x = tex[src.texture]
                x = F.relu(x) if src.sign == "+" else F.relu(-x) if src.sign == "-" else x
                out += torch.einsum("ij,bjhw->bihw", self.mats[k], shift(x, dx, dy))
                k += 1
            if conv.residual:
                out = out + tex["MAIN"]
            tex[conv.save] = out
        if self.d2s is None:
            return tex["MAIN"][:, :3]
        chans = [F.pixel_shuffle(tex[name], 2) for name in self.d2s.channels[:3]]
        return torch.cat(chans, 1) + F.interpolate(rgb, scale_factor=2, mode="bilinear", align_corners=False)
