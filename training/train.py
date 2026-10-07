"""Дообучение SRVGGNetCompact (SuperUltraCompact, 24nf/8nc/×2) на порче Kodik.

    python train.py --init model.onnx --run runs/<имя> [--iters 30000]

Стартует с весов ONNX (AnimeJaNai V2 SUC — то, что сейчас в «Деталях»),
учится на `data/train/{lq,gt}.npy` (`make_pairs.py`, грузятся в память), каждые `--every`
итераций считает PSNR на целых кадрах `data/val/` и пишет чекпойнт. В конце —
`model.onnx` той же формы, что исходная: её берёт `compact-to-wgsl.py convert`
из anion-dl.
"""

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper
from torch import nn

DATA = Path(__file__).parent / "data"


class Compact(nn.Module):
    """SRVGGNetCompact из Real-ESRGAN: свёртки 3×3 с PReLU, PixelShuffle и
    прибавка входа, растянутого ближайшим соседом (как в `srvgg_arch.py`)."""

    def __init__(self, num_feat: int = 24, num_conv: int = 8, scale: int = 2):
        super().__init__()
        self.scale = scale
        layers: list[nn.Module] = [nn.Conv2d(3, num_feat, 3, padding=1), nn.PReLU(num_feat)]
        for _ in range(num_conv):
            layers += [nn.Conv2d(num_feat, num_feat, 3, padding=1), nn.PReLU(num_feat)]
        layers.append(nn.Conv2d(num_feat, 3 * scale * scale, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.pixel_shuffle(self.body(x), self.scale)
        return out + F.interpolate(x, scale_factor=self.scale, mode="nearest")


def load_onnx(model: Compact, path: Path) -> None:
    """Свёртки и PReLU из графа — в порядке графа, как у `compact-to-wgsl.py`."""
    graph = onnx.load(str(path)).graph
    init = {t.name: numpy_helper.to_array(t).astype(np.float32) for t in graph.initializer}
    convs = [n for n in graph.node if n.op_type == "Conv"]
    prelus = [n for n in graph.node if n.op_type == "PRelu"]
    mods_conv = [m for m in model.body if isinstance(m, nn.Conv2d)]
    mods_prelu = [m for m in model.body if isinstance(m, nn.PReLU)]
    if len(convs) != len(mods_conv) or len(prelus) != len(mods_prelu):
        raise SystemExit(f"в ONNX {len(convs)} свёрток и {len(prelus)} PReLU — форма не та")
    with torch.no_grad():
        for node, mod in zip(convs, mods_conv):
            mod.weight.copy_(torch.from_numpy(init[node.input[1]]))
            mod.bias.copy_(torch.from_numpy(init[node.input[2]]))
        for node, mod in zip(prelus, mods_prelu):
            mod.weight.copy_(torch.from_numpy(init[node.input[1]].reshape(-1)))


def to_tensor(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """uint8 BGR (…, H, W, 3) → float RGB (…, 3, H, W) в 0..1."""
    t = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1]))
    return t.to(device).permute(*range(t.dim() - 3), -1, -3, -2).float() / 255


def sobel(x: torch.Tensor) -> torch.Tensor:
    k = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device)
    k = torch.stack([k, k.T])[:, None].repeat(3, 1, 1, 1)
    return F.conv2d(x, k, padding=1, groups=3)


def clean_input(gt: torch.Tensor) -> torch.Tensor:
    """Вход без порчи: эталон, уменьшенный усреднением 2×2 — то же, что
    INTER_AREA в `random_degrade` до порчи."""
    return F.avg_pool2d(gt, 2)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return 10 * math.log10(1 / F.mse_loss(a[..., 8:-8, 8:-8], b[..., 8:-8, 8:-8]).item())


def validate(model: Compact, teacher: Compact, device: torch.device) -> tuple[float, float]:
    """PSNR против эталона и против учителя на чистом входе."""
    model.eval()
    vs_gt, vs_teacher = [], []
    with torch.no_grad():
        for lq_path in sorted((DATA / "val" / "lq").glob("*.png")):
            lq = to_tensor(cv2.imread(str(lq_path)), device)[None]
            gt = to_tensor(cv2.imread(str(DATA / "val" / "gt" / lq_path.name)), device)[None]
            # В плеере выход уходит в rgba8unorm — там он и обрезается до 0..1.
            out = model(lq).clamp(0, 1)
            vs_gt.append(psnr(out, gt))
            vs_teacher.append(psnr(out, teacher(clean_input(gt)).clamp(0, 1)))
    model.train()
    return float(np.mean(vs_gt)), float(np.mean(vs_teacher))


def export(model: Compact, path: Path) -> None:
    model = model.to("cpu").eval()
    torch.onnx.export(
        model,
        torch.rand(1, 3, 64, 64),
        str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h2", 3: "w2"}},
        opset_version=13,
        dynamo=False,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--init", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--iters", type=int, default=30_000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--edge", type=float, default=0.5, help="вес L1 по градиентам Собеля")
    p.add_argument(
        "--teacher",
        type=float,
        default=0.0,
        help="доля цели «исходная модель на чистом входе» вместо эталона: "
        "L1 к эталону тянет к мылу, учитель сохраняет вид JaNai",
    )
    p.add_argument("--every", type=int, default=2_000)
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    args.run.mkdir(parents=True, exist_ok=True)
    log = open(args.run / "log.txt", "a")

    def say(text: str) -> None:
        print(text, flush=True)
        log.write(text + "\n")
        log.flush()

    model = Compact().to(device)
    load_onnx(model, args.init)
    teacher = Compact().to(device).eval().requires_grad_(False)
    load_onnx(teacher, args.init)
    say(f"старт с {args.init.name}, учитель {args.teacher}")
    say("val PSNR к эталону {:.3f} дБ, к учителю {:.3f} дБ".format(*validate(model, teacher, device)))

    # Целиком в память (~3 ГБ): случайное чтение из memmap держало 4 it/s.
    lq_all = np.load(DATA / "train" / "lq.npy")
    gt_all = np.load(DATA / "train" / "gt.npy")
    say(f"пар: {len(lq_all)}, устройство: {device}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, eta_min=args.lr / 100)
    rng = np.random.default_rng(0)
    best = -1.0
    started = time.time()
    running = 0.0

    for it in range(1, args.iters + 1):
        idx = np.sort(rng.choice(len(lq_all), args.batch, replace=False))
        lq = to_tensor(lq_all[idx], device)
        gt = to_tensor(gt_all[idx], device)
        # Повороты и отражения: порча x264 от них не меняется по смыслу.
        if rng.random() < 0.5:
            lq, gt = lq.flip(-1), gt.flip(-1)
        if rng.random() < 0.5:
            lq, gt = lq.flip(-2), gt.flip(-2)
        if rng.random() < 0.5:
            lq, gt = lq.transpose(-1, -2), gt.transpose(-1, -2)

        out = model(lq)
        loss = 0.0
        if args.teacher < 1:
            loss += (1 - args.teacher) * (F.l1_loss(out, gt) + args.edge * F.l1_loss(sobel(out), sobel(gt)))
        if args.teacher > 0:
            with torch.no_grad():
                target = teacher(clean_input(gt))
            loss += args.teacher * (F.l1_loss(out, target) + args.edge * F.l1_loss(sobel(out), sobel(target)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item()

        if it % 200 == 0:
            rate = it / (time.time() - started)
            say(f"{it:>6}  loss {running / 200:.5f}  lr {sched.get_last_lr()[0]:.2e}  {rate:.1f} it/s")
            running = 0.0
        if it % args.every == 0 or it == args.iters:
            vs_gt, vs_teacher = validate(model, teacher, device)
            # «Лучший» — по той цели, к которой модель и тянется.
            value = vs_teacher * args.teacher + vs_gt * (1 - args.teacher)
            torch.save(model.state_dict(), args.run / "last.pt")
            torch.save(model.state_dict(), args.run / f"it{it}.pt")
            mark = ""
            if value > best:
                best = value
                torch.save(model.state_dict(), args.run / "best.pt")
                mark = "  ← лучший"
            say(f"{it:>6}  val PSNR к эталону {vs_gt:.3f} дБ, к учителю {vs_teacher:.3f} дБ{mark}")

    model.load_state_dict(torch.load(args.run / "best.pt"))
    export(model, args.run / "model.onnx")
    say(f"лучшая цель {best:.3f} дБ, экспорт: {args.run / 'model.onnx'}")


if __name__ == "__main__":
    main()
