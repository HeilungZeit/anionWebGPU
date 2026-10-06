# Эталон для сверки ArtCNN: ONNX-модель ArtCNN на детерминированном
# кадре R=G=B. Нужны numpy и onnxruntime (python3 -m venv venv && venv/bin/pip install numpy onnxruntime).
# ONNX-модели — из https://github.com/Artoriuz/ArtCNN (папка ONNX/), путь — вторым аргументом.
# Запуск из корня: venv/bin/python bench/artcnn-onnx-ref.py bench/ref/artcnn-check <ArtCNN>/ONNX
import sys, numpy as np, onnxruntime as ort
out_dir, onnx_dir = sys.argv[1], sys.argv[2]
W, H = 160, 96
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
img = 0.5 + 0.25 * np.sin(xx * 0.31) * np.cos(yy * 0.17)
img[(xx // 12 + yy // 12) % 2 == 0] *= 0.6          # шахматка — резкие края
img[30:34, :] = 0.95; img[:, 70:72] = 0.05          # тонкие линии
img = np.round(np.clip(img, 0, 1) * 255).astype(np.uint8)
img.tofile(f"{out_dir}/input_{W}x{H}.u8")
for name in ["ArtCNN_C4F16_DS", "ArtCNN_C4F32_DS"]:
    s = ort.InferenceSession(f"{onnx_dir}/{name}.onnx")
    inp = s.get_inputs()[0]
    print(name, inp.name, inp.shape, inp.type, [o.shape for o in s.get_outputs()])
    x = (img.astype(np.float32) / 255.0)[None, None]
    if 'float16' in inp.type:
        x = x.astype(np.float16)
    if inp.shape[-1] == 1:  # NHWC
        x = x.transpose(0, 2, 3, 1)
    y = s.run(None, {inp.name: x})[0].astype(np.float32)
    y = y.reshape(y.shape[-2], y.shape[-1]) if y.shape[1] == 1 else y[0, :, :, 0]
    np.clip(y, 0, 1).astype(np.float32).tofile(f"{out_dir}/{name}.f32")
    print(name, y.shape, float(y.min()), float(y.max()))
