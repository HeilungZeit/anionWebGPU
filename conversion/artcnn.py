"""Генератор WGSL для ArtCNN (Artoriuz, MIT) из GLSL-шейдеров mpv в стиле compute.

Формат источника (//!COMPUTE, shared memory, макросы V4/M4/F) conversion/cnn.py
не понимает: каналы упакованы в соседние тексели текстуры большего размера,
а свёртка записана как сумма V4·скаляр (первый слой) или M4·V4.

Здесь сеть разбирается в модель и выпускается для helpers/CNN:
  * группа из 4 каналов — отдельная текстура rgba16float в разрешении входа;
  * стадия пишет не больше 4 текстур (лимит WebGPU по умолчанию), слой C4F32
    (8 групп) делится на две стадии;
  * последний слой + Depth-to-Space + сборка цвета — финальная стадия, если
    влезает в 16 текстур, иначе Depth-to-Space отдельной стадией.

Сеть работает по яркости Y (BT.709). Цвет на выходе:
bilinear(RGB) + (Y_сети − Y(bilinear(RGB))) — у BT.709 обратная матрица даёт 1
в каждой строке, так что это замена Y при билинейной цветности.
Паддинг — нули, как при обучении (Keras 'same').

Запуск (из корня репозитория):
  python3 conversion/artcnn.py <glsl> <out_dir> <ClassName>
"""
import os
import re
import sys
from dataclasses import dataclass, field

FLOATS = r"[-0-9.eE+,\s]+"
LUMA = (0.2126, 0.7152, 0.0722)


@dataclass
class Layer:
  desc: str
  save: str | None
  binds: list[str]
  pack: tuple[int, int]  # упаковка выхода: тексели на пиксель (x, y)
  scalar_input: bool = False
  # inp[k] = сумма по bind'ам texel (ox, oy) — список (bind, ox, oy)
  inputs: dict[int, list[tuple[str, int, int]]] = field(default_factory=dict)
  bias: dict[int, str] = field(default_factory=dict)
  # result → [(kind 'V'|'M', group, dx, dy, weights)]
  terms: dict[int, list[tuple[str, int, int, int, str]]] = field(default_factory=dict)
  stores: dict[int, tuple[int, int]] = field(default_factory=dict)  # result → (ox, oy)
  relu: bool = False


def parse(path: str) -> list[Layer]:
  text = open(path, encoding="utf-8").read()
  layers = []
  for block in text.split("//!DESC ")[1:]:
    desc = block.splitlines()[0].strip()
    binds = re.findall(r"^//!BIND (\S+)", block, re.M)
    save = re.search(r"^//!SAVE (\S+)", block, re.M)
    width = float(re.search(r"^//!WIDTH \S+ ([0-9.]+) \*", block, re.M).group(1))
    height = float(re.search(r"^//!HEIGHT \S+ ([0-9.]+) \*", block, re.M).group(1))
    layer = Layer(desc, save.group(1) if save else None, binds, (int(width), int(height)))
    if "Depth-To-Space" in desc:
      layers.append(layer)
      continue
    layer.scalar_input = "shared F inp" in block
    for m in re.finditer(r"inp\[(\d+)\]\[y\]\[x\] = (?:V4|F)\((.*)\);", block):
      parts = re.findall(r"(\w+)_mul \* texelFetch\(\w+_raw, input_base \+ ivec2\((\d+), (\d+)\), 0\)", m.group(2))
      layer.inputs[int(m.group(1))] = [(b, int(x), int(y)) for b, x, y in parts]
    for m in re.finditer(rf"V4 result(\d+) = V4\(({FLOATS})\);", block):
      layer.bias[int(m.group(1))] = m.group(2).strip()
    for m in re.finditer(rf"result(\d+) \+= (V4|M4)\(({FLOATS})\) \* inp_(\d+)_(\d+)_(\d+);", block):
      r, kind, w, g, x, y = m.groups()
      layer.terms.setdefault(int(r), []).append((kind[0], int(g), int(x) - 1, int(y) - 1, w.strip()))
    for m in re.finditer(r"imageStore\(out_image, output_base \+ ivec2\((\d+), (\d+)\), (max\()?result(\d+)", block):
      layer.stores[int(m.group(4))] = (int(m.group(1)), int(m.group(2)))
      layer.relu = layer.relu or bool(m.group(3))
    assert layer.terms and layer.stores, desc
    layers.append(layer)
  return layers


def group_of(pack: tuple[int, int], ox: int, oy: int) -> int:
  return oy * pack[0] + ox


@dataclass
class Stage:
  layer: Layer
  results: list[int]          # какие result'ы слоя считает стадия
  inputs: list[str]           # имена текстур-входов: "MAIN" или "<save>#<group>"
  outputs: list[str]
  final: bool = False         # + Depth-to-Space и сборка цвета
  d2s_only: bool = False      # только Depth-to-Space и сборка цвета


def plan(layers: list[Layer]) -> list[Stage]:
  packs = {l.save: l.pack for l in layers if l.save}
  stages = []
  convs = [l for l in layers if "Depth-To-Space" not in l.desc]
  for n, layer in enumerate(convs):
    if layer.scalar_input:
      inputs = ["MAIN"]
    else:
      inputs = []
      for k in sorted(layer.inputs):
        for bind, ox, oy in layer.inputs[k]:
          name = f"{bind}#{group_of(packs[bind], ox, oy)}"
          if name not in inputs:
            inputs.append(name)
    results = sorted(layer.stores)
    last = n == len(convs) - 1
    if last:
      assert len(results) == 1, "последний слой — 4 канала для Depth-to-Space"
      if len(inputs) + 1 <= 16:
        stages.append(Stage(layer, results, inputs, [], final=True))
        continue
      stages.append(Stage(layer, results, inputs, [f"{layer.save}#0"]))
      stages.append(Stage(layer, [], [f"{layer.save}#0"], [], final=True, d2s_only=True))
      continue
    for i in range(0, len(results), 4):
      chunk = results[i:i + 4]
      outs = [f"{layer.save}#{group_of(layer.pack, *layer.stores[r])}" for r in chunk]
      stages.append(Stage(layer, chunk, inputs, outs))
  return stages


def emit_d2s(emit, value: str) -> None:
  """Depth-to-Space яркости (канал sy*2+sx) и сборка цвета для 4 пикселей выхода."""
  emit("  let out_dim = vec2f(textureDimensions(tex_out));")
  for sy in range(2):
    for sx in range(2):
      emit("  {")
      emit(f"    let o = p * 2 + vec2i({sx}, {sy});")
      emit("    let base = textureSampleLevel(tex_main, main_sampler, (vec2f(o) + 0.5) / out_dim, 0.0);")
      emit(f"    let y = clamp({value}[{sy * 2 + sx}], 0.0, 1.0);")
      emit(f"    let rgb = base.rgb + (y - dot(base.rgb, vec3f{LUMA}));")
      emit("    textureStore(tex_out, o, vec4f(clamp(rgb, vec3f(0.0), vec3f(1.0)), 1.0));")
      emit("  }")


def wgsl_stage(stage: Stage, packs: dict[str, tuple[int, int]]) -> str:
  out = []
  emit = out.append
  layer = stage.layer
  emit(f"// {layer.desc}{' + Depth-To-Space' if stage.final else ''}")
  emit("// Сгенерировано conversion/artcnn.py — не править.")
  emit("// Точность — псевдонимы T4/M4/A4/S1, их объявляет helpers/CNN (f32 или f16).")
  for i, name in enumerate(stage.inputs):
    emit(f"@group(0) @binding({i}) var tex_{i}: texture_2d<f32>; // {name}")
  b = len(stage.inputs)
  if stage.final:
    emit(f"@group(0) @binding({b}) var tex_main: texture_2d<f32>; // RGB")
    emit(f"@group(0) @binding({b + 1}) var main_sampler: sampler;")
    emit(f"@group(0) @binding({b + 2}) var tex_out: texture_storage_2d<rgba16float, write>;")
  else:
    for j, name in enumerate(stage.outputs):
      emit(f"@group(0) @binding({b + j}) var out_{j}: texture_storage_2d<rgba16float, write>; // {name}")
  emit("")
  emit("// Нули за краем кадра — как паддинг 'same' при обучении.")
  emit("fn fetch(t: texture_2d<f32>, p: vec2i) -> vec4f {")
  emit("  if (any(p < vec2i(0)) || any(p >= vec2i(textureDimensions(t)))) {")
  emit("    return vec4f(0.0);")
  emit("  }")
  emit("  return textureLoad(t, p, 0);")
  emit("}")
  emit("")
  emit("@compute @workgroup_size(8, 8)")
  emit("fn computeMain(@builtin(global_invocation_id) gid: vec3u) {")
  emit("  let dim = vec2i(textureDimensions(tex_0));")
  emit("  let p = vec2i(gid.xy);")
  emit("  if (any(p >= dim)) {")
  emit("    return;")
  emit("  }")

  if stage.d2s_only:
    emit("  let v = textureLoad(tex_0, p, 0);")
    emit_d2s(emit, "v")
    emit("}")
    return "\n".join(out) + "\n"

  # Порядок «по входу»: точка окрестности загружается и сразу добавляется во
  # все выходы стадии. Веса — константы в коде: вариант с uniform-буфером и
  # неразвёрнутым циклом на M1 Pro медленнее (C4F16 f32: 61 мс против 33).
  for r in stage.results:
    emit(f"  var r{r} = A4({layer.bias[r]});")
  weights = {}
  for r in stage.results:
    for kind, g, dx, dy, w in layer.terms[r]:
      weights[(g, dx, dy, r)] = (kind, w)
  for g in sorted(layer.inputs):
    for dy in (-1, 0, 1):
      for dx in (-1, 0, 1):
        q = "p" if (dx, dy) == (0, 0) else f"p + vec2i({dx}, {dy})"
        if layer.scalar_input:
          value = f"S1(dot(fetch(tex_0, {q}).rgb, vec3f{LUMA}))"
        else:
          srcs = [f"fetch(tex_{stage.inputs.index(f'{bind}#{group_of(packs[bind], ox, oy)}')}, {q})"
                  for bind, ox, oy in layer.inputs[g]]
          value = f"T4({' + '.join(srcs)})"
        emit("  {")
        emit(f"    let v = {value};")
        for r in stage.results:
          kind, w = weights[(g, dx, dy, r)]
          emit(f"    r{r} += A4({'T4' if kind == 'V' else 'M4'}({w}) * v);")
        emit("  }")
  if layer.relu:
    for r in stage.results:
      emit(f"  r{r} = max(r{r}, A4(0.0));")

  if stage.final:
    emit_d2s(emit, f"vec4f(r{stage.results[0]})")
  else:
    for j, r in enumerate(stage.results):
      emit(f"  textureStore(out_{j}, p, vec4f(r{r}));")
  emit("}")
  return "\n".join(out) + "\n"


def main() -> None:
  if len(sys.argv) != 4:
    print(__doc__)
    sys.exit(1)
  glsl, out_dir, class_name = sys.argv[1:]
  layers = parse(glsl)
  stages = plan(layers)
  packs = {l.save: l.pack for l in layers if l.save}

  shader_dir = os.path.join(out_dir, "shaders")
  os.makedirs(shader_dir, exist_ok=True)
  for name in os.listdir(shader_dir):
    os.remove(os.path.join(shader_dir, name))

  ids = {"MAIN": 0}
  graph = []
  for n, stage in enumerate(stages):
    with open(os.path.join(shader_dir, f"stage{n}.wgsl"), "w", encoding="utf-8") as f:
      f.write(wgsl_stage(stage, packs))
    for name in stage.outputs:
      ids.setdefault(name, len(ids))
    graph.append((n, [ids[t] for t in stage.inputs], [ids[t] for t in stage.outputs], stage.final))

  src = os.path.relpath(glsl).replace(os.sep, "/")
  rel_helpers = os.path.relpath("src/pipelines/helpers/CNN", out_dir).replace(os.sep, "/")
  rel_interfaces = os.path.relpath("src/pipelines/interfaces", out_dir).replace(os.sep, "/")
  imports = "\n".join(f"import stage{n} from './shaders/stage{n}.wgsl';" for n, *_ in graph)
  stage_lines = "\n".join(
    f"    {{\n      wgsl: stage{n},\n      inputs: [{', '.join(map(str, ins))}],\n"
    f"      outputs: [{', '.join(map(str, outs))}],\n" + ("      final: true,\n" if final else "") + "    },"
    for n, ins, outs, final in graph)
  with open(os.path.join(out_dir, "model.ts"), "w", encoding="utf-8") as f:
    f.write(f"""// Сгенерировано conversion/artcnn.py — не править руками.
// Источник: {src}
// {len(layers)} шейдеров mpv → {len(stages)} стадий.
{imports}
import {{ CNNModel }} from '{rel_helpers}/model';

const model: CNNModel = {{
  scale: 2,
  block: [1, 1],
  textures: {len(ids)},
  stages: [
{stage_lines}
  ],
}};

export default model;
""")
  with open(os.path.join(out_dir, "index.ts"), "w", encoding="utf-8") as f:
    f.write(f"""// Сгенерировано conversion/artcnn.py — не править руками.
// ArtCNN (Artoriuz, MIT): увеличение ×2 по яркости, цветность — билинейно.
import {{ CNNModelPipelineDescriptor }} from '{rel_interfaces}';
import {{ CNN }} from '{rel_helpers}';
import model from './model';

export class {class_name} extends CNN {{
  constructor({{ device, inputTexture, precision }}: CNNModelPipelineDescriptor) {{
    super({{
      device, inputTexture, model, name: '{class_name}', precision,
    }});
  }}
}}
""")
  print(f"{class_name}: {len(layers)} шейдеров mpv → {len(stages)} стадий ({out_dir})")


if __name__ == "__main__":
  main()
