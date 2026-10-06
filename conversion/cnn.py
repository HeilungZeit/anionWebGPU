"""Генератор WGSL для CNN-моделей Anime4K (v3.2/v4.0) со слиянием проходов.

Читает GLSL-шейдер mpv, строит модель сети и выпускает:
  <out>/shaders/stageN.wgsl — по шейдеру на стадию;
  <out>/model.ts            — граф стадий для helpers/CNN;
  <out>/index.ts            — класс пайплайна.

Отличие от conversion/shader.py (шейдер на слой, как в mpv):
  * слои с одинаковыми входами (пары tf/tf1) — одна стадия с несколькими
    выходами: входы читаются один раз;
  * последние слои 1×1 + Depth-to-Space + сложение с исходником — одна
    стадия: поток считает все каналы своего пикселя и пишет сразу 2×2 пикселя
    выхода (bilinear(исходник) + residual).
Математика та же, что в mpv; порядок сложения внутри слоя сохранён.

Запуск (из anionWebGPU/):
  python3 conversion/cnn.py <glsl> <out_dir> <ClassName>
Все модели anion — conversion/generate.sh.
"""
import os
import re
import sys
from dataclasses import dataclass, field

FLOAT_LIST = r"[-0-9.eE+,\s]+"


@dataclass
class Source:
  """Вход слоя go_N / g_N: текстура и знак CReLU (None — без max)."""
  texture: str
  sign: str | None  # '+', '-' или None (сырое значение, первый слой)


@dataclass
class Conv:
  desc: str
  binds: list[str]
  save: str
  kernel: int = 1  # 3 — 3×3 (go_N с offset), 1 — 1×1 (g_N)
  sources: dict[int, Source] = field(default_factory=dict)
  # (номер source, dx, dy, текст 16 весов) в порядке исходника
  terms: list[tuple[int, int, int, str]] = field(default_factory=list)
  bias: str = "0.0, 0.0, 0.0, 0.0"
  residual: bool = False  # return result + MAIN_tex(MAIN_pos)


@dataclass
class DepthToSpace:
  # для c0..c3: имя текстуры последнего слоя
  channels: list[str]


def parse(path: str) -> tuple[list[Conv], DepthToSpace | None]:
  text = open(path, encoding="utf-8").read()
  blocks = text.split("//!DESC ")[1:]
  convs: list[Conv] = []
  d2s = None
  for block in blocks:
    lines = block.splitlines()
    desc = lines[0].strip()
    binds = re.findall(r"^//!BIND (\S+)", block, re.M)
    save = re.search(r"^//!SAVE (\S+)", block, re.M).group(1)

    if "Depth-to-Space" in desc:
      direct = dict(re.findall(r"float (c\d) = (\w+)_tex\(", block))
      alias = dict(re.findall(r"float (c\d) = (c\d);", block))
      channels = []
      for i in range(4):
        name = f"c{i}"
        while name in alias:
          name = alias[name]
        channels.append(direct[name])
      d2s = DepthToSpace(channels)
      continue

    assert "-Conv-" in desc, desc
    conv = Conv(desc, binds, save)
    for m in re.finditer(r"^#define go?_(\d+)(\(x_off, y_off\))? \((.*)\)$", block, re.M):
      index, offset, body = int(m.group(1)), m.group(2), m.group(3)
      conv.kernel = 3 if offset else 1
      tex = re.search(r"(\w+?)_tex(?:Off)?\(", body).group(1)
      sign = None
      if body.startswith("max(-("):
        sign = "-"
      elif body.startswith("max(("):
        sign = "+"
      conv.sources[index] = Source(tex, sign)
    for m in re.finditer(rf"mat4\(({FLOAT_LIST})\) \* go?_(\d+)(?:\(({FLOAT_LIST})\))?", block):
      weights, index, offs = m.group(1), int(m.group(2)), m.group(3)
      dx, dy = (int(float(v)) for v in offs.split(",")) if offs else (0, 0)
      conv.terms.append((index, dx, dy, weights.strip()))
    bias = re.search(rf"result \+= vec4\(({FLOAT_LIST})\);", block)
    if bias:
      conv.bias = bias.group(1).strip()
    conv.residual = "+ MAIN_tex(MAIN_pos)" in block
    convs.append(conv)
  return convs, d2s


@dataclass
class Stage:
  layers: list[Conv]
  inputs: list[str]  # имена текстур-входов в порядке привязки
  final: bool = False


def plan(convs: list[Conv], d2s: DepthToSpace | None) -> list[Stage]:
  """Слои с одинаковыми входами и ядром — одна стадия (не больше 4 выходов)."""
  stages: list[Stage] = []
  for conv in convs:
    inputs = [t for t in conv.binds if any(s.texture == t for s in conv.sources.values())]
    prev = stages[-1] if stages else None
    if (prev and prev.inputs == inputs and prev.layers[0].kernel == conv.kernel
        and len(prev.layers) < 4 and not conv.residual):
      prev.layers.append(conv)
    else:
      stages.append(Stage([conv], inputs))
  last = stages[-1]
  if d2s:
    assert last.layers[0].kernel == 1 and [l.save for l in last.layers] == sorted(set(d2s.channels), key=d2s.channels.index), \
      "последняя стадия должна быть ровно слоями для Depth-to-Space"
    last.final = True
  else:
    assert len(last.layers) == 1 and last.layers[0].residual, "нет ни Depth-to-Space, ни residual"
    last.final = True
  return stages


def wgsl_stage(stage: Stage, d2s: DepthToSpace | None, scale: int) -> str:
  out = []
  emit = out.append
  emit(f"// {stage.layers[0].desc}")
  emit(f"// Слои: {', '.join(l.save for l in stage.layers)}. Сгенерировано conversion/cnn.py — не править.")
  binding = 0
  for i, name in enumerate(stage.inputs):
    emit(f"@group(0) @binding({binding}) var tex_{i}: texture_2d<f32>; // {name}")
    binding += 1
  if stage.final:
    emit(f"@group(0) @binding({binding}) var tex_main: texture_2d<f32>; // MAIN")
    emit(f"@group(0) @binding({binding + 1}) var main_sampler: sampler;")
    emit(f"@group(0) @binding({binding + 2}) var tex_out: texture_storage_2d<rgba16float, write>;")
  else:
    for j, layer in enumerate(stage.layers):
      emit(f"@group(0) @binding({binding + j}) var out_{j}: texture_storage_2d<rgba16float, write>; // {layer.save}")
  emit("")
  emit("@compute @workgroup_size(8, 8)")
  emit("fn computeMain(@builtin(global_invocation_id) gid: vec3u) {")
  emit("  let dim = textureDimensions(tex_0);")
  emit("  if (gid.x >= dim.x || gid.y >= dim.y) {")
  emit("    return;")
  emit("  }")
  emit("  let p = vec2i(gid.xy);")

  # Загрузки: каждая текстура в каждом смещении — один раз на все слои стадии.
  kernel = stage.layers[0].kernel
  offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)] if kernel == 3 else [(0, 0)]
  if kernel == 3:
    emit("  let last = vec2i(dim) - 1;")

  def var(i: int, dx: int, dy: int) -> str:
    def part(v):
      return "m1" if v < 0 else ("p1" if v > 0 else "0")
    return f"t{i}" if kernel == 1 else f"t{i}_{part(dx)}_{part(dy)}"

  for i in range(len(stage.inputs)):
    for dx, dy in offsets:
      coord = "p" if (dx, dy) == (0, 0) else f"p + vec2i({dx}, {dy})"
      if kernel == 3:
        coord = f"clamp({coord}, vec2i(0), last)"  # как texOff в mpv (clamp to edge)
      emit(f"  let {var(i, dx, dy)} = textureLoad(tex_{i}, {coord}, 0);")

  for j, layer in enumerate(stage.layers):
    emit(f"  // {layer.save}")
    first = True
    for index, dx, dy, weights in layer.terms:
      src = layer.sources[index]
      v = var(stage.inputs.index(src.texture), dx, dy)
      if src.sign == "+":
        v = f"max({v}, vec4f(0.0))"
      elif src.sign == "-":
        v = f"max(-{v}, vec4f(0.0))"
      expr = f"mat4x4f({weights}) * {v}"
      emit(f"  var r{j} = {expr};" if first else f"  r{j} += {expr};")
      first = False
    emit(f"  r{j} += vec4f({layer.bias});")

  if not stage.final:
    for j in range(len(stage.layers)):
      emit(f"  textureStore(out_{j}, gid.xy, r{j});")
  elif d2s:
    # Depth-to-Space: канал (sy*2+sx) слоя → пиксель (2p + (sx, sy)).
    layer_index = {l.save: j for j, l in enumerate(stage.layers)}
    emit("  let out_dim = vec2f(textureDimensions(tex_out));")
    for sy in range(scale):
      for sx in range(scale):
        ch = sy * scale + sx
        comps = ", ".join(f"r{layer_index[t]}[{ch}]" for t in d2s.channels)
        emit(f"  {{")
        emit(f"    let q = gid.xy * {scale}u + vec2u({sx}u, {sy}u);")
        emit(f"    let uv = (vec2f(q) + 0.5) / out_dim;")
        emit(f"    let base = textureSampleLevel(tex_main, main_sampler, uv, 0.0);")
        emit(f"    textureStore(tex_out, q, clamp(base + vec4f({comps}), vec4f(0.0), vec4f(1.0)));")
        emit(f"  }}")
  else:
    emit("  let base = textureLoad(tex_main, p, 0);")
    emit("  textureStore(tex_out, gid.xy, clamp(base + r0, vec4f(0.0), vec4f(1.0)));")
  emit("}")
  return "\n".join(out) + "\n"


def main() -> None:
  if len(sys.argv) != 4:
    print(__doc__)
    sys.exit(1)
  glsl, out_dir, class_name = sys.argv[1:]
  convs, d2s = parse(glsl)
  stages = plan(convs, d2s)
  scale = 2 if d2s else 1

  shader_dir = os.path.join(out_dir, "shaders")
  os.makedirs(shader_dir, exist_ok=True)
  for name in os.listdir(shader_dir):
    os.remove(os.path.join(shader_dir, name))

  # Текстуры: 0 — вход (MAIN), дальше выходы стадий по порядку.
  ids = {"MAIN": 0}
  graph = []
  for n, stage in enumerate(stages):
    with open(os.path.join(shader_dir, f"stage{n}.wgsl"), "w", encoding="utf-8") as f:
      f.write(wgsl_stage(stage, d2s, scale))
    outputs = []
    if not stage.final:
      for layer in stage.layers:
        ids[layer.save] = len(ids)
        outputs.append(ids[layer.save])
    graph.append((n, [ids[t] for t in stage.inputs], outputs, stage.final))

  # Пути считаются от корня форка: запускать из anionWebGPU/.
  src = os.path.relpath(glsl, "..").replace(os.sep, "/")
  rel_helpers = os.path.relpath("src/pipelines/helpers/CNN", out_dir).replace(os.sep, "/")
  rel_interfaces = os.path.relpath("src/pipelines/interfaces", out_dir).replace(os.sep, "/")
  imports = "\n".join(f"import stage{n} from './shaders/stage{n}.wgsl';" for n, *_ in graph)
  stage_lines = "\n".join(
    f"    {{\n      wgsl: stage{n},\n      inputs: [{', '.join(map(str, ins))}],\n"
    f"      outputs: [{', '.join(map(str, outs))}],\n" + ("      final: true,\n" if final else "") + "    },"
    for n, ins, outs, final in graph)
  with open(os.path.join(out_dir, "model.ts"), "w", encoding="utf-8") as f:
    f.write(f"""// Сгенерировано conversion/cnn.py — не править руками.
// Источник: {src}
// {len(convs)} слоёв mpv → {len(stages)} стадий.
{imports}
import {{ CNNModel }} from '{rel_helpers}/model';

const model: CNNModel = {{
  scale: {scale},
  textures: {len(ids)},
  stages: [
{stage_lines}
  ],
}};

export default model;
""")
  with open(os.path.join(out_dir, "index.ts"), "w", encoding="utf-8") as f:
    f.write(f"""// Сгенерировано conversion/cnn.py — не править руками.
import {{ Anime4KPipelineDescriptor }} from '{rel_interfaces}';
import {{ CNN }} from '{rel_helpers}';
import model from './model';

export class {class_name} extends CNN {{
  constructor({{ device, inputTexture }}: Anime4KPipelineDescriptor) {{
    super({{
      device, inputTexture, model, name: '{class_name}',
    }});
  }}
}}
""")
  print(f"{class_name}: {len(convs)} слоёв → {len(stages)} стадий ({out_dir})")


if __name__ == "__main__":
  main()
