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

Запуск (из корня репозитория):
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
    # Ядро любое: у M/VL последние слои 1×1, у L — 3×3.
    assert [l.save for l in last.layers] == sorted(set(d2s.channels), key=d2s.channels.index), \
      "последняя стадия должна быть ровно слоями для Depth-to-Space"
    last.final = True
  else:
    assert len(last.layers) == 1 and last.layers[0].residual, "нет ни Depth-to-Space, ни residual"
    last.final = True
  return stages


# Параметры ядра (Э5), переопределяются окружением для замеров. По умолчанию —
# то, что быстрее на Apple M1 Pro (журнал в PLAN.md): поток на пиксель, без
# плитки. На M1 Pro блок 2×1 и больше — в 4–5 раз медленнее (регистры
# вытесняются в память), плитка — на 17% медленнее.
#   CNN_BX, CNN_BY — пикселей на поток по x/y (блокировка по регистрам:
#                    каждый вес подгружается один раз на BX·BY пикселей);
#   CNN_TILE=1     — плитка входа в shared memory для 3×3.
WG = 8
BX = int(os.environ.get("CNN_BX", "1"))
BY = int(os.environ.get("CNN_BY", "1"))
TILE = os.environ.get("CNN_TILE", "0") == "1"



def wgsl_stage(stage: Stage, d2s: DepthToSpace | None, scale: int) -> str:
  out = []
  emit = out.append
  emit(f"// {stage.layers[0].desc}")
  emit(f"// Слои: {', '.join(l.save for l in stage.layers)}. Сгенерировано conversion/cnn.py — не править.")
  emit("// Точность — псевдонимы T4/M4/A4, их объявляет helpers/CNN (f32 или f16).")
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

  kernel = stage.layers[0].kernel
  halo = 1 if kernel == 3 else 0
  tiled = TILE and kernel == 3
  pixels = [(kx, ky) for ky in range(BY) for kx in range(BX)]
  # Окрестность блока в локальных координатах (от левого верхнего пикселя блока).
  xs = range(-halo, BX + halo)
  ys = range(-halo, BY + halo)
  tw, th = WG * BX + 2, WG * BY + 2  # плитка группы с ореолом

  emit("")
  if tiled:
    for i in range(len(stage.inputs)):
      emit(f"var<workgroup> tile_{i}: array<T4, {tw * th}>;")
    emit("")
  emit(f"@compute @workgroup_size({WG}, {WG})")
  emit("fn computeMain(")
  emit("  @builtin(global_invocation_id) gid: vec3u,")
  emit("  @builtin(local_invocation_id) lid: vec3u,")
  emit("  @builtin(workgroup_id) wid: vec3u,")
  emit(") {")
  emit("  let dim = vec2i(textureDimensions(tex_0));")
  emit("  let last = dim - 1;")
  emit(f"  let p0 = vec2i(gid.xy) * vec2i({BX}, {BY});")

  def var(i: int, lx: int, ly: int) -> str:
    return f"t{i}_{lx + halo}_{ly + halo}"

  if tiled:
    # Плитка входа в shared memory; края — clamp, как texOff в mpv.
    emit(f"  let origin = vec2i(wid.xy) * vec2i({WG * BX}, {WG * BY}) - 1;")
    emit(f"  for (var k = lid.y * {WG}u + lid.x; k < {tw * th}u; k += {WG * WG}u) {{")
    emit(f"    let q = clamp(origin + vec2i(i32(k % {tw}u), i32(k / {tw}u)), vec2i(0), last);")
    for i in range(len(stage.inputs)):
      emit(f"    tile_{i}[k] = T4(textureLoad(tex_{i}, q, 0));")
    emit("  }")
    emit("  workgroupBarrier();")
    emit(f"  let c = (lid.y * {BY}u + 1u) * {tw}u + lid.x * {BX}u + 1u;")

  # Загрузки: каждая текстура в каждой точке окрестности — один раз на стадию.
  for i in range(len(stage.inputs)):
    for ly in ys:
      for lx in xs:
        if tiled:
          delta = ly * tw + lx
          index = "c" if delta == 0 else (f"c + {delta}u" if delta > 0 else f"c - {-delta}u")
          emit(f"  let {var(i, lx, ly)} = tile_{i}[{index}];")
        else:
          coord = "p0" if (lx, ly) == (0, 0) else f"p0 + vec2i({lx}, {ly})"
          emit(f"  let {var(i, lx, ly)} = T4(textureLoad(tex_{i}, clamp({coord}, vec2i(0), last), 0));")

  # Вся арифметика — в T4/M4/A4: f32 или f16 целиком. Смешанный вариант
  # (произведения f16, суммы f32) на M1 Pro вдвое медленнее f32 из-за
  # приведений на каждом слагаемом, а точность f16 целиком достаточна:
  # maxΔ 0.81/255, PSNR 70 дБ против f32.
  for j, layer in enumerate(stage.layers):
    emit(f"  // {layer.save}")
    for k in range(len(pixels)):
      emit(f"  var r{j}_{k} = A4(0.0);")
    for index, dx, dy, weights in layer.terms:
      src = layer.sources[index]
      i = stage.inputs.index(src.texture)
      emit("  {")
      emit(f"    let w = M4({weights});")
      for k, (kx, ky) in enumerate(pixels):
        v = var(i, kx + dx, ky + dy)
        if src.sign == "+":
          v = f"max({v}, T4(0.0))"
        elif src.sign == "-":
          v = f"max(-{v}, T4(0.0))"
        emit(f"    r{j}_{k} += A4(w * {v});")
      emit("  }")
    for k in range(len(pixels)):
      emit(f"  r{j}_{k} += A4({layer.bias});")
      emit(f"  let s{j}_{k} = vec4f(r{j}_{k});")

  layer_index = {l.save: j for j, l in enumerate(stage.layers)}
  res = "s"  # итог слоя в f32
  if stage.final and d2s:
    emit("  let out_dim = vec2f(textureDimensions(tex_out));")
  for k, (kx, ky) in enumerate(pixels):
    emit(f"  let q{k} = p0 + vec2i({kx}, {ky});")
    emit(f"  if (all(q{k} < dim)) {{")
    if not stage.final:
      for j in range(len(stage.layers)):
        emit(f"    textureStore(out_{j}, q{k}, {res}{j}_{k});")
    elif d2s:
      # Depth-to-Space: канал (sy*2+sx) → пиксель выхода (2q + (sx, sy)),
      # плюс билинейный исходник (Overlay).
      for sy in range(scale):
        for sx in range(scale):
          ch = sy * scale + sx
          comps = ", ".join(f"{res}{layer_index[t]}_{k}[{ch}]" for t in d2s.channels)
          emit("    {")
          emit(f"      let o = q{k} * {scale} + vec2i({sx}, {sy});")
          emit("      let base = textureSampleLevel(tex_main, main_sampler, (vec2f(o) + 0.5) / out_dim, 0.0);")
          emit(f"      textureStore(tex_out, o, clamp(base + vec4f({comps}), vec4f(0.0), vec4f(1.0)));")
          emit("    }")
    else:
      emit(f"    let base = textureLoad(tex_main, q{k}, 0);")
      emit(f"    textureStore(tex_out, q{k}, clamp(base + {res}0_{k}, vec4f(0.0), vec4f(1.0)));")
    emit("  }")
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

  # Пути считаются от корня репозитория: запускать из него.
  src = os.path.relpath(glsl).replace(os.sep, "/")
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
  block: [{BX}, {BY}],
  textures: {len(ids)},
  stages: [
{stage_lines}
  ],
}};

export default model;
""")
  with open(os.path.join(out_dir, "index.ts"), "w", encoding="utf-8") as f:
    f.write(f"""// Сгенерировано conversion/cnn.py — не править руками.
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
  print(f"{class_name}: {len(convs)} слоёв → {len(stages)} стадий ({out_dir})")


if __name__ == "__main__":
  main()
