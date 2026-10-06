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
# то, что быстрее на Apple M1 Pro (журнал в docs/PLAN.md): поток на пиксель,
# без плитки и subgroups. На M1 Pro блок 2×1 и больше — в 4–5 раз медленнее
# (регистры вытесняются в память), плитка — на 17%, subgroups — на 30–55%.
#   CNN_BX, CNN_BY — пикселей на поток по x/y (блокировка по регистрам:
#                    каждый вес подгружается один раз на BX·BY пикселей);
#   CNN_TILE=1     — плитка входа в shared memory для 3×3.
WG = 8
BX = int(os.environ.get("CNN_BX", "1"))
BY = int(os.environ.get("CNN_BY", "1"))
TILE = os.environ.get("CNN_TILE", "0") == "1"
#   CNN_SUBGROUPS=1 — соседей 3×3 брать у соседних потоков (subgroupShuffle),
#                    из текстуры читать только свой пиксель и края subgroup.
SUBGROUPS = os.environ.get("CNN_SUBGROUPS", "0") == "1"
#   CNN_PACK=1      — выходы стадии парами слоёв в одну rgba32uint (8 каналов
#                    f16 через pack2x16float): чтений текстур и привязок вдвое
#                    меньше при том же объёме памяти. Только без плитки и
#                    subgroups.
PACK = os.environ.get("CNN_PACK", "0") == "1"
assert not (PACK and (TILE or SUBGROUPS)), "CNN_PACK — только без CNN_TILE и CNN_SUBGROUPS"


@dataclass
class Unit:
  """Где лежит выход слоя: номер текстуры модели и половина (None — rgba16float)."""
  texture: int
  half: int | None = None


def assign_units(stages: list[Stage]) -> tuple[dict[str, Unit], list[int]]:
  """Текстуры модели: 0 — вход (MAIN), дальше выходы стадий по порядку."""
  units = {"MAIN": Unit(0)}
  packed: list[int] = []
  count = 1
  for stage in stages:
    if stage.final:
      continue
    saves = [l.save for l in stage.layers]
    step = 2 if PACK else 1
    for k in range(0, len(saves), step):
      group = saves[k:k + step]
      if len(group) == 2:
        packed.append(count)
        units[group[0]] = Unit(count, 0)
        units[group[1]] = Unit(count, 1)
      else:
        units[group[0]] = Unit(count)
      count += 1
  return units, packed


def input_units(stage: Stage, units: dict[str, Unit]) -> list[int]:
  """Текстуры-входы стадии в порядке привязки, без повторов."""
  ids: list[int] = []
  for name in stage.inputs:
    if units[name].texture not in ids:
      ids.append(units[name].texture)
  return ids


def output_units(stage: Stage, units: dict[str, Unit]) -> list[int]:
  ids: list[int] = []
  for layer in stage.layers:
    if units[layer.save].texture not in ids:
      ids.append(units[layer.save].texture)
  return ids



def wgsl_stage(stage: Stage, d2s: DepthToSpace | None, scale: int,
               units: dict[str, Unit], packed: list[int]) -> str:
  out = []
  emit = out.append
  emit(f"// {stage.layers[0].desc}")
  emit(f"// Слои: {', '.join(l.save for l in stage.layers)}. Сгенерировано conversion/cnn.py — не править.")
  emit("// Точность — псевдонимы T4/M4/A4, их объявляет helpers/CNN (f32 или f16).")
  if stage.final:
    emit("// deRing() — эпилог Clamp Highlights, его добавляет helpers/CNN.")
  binding = 0
  ins = input_units(stage, units)
  for u, tid in enumerate(ins):
    names = ", ".join(n for n in stage.inputs if units[n].texture == tid)
    kind = "u32" if tid in packed else "f32"
    emit(f"@group(0) @binding({binding}) var tex_{u}: texture_2d<{kind}>; // {names}")
    binding += 1
  if stage.final:
    emit(f"@group(0) @binding({binding}) var tex_main: texture_2d<f32>; // MAIN")
    emit(f"@group(0) @binding({binding + 1}) var main_sampler: sampler;")
    emit(f"@group(0) @binding({binding + 2}) var tex_out: texture_storage_2d<rgba16float, write>;")
  else:
    for o, tid in enumerate(output_units(stage, units)):
      names = ", ".join(l.save for l in stage.layers if units[l.save].texture == tid)
      fmt = "rgba32uint" if tid in packed else "rgba16float"
      emit(f"@group(0) @binding({binding + o}) var out_{o}: texture_storage_2d<{fmt}, write>; // {names}")

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
  shuffled = SUBGROUPS and kernel == 3 and not tiled and BX == 1 and BY == 1
  if shuffled:
    emit("  @builtin(local_invocation_index) lidx: u32,")
    emit("  @builtin(subgroup_invocation_id) sg_lane: u32,")
    emit("  @builtin(subgroup_size) sg_size: u32,")
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

  if shuffled:
    # Свой пиксель — из текстуры, соседи — у соседних потоков группы 8×8.
    # Раскладку потоков по subgroup WebGPU не гарантирует, поэтому вместе с
    # пикселем передаётся индекс потока: пришёл не тот сосед (край subgroup
    # или группы) — читаем сами. Математика та же на любой раскладке.
    emit("  let lpos = vec2i(lid.xy);")
    for dy in (-1, 0, 1):
      for dx in (-1, 0, 1):
        if (dx, dy) == (0, 0):
          continue
        n = f"{dx + 1}{dy + 1}"
        emit(f"  let n{n} = lpos + vec2i({dx}, {dy});")
        delta = dy * WG + dx
        shift = f"+ {delta}" if delta > 0 else f"- {-delta}"
        emit(f"  let src{n} = u32(clamp(i32(sg_lane) {shift}, 0, i32(sg_size) - 1));")
        # Shuffle — до проверок: && ленивый, а subgroup-операцию обязаны
        # выполнить все потоки subgroup.
        emit(f"  let from{n} = subgroupShuffle(lidx, src{n});")
        emit(f"  let ok{n} = all(n{n} >= vec2i(0)) && all(n{n} < vec2i({WG}))"
             f" && from{n} == u32(n{n}.y * {WG} + n{n}.x);")
    for i in range(len(stage.inputs)):
      emit(f"  let c{i} = textureLoad(tex_{i}, clamp(p0, vec2i(0), last), 0);")
      for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
          if (dx, dy) == (0, 0):
            emit(f"  let {var(i, 0, 0)} = T4(c{i});")
            continue
          n = f"{dx + 1}{dy + 1}"
          emit(f"  var s{i}_{n} = subgroupShuffle(c{i}, src{n});")
          emit(f"  if (!ok{n}) {{")
          emit(f"    s{i}_{n} = textureLoad(tex_{i}, clamp(p0 + vec2i({dx}, {dy}), vec2i(0), last), 0);")
          emit("  }")
          emit(f"  let {var(i, dx, dy)} = T4(s{i}_{n});")

  # Загрузки: каждая текстура в каждой точке окрестности — один раз на стадию.
  # Упакованная (rgba32uint) читается одним textureLoad на оба слоя;
  # unpack_half() объявляет helpers/CNN под точность (f32 или f16).
  for u, tid in enumerate(ins if packed and not shuffled else []):
    if tid not in packed:
      continue
    for ly in ys:
      for lx in xs:
        coord = "p0" if (lx, ly) == (0, 0) else f"p0 + vec2i({lx}, {ly})"
        emit(f"  let u{u}_{lx + halo}_{ly + halo} = textureLoad(tex_{u}, clamp({coord}, vec2i(0), last), 0);")
  for i in range(len(stage.inputs) if not shuffled else 0):
    unit = units[stage.inputs[i]]
    u = ins.index(unit.texture)
    for ly in ys:
      for lx in xs:
        if unit.texture in packed:
          a, b = ("x", "y") if unit.half == 0 else ("z", "w")
          v = f"u{u}_{lx + halo}_{ly + halo}"
          emit(f"  let {var(i, lx, ly)} = unpack_half({v}.{a}, {v}.{b});")
        elif tiled:
          delta = ly * tw + lx
          index = "c" if delta == 0 else (f"c + {delta}u" if delta > 0 else f"c - {-delta}u")
          emit(f"  let {var(i, lx, ly)} = tile_{i}[{index}];")
        else:
          coord = "p0" if (lx, ly) == (0, 0) else f"p0 + vec2i({lx}, {ly})"
          emit(f"  let {var(i, lx, ly)} = T4(textureLoad(tex_{u}, clamp({coord}, vec2i(0), last), 0));")

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
      outs = output_units(stage, units)
      for o, tid in enumerate(outs):
        js = [j for j, l in enumerate(stage.layers) if units[l.save].texture == tid]
        if tid in packed:
          a, b = (f"{res}{j}_{k}" for j in js)
          emit(f"    textureStore(out_{o}, q{k}, vec4u(pack2x16float({a}.xy), pack2x16float({a}.zw),"
               f" pack2x16float({b}.xy), pack2x16float({b}.zw)));")
        else:
          emit(f"    textureStore(out_{o}, q{k}, {res}{js[0]}_{k});")
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
          emit(f"      textureStore(tex_out, o, deRing(clamp(base + vec4f({comps}), vec4f(0.0), vec4f(1.0)), o, out_dim));")
          emit("    }")
    else:
      emit(f"    let base = textureLoad(tex_main, q{k}, 0);")
      emit(f"    textureStore(tex_out, q{k}, deRing(clamp(base + {res}0_{k}, vec4f(0.0), vec4f(1.0)), q{k}, vec2f(dim)));")
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

  units, packed = assign_units(stages)
  textures = 1 + max(u.texture for u in units.values())
  graph = []
  for n, stage in enumerate(stages):
    with open(os.path.join(shader_dir, f"stage{n}.wgsl"), "w", encoding="utf-8") as f:
      f.write(wgsl_stage(stage, d2s, scale, units, packed))
    outputs = [] if stage.final else output_units(stage, units)
    graph.append((n, input_units(stage, units), outputs, stage.final,
                  SUBGROUPS and stage.layers[0].kernel == 3 and not TILE and BX == 1 and BY == 1))

  # Пути считаются от корня репозитория: запускать из него.
  src = os.path.relpath(glsl).replace(os.sep, "/")
  rel_helpers = os.path.relpath("src/pipelines/helpers/CNN", out_dir).replace(os.sep, "/")
  rel_interfaces = os.path.relpath("src/pipelines/interfaces", out_dir).replace(os.sep, "/")
  imports = "\n".join(f"import stage{n} from './shaders/stage{n}.wgsl';" for n, *_ in graph)
  stage_lines = "\n".join(
    f"    {{\n      wgsl: stage{n},\n      inputs: [{', '.join(map(str, ins))}],\n"
    f"      outputs: [{', '.join(map(str, outs))}],\n" + ("      final: true,\n" if final else "")
    + ("      subgroups: true,\n" if sg else "") + "    },"
    for n, ins, outs, final, sg in graph)
  with open(os.path.join(out_dir, "model.ts"), "w", encoding="utf-8") as f:
    f.write(f"""// Сгенерировано conversion/cnn.py — не править руками.
// Источник: {src}
// {len(convs)} слоёв mpv → {len(stages)} стадий.
{imports}
import {{ CNNModel }} from '{rel_helpers}/model';

const model: CNNModel = {{
  scale: {scale},
  block: [{BX}, {BY}],
  textures: {textures},
{f"  packed: [{', '.join(map(str, packed))}],{chr(10)}" if packed else ""}  stages: [
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
  constructor({{
    device, inputTexture, precision, deRing, gate,
  }}: CNNModelPipelineDescriptor) {{
    super({{
      device, inputTexture, model, name: '{class_name}', precision, deRing, gate,
    }});
  }}
}}
""")
  print(f"{class_name}: {len(convs)} слоёв → {len(stages)} стадий ({out_dir})")


if __name__ == "__main__":
  main()
