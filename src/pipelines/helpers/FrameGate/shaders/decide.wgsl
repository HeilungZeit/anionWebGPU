// Ворота повторов, шаг 2: один поток решает, считать ли кадр, и пишет
// аргументы косвенного запуска всех звеньев: настоящие из шаблона или нули.
// Звенья цепочки запускаются dispatchWorkgroupsIndirect из этого буфера —
// на повторе GPU сразу пропускает их, без чтения результата на CPU.
//
// state: 0 — max |Δ| кадра (пишет diff), 1 — кадров, 2 — пропущено,
// 3 — порог, 4 — число слотов, 5 — принудительный пересчёт, 6 — max |Δ|
// последнего кадра (для стенда), 8–10 — суммы знаковых Δ по R, G, B (пишет
// diff, i32), 11 — порог суммы (средний сдвиг × число пикселей).
@group(0) @binding(0) var<storage, read_write> state: array<u32, 16>;
@group(0) @binding(1) var<storage, read> template_args: array<u32>;
@group(0) @binding(2) var<storage, read_write> args: array<u32>;

@compute
@workgroup_size(1)
fn computeMain() {
  // Шум сжатия даёт разницу до ~16 уровней, но в среднем по кадру — около
  // нуля; затемнение — малую разницу, но сдвиг среднего.
  let sums = abs(vec3i(bitcast<i32>(state[8]), bitcast<i32>(state[9]), bitcast<i32>(state[10])));
  let shifted = u32(max(sums.x, max(sums.y, sums.z))) > state[11];
  let changed = state[0] > state[3] || shifted || state[5] != 0u;
  let count = state[4] * 3u;
  for (var i: u32 = 0u; i < count; i = i + 1u) {
    args[i] = select(0u, template_args[i], changed);
  }
  state[1] = state[1] + 1u;
  state[2] = state[2] + select(1u, 0u, changed);
  state[5] = 0u;
  state[6] = state[0];
}
