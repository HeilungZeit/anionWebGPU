// Ворота повторов, шаг 1: максимальная разница нового кадра с последним
// посчитанным, в уровнях 8 бит (|Δ| по R, G, B), и сумма знаковых разниц по
// каналам — средний сдвиг кадра (затемнение). Сначала по группе в
// workgroup-памяти, затем по одному atomic на группу в буфер.
@group(0) @binding(0) var tex_cur: texture_2d<f32>; // новый кадр
@group(0) @binding(1) var tex_prev: texture_2d<f32>; // последний посчитанный
// Раскладка state — в decide.wgsl: 0 — max |Δ|, 8–10 — суммы Δ по каналам.
struct State {
  max_diff: atomic<u32>,
  other: array<u32, 7>,
  sum_diff: array<atomic<i32>, 3>,
}
@group(0) @binding(2) var<storage, read_write> state: State;

var<workgroup> group_max: atomic<u32>;
var<workgroup> group_sum: array<atomic<i32>, 3>;

@compute
@workgroup_size(8, 8)
fn computeMain(
  @builtin(global_invocation_id) pixel: vec3u,
  @builtin(local_invocation_index) index: u32,
) {
  let dim: vec2u = textureDimensions(tex_cur);
  // Без return до барьера: workgroupBarrier должен быть в однородном потоке.
  if (pixel.x < dim.x && pixel.y < dim.y) {
    let cur = round(textureLoad(tex_cur, pixel.xy, 0).rgb * 255.0);
    let prev = round(textureLoad(tex_prev, pixel.xy, 0).rgb * 255.0);
    let s = vec3i(cur - prev);
    let d = abs(s);
    atomicMax(&group_max, u32(max(d.r, max(d.g, d.b))));
    atomicAdd(&group_sum[0], s.r);
    atomicAdd(&group_sum[1], s.g);
    atomicAdd(&group_sum[2], s.b);
  }
  workgroupBarrier();
  if (index == 0u) {
    atomicMax(&state.max_diff, atomicLoad(&group_max));
    for (var c = 0u; c < 3u; c++) {
      atomicAdd(&state.sum_diff[c], atomicLoad(&group_sum[c]));
    }
  }
}
