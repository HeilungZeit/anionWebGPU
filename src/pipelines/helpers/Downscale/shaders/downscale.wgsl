// Уменьшение с антиалиасингом: ядро Catmull-Rom (B=0, C=0.5), растянутое на
// коэффициент уменьшения, — каждый пиксель выхода усредняет все пиксели
// входа под своим следом. Билинейная выборка в точке (FILTER = 0, как
// AutoDownscalePre в mpv) при 1440→1080 пропускает часть пикселей и даёт
// алиасинг на тонких линиях.
override FILTER: u32 = 1u; // 0 — билинейно, 1 — Catmull-Rom
override AXIS: u32 = 0u; // для Catmull-Rom: 0 — проход по x, 1 — по y

@group(0) @binding(0) var tex_in: texture_2d<f32>;
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var tex_out: texture_storage_2d<rgba16float, write>;

fn catmullRom(x: f32) -> f32 {
  let a = abs(x);
  if (a < 1.0) {
    return 1.5 * a * a * a - 2.5 * a * a + 1.0;
  }
  if (a < 2.0) {
    return -0.5 * a * a * a + 2.5 * a * a - 4.0 * a + 2.0;
  }
  return 0.0;
}

@compute @workgroup_size(8, 8)
fn computeMain(@builtin(global_invocation_id) gid: vec3u) {
  let dim_out = textureDimensions(tex_out);
  if (gid.x >= dim_out.x || gid.y >= dim_out.y) {
    return;
  }
  let dim_in = vec2f(textureDimensions(tex_in));
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(dim_out);
  if (FILTER == 0u) {
    textureStore(tex_out, gid.xy, textureSampleLevel(tex_in, lin, uv, 0.0));
    return;
  }

  // Сепарабельно: AXIS 0 — по x (вход → ширина выхода × высота входа),
  // AXIS 1 — по y. Шаг выхода в пикселях входа; при увеличении ядро не
  // сужаем меньше 1.
  let axis = vec2f(f32(AXIS == 0u), f32(AXIS == 1u));
  let step = max(dot(dim_in / vec2f(dim_out), axis), 1.0);
  let center = dot(uv * dim_in - 0.5, axis);
  let first = i32(floor(center - 2.0 * step)) + 1;
  let last = i32(floor(center + 2.0 * step));
  let edge = vec2i(dim_in) - 1;
  let base = vec2i(gid.xy);
  let dir = vec2i(axis);

  var sum = vec4f(0.0);
  var weight = 0.0;
  for (var i = first; i <= last; i = i + 1) {
    let w = catmullRom((f32(i) - center) / step);
    let p = base * (vec2i(1) - dir) + dir * i;
    sum += w * textureLoad(tex_in, clamp(p, vec2i(0), edge), 0);
    weight += w;
  }
  textureStore(tex_out, gid.xy, sum / weight);
}
