// Anime4K-v4.0-De-Ring-Clamp (HOOK PREKERNEL): последний шаг цепочки, в
// разрешении выхода. Яркость не выше максимума соседей в исходнике — гасит
// светлые ореолы (ringing), которые дорисовали CNN.
@group(0) @binding(0) var tex_in: texture_2d<f32>; // результат цепочки
@group(0) @binding(1) var tex_stats: texture_2d<f32>; // STATSMAX, разрешение исходника
@group(0) @binding(2) var tex_out: texture_storage_2d<rgba16float, write>;

fn get_luma(rgba: vec4f) -> f32 {
  return dot(rgba, vec4f(0.299, 0.587, 0.114, 0.0));
}

fn statsAt(p: vec2i, last: vec2i) -> f32 {
  return textureLoad(tex_stats, clamp(p, vec2i(0), last), 0).x;
}

// STATSMAX_tex(HOOKED_pos): билинейная выборка в координатах выхода.
// r32float не фильтруется без float32-filterable — интерполируем вручную.
fn statsBilinear(pos: vec2f) -> f32 {
  let dim = textureDimensions(tex_stats);
  let last = vec2i(dim) - 1;
  let t = pos * vec2f(dim) - 0.5;
  let base = floor(t);
  let w = t - base;
  let i = vec2i(base);
  let top = mix(statsAt(i, last), statsAt(i + vec2i(1, 0), last), w.x);
  let bottom = mix(statsAt(i + vec2i(0, 1), last), statsAt(i + vec2i(1, 1), last), w.x);
  return mix(top, bottom, w.y);
}

@compute
@workgroup_size(8, 8)
fn computeMain(@builtin(global_invocation_id) pixel: vec3u) {
  let dim: vec2u = textureDimensions(tex_out);
  if (pixel.x >= dim.x || pixel.y >= dim.y) {
    return;
  }

  let color: vec4f = textureLoad(tex_in, pixel.xy, 0);
  let luma: f32 = get_luma(color);
  let pos = (vec2f(pixel.xy) + 0.5) / vec2f(dim);
  let new_luma: f32 = min(luma, statsBilinear(pos));
  // У BT.709 обратная матрица Y→RGB даёт 1 в каждой строке: сдвиг яркости —
  // это одинаковый сдвиг всех трёх каналов (так и в оригинале).
  let diff: f32 = luma - new_luma;
  textureStore(tex_out, pixel.xy, vec4f(color.rgb - diff, color.a));
}
