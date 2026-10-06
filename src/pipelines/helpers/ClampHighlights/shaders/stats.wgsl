// Anime4K-v4.0-De-Ring-Compute-Statistics: максимум яркости в окне 5×5.
// В оригинале — два прохода (X, Y) на HOOK MAIN; max сепарабелен, так что
// один проход 5×5 даёт то же самое. Края — clamp, как texOff в mpv.
@group(0) @binding(0) var tex_in: texture_2d<f32>; // исходный кадр
@group(0) @binding(1) var tex_out: texture_storage_2d<r32float, write>; // STATSMAX

const KERNEL_HALF: i32 = 2;

fn get_luma(rgba: vec4f) -> f32 {
  return dot(rgba, vec4f(0.299, 0.587, 0.114, 0.0));
}

@compute
@workgroup_size(8, 8)
fn computeMain(@builtin(global_invocation_id) pixel: vec3u) {
  let dim: vec2u = textureDimensions(tex_in);
  if (pixel.x >= dim.x || pixel.y >= dim.y) {
    return;
  }

  let last = vec2i(dim) - 1;
  var gmax: f32 = 0.0;
  for (var dy: i32 = -KERNEL_HALF; dy <= KERNEL_HALF; dy = dy + 1) {
    for (var dx: i32 = -KERNEL_HALF; dx <= KERNEL_HALF; dx = dx + 1) {
      let p = clamp(vec2i(pixel.xy) + vec2i(dx, dy), vec2i(0), last);
      gmax = max(gmax, get_luma(textureLoad(tex_in, p, 0)));
    }
  }
  textureStore(tex_out, pixel.xy, vec4f(gmax, 0.0, 0.0, 1.0));
}
