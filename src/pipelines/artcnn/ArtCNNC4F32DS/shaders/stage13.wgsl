// ArtCNN C4F32 DS (Conv2D-6) + Depth-To-Space
// Сгенерировано conversion/artcnn.py — не править.
// Точность — псевдонимы T4/M4/A4/S1, их объявляет helpers/CNN (f32 или f16).
@group(0) @binding(0) var tex_0: texture_2d<f32>; // conv2d_6#0
@group(0) @binding(1) var tex_main: texture_2d<f32>; // RGB
@group(0) @binding(2) var main_sampler: sampler;
@group(0) @binding(3) var tex_out: texture_storage_2d<rgba16float, write>;

// Нули за краем кадра — как паддинг 'same' при обучении.
fn fetch(t: texture_2d<f32>, p: vec2i) -> vec4f {
  if (any(p < vec2i(0)) || any(p >= vec2i(textureDimensions(t)))) {
    return vec4f(0.0);
  }
  return textureLoad(t, p, 0);
}

@compute @workgroup_size(8, 8)
fn computeMain(@builtin(global_invocation_id) gid: vec3u) {
  let dim = vec2i(textureDimensions(tex_0));
  let p = vec2i(gid.xy);
  if (any(p >= dim)) {
    return;
  }
  let v = textureLoad(tex_0, p, 0);
  let out_dim = vec2f(textureDimensions(tex_out));
  {
    let o = p * 2 + vec2i(0, 0);
    let base = textureSampleLevel(tex_main, main_sampler, (vec2f(o) + 0.5) / out_dim, 0.0);
    let y = clamp(v[0], 0.0, 1.0);
    let rgb = base.rgb + (y - dot(base.rgb, vec3f(0.2126, 0.7152, 0.0722)));
    textureStore(tex_out, o, deRing(vec4f(clamp(rgb, vec3f(0.0), vec3f(1.0)), 1.0), o, out_dim));
  }
  {
    let o = p * 2 + vec2i(1, 0);
    let base = textureSampleLevel(tex_main, main_sampler, (vec2f(o) + 0.5) / out_dim, 0.0);
    let y = clamp(v[1], 0.0, 1.0);
    let rgb = base.rgb + (y - dot(base.rgb, vec3f(0.2126, 0.7152, 0.0722)));
    textureStore(tex_out, o, deRing(vec4f(clamp(rgb, vec3f(0.0), vec3f(1.0)), 1.0), o, out_dim));
  }
  {
    let o = p * 2 + vec2i(0, 1);
    let base = textureSampleLevel(tex_main, main_sampler, (vec2f(o) + 0.5) / out_dim, 0.0);
    let y = clamp(v[2], 0.0, 1.0);
    let rgb = base.rgb + (y - dot(base.rgb, vec3f(0.2126, 0.7152, 0.0722)));
    textureStore(tex_out, o, deRing(vec4f(clamp(rgb, vec3f(0.0), vec3f(1.0)), 1.0), o, out_dim));
  }
  {
    let o = p * 2 + vec2i(1, 1);
    let base = textureSampleLevel(tex_main, main_sampler, (vec2f(o) + 0.5) / out_dim, 0.0);
    let y = clamp(v[3], 0.0, 1.0);
    let rgb = base.rgb + (y - dot(base.rgb, vec3f(0.2126, 0.7152, 0.0722)));
    textureStore(tex_out, o, deRing(vec4f(clamp(rgb, vec3f(0.0), vec3f(1.0)), 1.0), o, out_dim));
  }
}
