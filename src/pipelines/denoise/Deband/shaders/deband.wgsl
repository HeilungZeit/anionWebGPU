// Дебандинг до сети: как deband в mpv (4 отсчёта на случайном радиусе, замена
// средним только при малой разнице), но без зерна — сеть приняла бы его за
// фактуру, квантование закрывает дизеринг финального blit.

struct Params {
  // Порог в долях единицы: уровни 8 бит / 255.
  threshold: f32,
  // Радиус выборки первой итерации, пикселей входа.
  range: f32,
  iterations: u32,
}

@group(0) @binding(0) var tex_in: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(0) @binding(2) var tex_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> params: Params;

// PCG-хэш. Затравка — только координата пикселя: узор неподвижен во времени,
// и сглаженная заливка на статичном плане не кипит от кадра к кадру.
fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand(seed: ptr<function, u32>) -> f32 {
  *seed = pcg(*seed);
  return f32(*seed) / 4294967296.0;
}

fn at(uv: vec2f) -> vec3f {
  return textureSampleLevel(tex_in, tex_sampler, uv, 0.0).rgb;
}

@compute @workgroup_size(8, 8)
fn computeMain(@builtin(global_invocation_id) id: vec3u) {
  let dim = textureDimensions(tex_out);
  if (id.x >= dim.x || id.y >= dim.y) {
    return;
  }
  let texel = 1.0 / vec2f(dim);
  let uv = (vec2f(id.xy) + 0.5) * texel;
  var color = textureLoad(tex_in, vec2i(id.xy), 0);
  var seed = pcg(id.x + pcg(id.y));

  for (var i = 1u; i <= params.iterations; i++) {
    let dist = rand(&seed) * params.range * f32(i);
    let dir = rand(&seed) * 6.2831853;
    let o = dist * vec2f(cos(dir), sin(dir));
    let avg = 0.25 * (
      at(uv + vec2f(o.x, o.y) * texel) + at(uv + vec2f(-o.y, o.x) * texel)
      + at(uv - o * texel) + at(uv + vec2f(o.y, -o.x) * texel));
    // Решение на пиксель, а не на канал: на цветном крае один канал мог бы
    // сгладиться без остальных и сдвинуть оттенок. Замена полная до порога и
    // сходит на нет к двойному: жёсткий порог оставлял на сглаженной заливке
    // одиночные точки, которые сеть прорисовала бы.
    let diff = abs(avg - color.rgb);
    let t = params.threshold / f32(i);
    let w = clamp(2.0 - max(diff.r, max(diff.g, diff.b)) / max(t, 1e-6), 0.0, 1.0);
    color = vec4f(mix(color.rgb, avg, w), color.a);
  }

  textureStore(tex_out, vec2i(id.xy), color);
}
