
//#region src/pipelines/compact/shaders.ts
/**
* WGSL для SRVGGNetCompact («Детали» в anion-dl, AnimeJaNai V2
* SuperUltraCompact): 10 свёрток 3×3 с PReLU в разрешении источника,
* увеличение — pixelshuffle последнего слоя.
*
* Ядра (`CompactKernel`):
* - `reference` — перенос ядра anion-dl (`src/app/player/compact/shaders.ts`,
*   сверено там с onnxruntime на 72.6 дБ): по пикселю на поток, 24
*   накопителя f32, `dot` на vec4<f16>, веса из storage-буфера. Эталон для
*   сверки быстрых ядер.
* - `fast` — слой как умножение mat4x4<f16> на vec4<f16>: накопители — vec4
*   по четыре выходных канала, 16 FMA на 16 умножений, без переводов
*   f16 → f32 на каждом слагаемом; поток — 2×2 пикселя × 3 выходные группы
*   из 6 (вес читается раз на четыре пикселя), тайл `[группа][пиксель]`; веса слоя — отдельный storage-буфер `array<mat4x4<f16>>`: матрица
*   читается одной выборкой с одной проверкой границ. Сумма целиком в f16.
* - `fast-f32acc` — то же, но сумма по каждому отводу ядра (24 слагаемых)
*   копится в f16 и переносится в накопитель f32: точнее, но медленнее.
*
* Замер (Chrome 153, M1 Pro, 1280×720, только сеть): reference 45.6 мс,
* fast-f32acc 38.1 мс, fast 29.3 мс; раскладка 2×2 × 3 группы (Э14) — ещё
* −9.5% у fast и −18% у fast-f32acc, выход бит в бит прежний. Проверены и отброшены (docs/PLAN.md, Э10):
* веса литералами в коде (в 10 раз медленнее), mat4x4<f32>, веса в общей
* памяти группы, 3–4 пикселя на поток, веса в uniform-буфере (на wgpu −8%, в
* Chrome +50%: Tint раскладывает mat4x4<f16> в uniform по-своему).
*
* Отличия от anion-dl — только ввод-вывод: первый слой читает кадр прямо из
* текстуры (без прохода препроцесса), последний пишет выход ×2 в текстуру
* rgba16float, зажатый в 0..1.
*/
/** Каналы пакуются по четыре в vec4<f16>. */
const PACK = 4;
function groups(channels) {
	return Math.ceil(channels / PACK);
}
/**
* Быстрые ядра: поток считает 2×2 пикселя × 3 выходные группы из 6 — весов
* на умножение читается вдвое меньше, чем при 2 пикселях × 6 группах, а
* накопителей столько же. Хвост (3 группы) на z не делится — у него 2×1
* пикселя × все группы; область та же, 32×8.
*/
const GEOMETRY = {
	reference: {
		threads: [
			8,
			8,
			1
		],
		block: [1, 1]
	},
	fast: {
		threads: [
			16,
			4,
			2
		],
		block: [2, 2]
	},
	"fast-f32acc": {
		threads: [
			16,
			4,
			2
		],
		block: [2, 2]
	}
};
const FAST_TAIL = {
	threads: [
		16,
		8,
		1
	],
	block: [2, 1]
};
function geometry(kernel, tail) {
	return tail && kernel !== "reference" ? FAST_TAIL : GEOMETRY[kernel];
}
/** Сколько пикселей источника покрывает рабочая группа — для диспетча. */
function kernelRegion(kernel) {
	const { threads, block } = GEOMETRY[kernel];
	return [threads[0] * block[0], threads[1] * block[1]];
}
/**
* Смещения блоков слоя в блобе конвертера, в vec4<f16>: веса
* `[out][tap][inGroup]`, затем смещения (outGroups vec4), затем наклоны PReLU.
*/
function layerBases(layer) {
	const w = layer.offset / 8;
	const bias = w + layer.out * 9 * layer.inGroups;
	return {
		w,
		bias,
		prelu: bias + layer.outGroups
	};
}
/** Число mat4x4 весов слоя у быстрых ядер. */
function layerMatrices(layer) {
	return 9 * layer.inGroups * groups(layer.out);
}
/**
* Блоб быстрых ядер: по слою — веса `[tap][inGroup][outGroup]` как mat4x4
* (столбец c — входной канал inGroup·4+c, строка r — выходной outGroup·4+r),
* затем смещения и наклоны PReLU по outGroups vec4. `order` — индексы f16 в
* исходном блобе (0xffffffff — ноль: хвост выходных каналов); смещения — в vec4.
*/
function packMatrices(meta) {
	const order = [];
	const bases = meta.layers.map((layer) => {
		const src = layerBases(layer);
		const outGroups = groups(layer.out);
		const w = order.length / PACK;
		for (let tap = 0; tap < 9; tap += 1) for (let g = 0; g < layer.inGroups; g += 1) for (let og = 0; og < outGroups; og += 1) for (let c = 0; c < PACK; c += 1) for (let r = 0; r < PACK; r += 1) {
			const o = og * PACK + r;
			order.push(o < layer.out ? (src.w + (o * 9 + tap) * layer.inGroups + g) * PACK + c : -1);
		}
		const bias = order.length / PACK;
		for (let i = 0; i < outGroups * PACK; i += 1) order.push(src.bias * PACK + i);
		const prelu = order.length / PACK;
		if (layer.prelu) for (let i = 0; i < outGroups * PACK; i += 1) order.push(src.prelu * PACK + i);
		return {
			w,
			bias,
			prelu
		};
	});
	return {
		order: Uint32Array.from(order, (i) => i < 0 ? 4294967295 : i),
		bases
	};
}
const range = (n) => Array.from({ length: n }, (_, i) => i);
/** Индекс в тайле: ячейка `pixel` (HW×HH), группа каналов `g`. */
function tileAt(pixel, g, planar) {
	return planar ? `${g} * HW * HH + ${pixel}` : `(${pixel}) * IN_G + ${g}`;
}
/**
* Тайл активаций с каймой 1 px: кайма за краем кадра — нули, как padding=1 в
* Conv2d. Обнулять общую память заранее не нужно: тайл пишется целиком.
* Раскладка: `[пиксель][группа]` или, с `planar`, `[группа][пиксель]` —
* соседние потоки читают соседние ячейки (`tileAt`).
*/
function loadTile(first, threads, region, planar) {
	const fill = first ? `    var v = V(0.0);
    if (inside) {
      let c = textureLoad(src, vec2i(gx, gy), 0);
      v = V(f16(c.r), f16(c.g), f16(c.b), 0.0h);
    }
    tile[i] = v;` : `    if (inside) {
      let s = (u32(gy) * dims.width + u32(gx)) * IN_G;
      for (var g = 0u; g < IN_G; g++) { tile[${tileAt("i", "g", planar)}] = src[s + g]; }
    } else {
      for (var g = 0u; g < IN_G; g++) { tile[${tileAt("i", "g", planar)}] = V(0.0); }
    }`;
	return `  let baseX = i32(wg.x * ${region[0]}u) - 1;
  let baseY = i32(wg.y * ${region[1]}u) - 1;
  for (var i = li; i < HW * HH; i += ${threads}u) {
    let gx = baseX + i32(i % HW);
    let gy = baseY + i32(i / HW);
    let inside = gx >= 0 && gy >= 0 && gx < i32(dims.width) && gy < i32(dims.height);
${fill}
  }
  workgroupBarrier();`;
}
/**
* pixelshuffle (CRD, как в PyTorch: канал = цвет·s² + строка·s + столбец) и
* skip ближайшим соседом — в srvgg_arch.py стоит `mode='nearest'`. Сумма — в
* f16, как в anion-dl; выход зажат в 0..1 (там зажимал вывод в rgba8unorm).
* `at(o)` — выражение f16 для выходного канала o.
*
* С `chroma` от сети берётся только яркость, цветность — из сглаженной
* цветности кадра (`chromaShader`), билинейно: см. CompactPipelineDescriptor.
*/
function storeTail(scale, at, indent, chroma) {
	const lines = ["let c = textureLoad(orig, vec2i(i32(ox), i32(oy)), 0);", "let base = V(f16(c.r), f16(c.g), f16(c.b), 0.0h);"];
	for (let rh = 0; rh < scale; rh += 1) for (let rw = 0; rw < scale; rw += 1) {
		const ch = (colour) => at(colour * scale * scale + rh * scale + rw);
		const at2 = `vec2u(ox * ${scale}u + ${rw}u, oy * ${scale}u + ${rh}u)`;
		const rgb = `vec3f(vec3h(${ch(0)} + base.x, ${ch(1)} + base.y, ${ch(2)} + base.z))`;
		if (chroma) lines.push(`{ let p = ${at2}; let uv = (vec2f(p) + 0.5) / vec2f(f32(dims.width * ${scale}u), f32(dims.height * ${scale}u)); textureStore(dst, p, vec4f(withChroma(${rgb}, textureSampleLevel(chroma, chromaSampler, uv, 0.0).xy), 1.0)); }`);
		else lines.push(`textureStore(dst, ${at2}, vec4f(saturate(${rgb}), 1.0));`);
	}
	return lines.map((line) => indent + line).join("\n");
}
/** BT.709: так браузер переводит в RGB видео без меток цвета, а Kodik их не ставит. */
const LUMA = {
	r: .2126,
	b: .0722
};
const CHROMA_WGSL = `const KR = ${LUMA.r};
const KB = ${LUMA.b};
const KG = ${1 - LUMA.r - LUMA.b};

fn toChroma(rgb: vec3f) -> vec2f {
  let y = dot(rgb, vec3f(KR, KG, KB));
  return vec2f((rgb.b - y) / (2.0 * (1.0 - KB)), (rgb.r - y) / (2.0 * (1.0 - KR)));
}`;
/** Яркость — из `rgb`, цветность (Cb, Cr) — `cbcr`; результат зажат в 0..1. */
const WITH_CHROMA_WGSL = `${CHROMA_WGSL}

fn withChroma(rgb: vec3f, cbcr: vec2f) -> vec3f {
  let y = dot(rgb, vec3f(KR, KG, KB));
  let r = y + 2.0 * (1.0 - KR) * cbcr.y;
  let b = y + 2.0 * (1.0 - KB) * cbcr.x;
  return saturate(vec3f(r, (y - KR * r - KB * b) / KG, b));
}`;
/** Веса гаусса σ на отводы −R…R, R = ⌈3σ⌉, нормированы на сумму. */
function gaussTaps(sigma) {
	const radius = Math.ceil(3 * sigma);
	const raw = Array.from({ length: 2 * radius + 1 }, (_, i) => Math.exp(-((i - radius) ** 2) / (2 * sigma * sigma)));
	const sum = raw.reduce((a, b) => a + b, 0);
	return raw.map((w) => w / sum);
}
/**
* Проход сглаживания цветности в разрешении кадра, по одной оси. Ось `x`
* читает кадр и переводит его в Cb/Cr, ось `y` — результат оси `x`.
* Края — повтором крайнего пикселя. Выход rgba16float (rg16float нет среди
* базовых форматов storage-текстур), заняты xy.
*/
function chromaShader(axis, sigma) {
	const taps = gaussTaps(sigma);
	const radius = (taps.length - 1) / 2;
	const read = axis === "x" ? "toChroma(textureLoad(src, q, 0).rgb)" : "textureLoad(src, q, 0).xy";
	const step = axis === "x" ? "vec2i(k, 0)" : "vec2i(0, k)";
	return `${CHROMA_WGSL}

const TAPS = array<f32, ${taps.length}>(${taps.map((w) => w.toFixed(8)).join(", ")});

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2i(textureDimensions(src));
  let p = vec2i(id.xy);
  if (p.x >= size.x || p.y >= size.y) { return; }
  var sum = vec2f(0.0);
  for (var i = 0; i < ${taps.length}; i++) {
    let k = i - ${radius};
    let q = clamp(p + ${step}, vec2i(0), size - 1);
    sum += TAPS[i] * ${read};
  }
  textureStore(dst, p, vec4f(sum, 0.0, 1.0));
}
`;
}
/** Ядро `reference`: как в anion-dl. */
function referenceMain(layer, inGroups, tailScale, chroma) {
	const out = layer.out;
	const lines = [
		"  let ox = wg.x * 8u + lid.x;",
		"  let oy = wg.y * 8u + lid.y;",
		"  // Выход за кадр — только после барьера: соседям нужна загрузка тайла.",
		"  if (ox >= dims.width || oy >= dims.height) { return; }",
		...range(out).map((o) => `  var a${o} = 0.0f;`),
		`  for (var tap = 0u; tap < 9u; tap++) {
    let p = ((lid.y + tap / 3u) * HW + (lid.x + tap % 3u)) * IN_G;`,
		...range(out).map((o) => `    let wo${o} = dims.wBase + (${o}u * 9u + tap) * ${inGroups}u;`),
		`    for (var g = 0u; g < IN_G; g++) {
      let x = tile[p + g];`,
		...range(out).map((o) => `      a${o} += f32(dot(blob[wo${o} + g], x));`),
		"    }",
		"  }",
		...range(out).map((o) => `  a${o} += f32(blob[dims.biasBase + ${Math.floor(o / PACK)}u][${o % PACK}]);`)
	];
	if (layer.prelu) lines.push(...range(out).map((o) => `  a${o} = select(f32(blob[dims.preluBase + ${Math.floor(o / PACK)}u][${o % PACK}]) * a${o}, a${o}, a${o} >= 0.0);`));
	if (tailScale) lines.push(storeTail(tailScale, (o) => `f16(a${o})`, "  ", chroma));
	else {
		lines.push(`  let outBase = (oy * dims.width + ox) * ${groups(out)}u;`);
		for (let og = 0; og < groups(out); og += 1) {
			const parts = range(PACK).map((k) => og * PACK + k < out ? `f16(a${og * PACK + k})` : "0.0h");
			lines.push(`  dst[outBase + ${og}u] = V(${parts.join(", ")});`);
		}
	}
	return lines.join("\n");
}
/**
* Быстрые ядра: mat4x4<f16> × vec4<f16>, несколько пикселей на поток. Поток
* считает `outG / threads.z` выходных групп: группа k — `lid.z * mine + k`.
*/
function fastMain(layer, inGroups, kernel, tailScale, chroma) {
	const { threads, block } = geometry(kernel, Boolean(tailScale));
	const [rw, rh] = kernelRegion(kernel);
	const outG = groups(layer.out);
	const split = threads[2];
	if (outG % split) throw new Error(`Compact: ${outG} выходных групп не делятся на ${split} потока`);
	const mine = outG / split;
	const og = (k) => split === 1 ? `${k}u` : `lid.z * ${mine}u + ${k}u`;
	const wide = kernel === "fast-f32acc";
	const acc = wide ? "vec4f" : "V";
	const px = range(block[0] * block[1]).map((j) => ({
		dx: j % block[0] * threads[0],
		dy: Math.floor(j / block[0]) * threads[1]
	}));
	const sum = wide ? "t" : "a";
	const lines = [
		...range(mine).flatMap((k) => px.map((_, j) => `  var a${k}_${j} = ${acc}(0.0);`)),
		"  for (var tap = 0u; tap < 9u; tap++) {",
		"    let ty = lid.y + tap / 3u;",
		"    let tx = lid.x + tap % 3u;",
		...wide ? range(mine).flatMap((k) => px.map((_, j) => `    var t${k}_${j} = V(0.0);`)) : [],
		"    for (var g = 0u; g < IN_G; g++) {",
		...px.map(({ dx, dy }, j) => `      let x${j} = tile[${tileAt(`(ty + ${dy}u) * HW + tx + ${dx}u`, "g", true)}];`),
		`      let w = (tap * IN_G + g) * ${outG}u;`,
		...range(mine).flatMap((k) => [
			`      { let m = weights[w + ${og(k)}];`,
			...px.map((_, j) => `        ${sum}${k}_${j} += m * x${j};`),
			"      }"
		]),
		"    }",
		...wide ? range(mine).flatMap((k) => px.map((_, j) => `    a${k}_${j} += vec4f(t${k}_${j});`)) : [],
		"  }"
	];
	px.forEach(({ dx, dy }, j) => {
		lines.push(`  {
    let ox = wg.x * ${rw}u + lid.x + ${dx}u;
    let oy = wg.y * ${rh}u + lid.y + ${dy}u;
    if (ox < dims.width && oy < dims.height) {`);
		for (let k = 0; k < mine; k += 1) {
			lines.push(`      var r${k} = a${k}_${j} + ${acc}(blob[dims.biasBase + ${og(k)}]);`);
			if (layer.prelu) lines.push(`      r${k} = select(${acc}(blob[dims.preluBase + ${og(k)}]) * r${k}, r${k}, r${k} >= ${acc}(0.0));`);
		}
		if (tailScale) lines.push(storeTail(tailScale, (o) => `f16(r${Math.floor(o / PACK)}[${o % PACK}])`, "      ", chroma));
		else {
			lines.push(`      let ob = (oy * dims.width + ox) * ${outG}u;`);
			lines.push(...range(mine).map((k) => `      dst[ob + ${og(k)}] = V(r${k});`));
		}
		lines.push("    }\n  }");
	});
	return lines.join("\n");
}
/**
* Шейдер слоя. Привязки: 0 — dims, 1 — вход (текстура или буфер признаков),
* 2 — выход (буфер или текстура), 3 — блоб (у быстрых ядер — только смещения
* и наклоны PReLU), 4 — кадр для skip (последний слой), 5 — веса слоя
* матрицами (быстрые ядра), 6 и 7 — сэмплер и сглаженная цветность
* (последний слой с `chroma`).
*/
function layerShader({ kernel, layer, first, tailScale, chroma = false }) {
	const withChroma = Boolean(tailScale) && chroma;
	const inGroups = first ? 1 : layer.inGroups;
	const { threads } = geometry(kernel, Boolean(tailScale));
	const region = kernelRegion(kernel);
	const fast = kernel !== "reference";
	const hw = region[0] + 2;
	const hh = region[1] + 2;
	return `enable f16;
alias V = vec4<f16>;

const HW: u32 = ${hw}u;
const HH: u32 = ${hh}u;
const IN_G: u32 = ${inGroups}u;

struct Dims { width: u32, height: u32, wBase: u32, biasBase: u32, preluBase: u32 }

@group(0) @binding(0) var<uniform> dims: Dims;
${first ? "@group(0) @binding(1) var src: texture_2d<f32>;" : "@group(0) @binding(1) var<storage, read> src: array<V>;"}
${tailScale ? "@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;" : "@group(0) @binding(2) var<storage, read_write> dst: array<V>;"}
@group(0) @binding(3) var<storage, read> blob: array<V>;
${tailScale ? "@group(0) @binding(4) var orig: texture_2d<f32>;" : ""}
${fast ? "@group(0) @binding(5) var<storage, read> weights: array<mat4x4<f16>>;" : ""}
${withChroma ? `@group(0) @binding(6) var chromaSampler: sampler;
@group(0) @binding(7) var chroma: texture_2d<f32>;

${WITH_CHROMA_WGSL}
` : ""}
var<workgroup> tile: array<V, ${hw * hh * inGroups}>;

@compute @workgroup_size(${threads.join(", ")})
fn main(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
${loadTile(first, threads[0] * threads[1] * threads[2], region, fast)}

${fast ? fastMain(layer, inGroups, kernel, tailScale, withChroma) : referenceMain(layer, inGroups, tailScale, withChroma)}
}
`;
}
//#endregion
export { chromaShader, gaussTaps, groups, kernelRegion, layerBases, layerMatrices, layerShader, packMatrices };
