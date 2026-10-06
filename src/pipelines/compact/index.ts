import { Anime4KPipeline } from '../interfaces';
import { FrameGate, Launch, launcher } from '../helpers/FrameGate';
import {
  CompactKernel, CompactModelMeta, groups, kernelRegion, layerBases, layerMatrices, layerShader, packMatrices,
} from './shaders';

export type { CompactKernel, CompactLayerMeta, CompactModelMeta } from './shaders';

/** Веса и описание модели — как их выдаёт конвертер anion-dl. */
export interface CompactModelData {
  meta: CompactModelMeta;
  /** `weights.bin`: f16 в порядке чтения шейдера. */
  weights: ArrayBuffer;
}

export interface CompactPipelineDescriptor {
  /** Нужна фича 'shader-f16'. */
  device: GPUDevice;
  inputTexture: GPUTexture;
  model: CompactModelData;
  /**
   * Ядро, по умолчанию `fast`. `reference` — как в anion-dl (эталон),
   * `fast-f32acc` — быстрое с суммой в f32, точнее и медленнее.
   */
  kernel?: CompactKernel;
  /** Ворота повторов: слои запускаются косвенно и на повторе пропускаются. */
  gate?: FrameGate;
  name?: string;
}

interface Step {
  pipeline?: GPUComputePipeline;
  bindGroup: GPUBindGroup;
}

/**
 * SRVGGNetCompact ×2 (AnimeJaNai V2 SuperUltraCompact, режим «Детали»):
 * слой — compute-проход, признаки между слоями — планарные буферы f16
 * (6 × vec4<f16> на пиксель), выход — текстура rgba16float ×2.
 */
export class CompactSR implements Anime4KPipeline {
  name: string;

  ready: Promise<void>;

  outputTexture: GPUTexture;

  private steps: Step[] = [];

  private launch: Launch;

  constructor({
    device, inputTexture, model, kernel = 'fast', gate, name = 'CompactSR',
  }: CompactPipelineDescriptor) {
    this.name = name;
    const { meta, weights } = model;
    if (meta.arch !== 'SRVGGNetCompact') throw new Error(`${name}: архитектура ${meta.arch} не поддерживается`);
    if (weights.byteLength !== meta.bytes) {
      throw new Error(`${name}: веса ${weights.byteLength} Б, в описании ${meta.bytes} Б`);
    }
    if (!device.features.has('shader-f16')) throw new Error(`${name}: нужна фича 'shader-f16'`);

    const { width, height } = inputTexture;
    const { scale } = meta;
    const [regionX, regionY] = kernelRegion(kernel);
    // У всех слоёв одна сетка групп — один слот косвенного запуска на все.
    this.launch = launcher(gate, Math.ceil(width / regionX), Math.ceil(height / regionY));

    this.outputTexture = device.createTexture({
      label: `${name}: output`,
      size: [width * scale, height * scale, 1],
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    });

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const featBytes = width * height * groups(meta.numFeat) * 8;
    const ping = device.createBuffer({ label: `${name}: ping`, size: featBytes, usage: storage });
    const pong = device.createBuffer({ label: `${name}: pong`, size: featBytes, usage: storage });

    // Быстрым ядрам веса переложены матрицами (packMatrices): f16 берутся
    // битами, без пересчёта. Смещения слоёв — свои у каждой раскладки.
    const fast = kernel !== 'reference';
    const raw = new Uint16Array(weights);
    const packed = fast ? packMatrices(meta) : null;
    const halves = packed
      ? Uint16Array.from(packed.order, (i) => (i === 0xffffffff ? 0 : raw[i]))
      : raw;
    const blob = device.createBuffer({ label: `${name}: weights`, size: halves.byteLength, usage: storage });
    device.queue.writeBuffer(blob, 0, halves);

    const compiling: Promise<void>[] = [];
    let src: GPUBuffer | null = null;
    let dst = ping;
    meta.layers.forEach((layer, index) => {
      const first = index === 0;
      const last = index === meta.layers.length - 1;
      const bases = packed ? packed.bases[index] : layerBases(layer);

      const dims = device.createBuffer({
        label: `${name}: dims ${index}`,
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(dims, 0, new Uint32Array([width, height, bases.w, bases.bias, bases.prelu, 0, 0, 0]));

      const entries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: dims } },
        { binding: 1, resource: first ? inputTexture.createView() : { buffer: src as GPUBuffer } },
        { binding: 2, resource: last ? this.outputTexture.createView() : { buffer: dst } },
      ];
      entries.push({ binding: 3, resource: { buffer: blob } });
      if (last) entries.push({ binding: 4, resource: inputTexture.createView() });
      if (packed) {
        // Веса слоя — отдельным буфером матриц: выборка mat4x4 целиком, с
        // одной проверкой границ (из общего блоба по vec4 — четыре).
        const size = layerMatrices(layer) * 32;
        const matrices = device.createBuffer({ label: `${name}: weights ${index}`, size, usage: storage });
        device.queue.writeBuffer(matrices, 0, halves, bases.w * 4, size / 2);
        entries.push({ binding: 5, resource: { buffer: matrices } });
      }

      const module = device.createShaderModule({
        label: `${name}: layer ${index}`,
        code: layerShader({
          kernel, layer, first, tailScale: last ? scale : undefined,
        }),
      });
      // Раскладка 'auto' видит только привязки, которые шейдер читает, —
      // группа собирается после компиляции.
      const step: Step = { bindGroup: undefined as unknown as GPUBindGroup };
      this.steps.push(step);
      compiling.push(device.createComputePipelineAsync({
        label: `${name}: layer ${index}`,
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      }).then((pipeline) => {
        step.pipeline = pipeline;
        step.bindGroup = device.createBindGroup({
          label: `${name}: layer ${index}`,
          layout: pipeline.getBindGroupLayout(0),
          entries,
        });
      }));

      if (!last) {
        src = dst;
        dst = dst === ping ? pong : ping;
      }
    });
    this.ready = Promise.all(compiling).then(() => undefined);
  }

  updateParam(param: string, value: any): void {
    throw new Error(`${this.name} has no param.`);
  }

  pass(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: this.name });
    this.steps.forEach((step) => {
      if (!step.pipeline) throw new Error(`${this.name}: шейдеры ещё компилируются — дождитесь ready.`);
      pass.setPipeline(step.pipeline);
      pass.setBindGroup(0, step.bindGroup);
      this.launch.dispatch(pass);
    });
    pass.end();
  }

  getOutputTexture(): GPUTexture {
    return this.outputTexture;
  }
}
