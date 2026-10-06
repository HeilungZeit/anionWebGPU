import { Anime4KPipeline } from '../../interfaces';
import diffWGSL from './shaders/diff.wgsl';
import decideWGSL from './shaders/decide.wgsl';
import copyWGSL from './shaders/copy.wgsl';

export interface FrameGateDescriptor {
  device: GPUDevice;
  /** Исходный кадр цепочки — его и сравниваем с прошлым. */
  inputTexture: GPUTexture;
  /**
   * Порог в уровнях 8 бит: кадр пропускается, если ни один канал ни одного
   * пикселя не отличается от последнего посчитанного больше чем на него.
   * 0 (по умолчанию) — только точные повторы: выход тот же, что без ворот.
   */
  threshold?: number;
  name?: string;
}

/** Запуск звена: напрямую или косвенно через ворота. */
export interface Launch {
  dispatch(pass: GPUComputePassEncoder): void;
}

/** Счётчики ворот с момента сборки. */
export interface FrameGateStats {
  frames: number;
  skipped: number;
  /** max |Δ| последнего кадра с опорным, в уровнях 8 бит. */
  lastDiff: number;
}

/** Слотов хватает на любую цепочку ModeC/ModeCA (их там до ~25). */
const CAPACITY = 64;
const SLOT_BYTES = 12;

const storage = (type: GPUBufferBindingType): GPUBindGroupLayoutEntry['buffer'] => ({ type });

/**
 * Запуск звена с `gate` — косвенный из буфера ворот, без него — как раньше.
 * Число групп известно при сборке звена, поэтому слот берётся там же.
 */
export function launcher(gate: FrameGate | undefined, x: number, y: number): Launch {
  if (gate) return gate.slot(x, y);
  return { dispatch: (pass) => pass.dispatchWorkgroups(x, y) };
}

/**
 * Ворота повторов. В аниме много кадров подряд с тем же рисунком (анимация
 * «на двойках», статичные планы), а цепочка CNN каждый раз считала их заново.
 * Ворота сравнивают новый кадр с последним посчитанным и на повторе обнуляют
 * аргументы косвенного запуска всех звеньев: GPU пропускает работу, выход
 * остаётся прошлым. Решение принимается на GPU — CPU не ждёт чтения.
 *
 * Звено ставится первым в цепочке; остальные получают `gate` и запускаются
 * через `launcher()`.
 */
export class FrameGate implements Anime4KPipeline {
  name: string;

  ready: Promise<void>;

  /** Аргументы косвенного запуска звеньев (INDIRECT). */
  readonly args: GPUBuffer;

  private device: GPUDevice;

  private template: GPUBuffer;

  private state: GPUBuffer;

  private readback?: GPUBuffer;

  private reading?: Promise<FrameGateStats>;

  private prev: GPUTexture;

  private input: GPUTexture;

  private slots = 0;

  private width: number;

  private height: number;

  private pipelines: { diff?: GPUComputePipeline; decide?: GPUComputePipeline; copy?: GPUComputePipeline } = {};

  private bindGroups: { diff: GPUBindGroup; decide: GPUBindGroup; copy: GPUBindGroup };

  private copyLaunch: Launch;

  constructor({
    device, inputTexture, threshold = 0, name = 'frame gate',
  }: FrameGateDescriptor) {
    this.name = name;
    this.device = device;
    this.input = inputTexture;
    this.width = inputTexture.width;
    this.height = inputTexture.height;

    this.args = device.createBuffer({
      label: `${name}: args`,
      size: CAPACITY * SLOT_BYTES,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE,
    });
    this.template = device.createBuffer({
      label: `${name}: template`,
      size: CAPACITY * SLOT_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.state = device.createBuffer({
      label: `${name}: state`,
      size: 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    // Опорный кадр в 8 битах: сравнение идёт в уровнях 8 бит, как у видео.
    this.prev = device.createTexture({
      label: `${name}: previous frame`,
      size: [this.width, this.height, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    });

    this.setThreshold(threshold);
    // Первый кадр считается всегда: опорный ещё пуст.
    this.invalidate();
    this.copyLaunch = this.slot(Math.ceil(this.width / 8), Math.ceil(this.height / 8));

    const diffLayout = device.createBindGroupLayout({
      label: `${name}: diff layout`,
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: {} },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: {} },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: storage('storage') },
      ],
    });
    const decideLayout = device.createBindGroupLayout({
      label: `${name}: decide layout`,
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: storage('storage') },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: storage('read-only-storage') },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: storage('storage') },
      ],
    });
    const copyLayout = device.createBindGroupLayout({
      label: `${name}: copy layout`,
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: {} },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'rgba8unorm' },
        },
      ],
    });
    this.bindGroups = {
      diff: device.createBindGroup({
        label: `${name}: diff`,
        layout: diffLayout,
        entries: [
          { binding: 0, resource: inputTexture.createView() },
          { binding: 1, resource: this.prev.createView() },
          { binding: 2, resource: { buffer: this.state, offset: 0, size: 4 } },
        ],
      }),
      decide: device.createBindGroup({
        label: `${name}: decide`,
        layout: decideLayout,
        entries: [
          { binding: 0, resource: { buffer: this.state } },
          { binding: 1, resource: { buffer: this.template } },
          { binding: 2, resource: { buffer: this.args } },
        ],
      }),
      copy: device.createBindGroup({
        label: `${name}: copy`,
        layout: copyLayout,
        entries: [
          { binding: 0, resource: inputTexture.createView() },
          { binding: 1, resource: this.prev.createView() },
        ],
      }),
    };

    const compile = (key: 'diff' | 'decide' | 'copy', code: string, layout: GPUBindGroupLayout) => device
      .createComputePipelineAsync({
        label: `${name}: ${key}`,
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module: device.createShaderModule({ label: `${name}: ${key}`, code }), entryPoint: 'computeMain' },
      })
      .then((pipeline) => { this.pipelines[key] = pipeline; });
    this.ready = Promise.all([
      compile('diff', diffWGSL, diffLayout),
      compile('decide', decideWGSL, decideLayout),
      compile('copy', copyWGSL, copyLayout),
    ]).then(() => undefined);
  }

  /** Слот косвенного запуска на `x × y` групп; звено зовёт при сборке. */
  slot(x: number, y: number): Launch {
    if (this.slots >= CAPACITY) {
      throw new Error(`${this.name}: больше ${CAPACITY} запусков в цепочке.`);
    }
    const offset = this.slots * SLOT_BYTES;
    this.slots += 1;
    this.device.queue.writeBuffer(this.template, offset, new Uint32Array([x, y, 1]));
    this.device.queue.writeBuffer(this.state, 16, new Uint32Array([this.slots]));
    return { dispatch: (pass) => pass.dispatchWorkgroupsIndirect(this.args, offset) };
  }

  /** Порог в уровнях 8 бит (см. `threshold`). */
  setThreshold(threshold: number): void {
    this.device.queue.writeBuffer(this.state, 12, new Uint32Array([Math.max(0, Math.floor(threshold))]));
  }

  /** Посчитать следующий кадр, даже если он повтор. */
  invalidate(): void {
    this.device.queue.writeBuffer(this.state, 20, new Uint32Array([1]));
  }

  /** Счётчики с GPU; параллельные вызовы получают одно чтение. */
  readStats(): Promise<FrameGateStats> {
    if (this.reading) return this.reading;
    this.readback ??= this.device.createBuffer({
      label: `${this.name}: readback`,
      size: 32,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const readback = this.readback;
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.state, 0, readback, 0, 32);
    this.device.queue.submit([encoder.finish()]);
    this.reading = readback.mapAsync(GPUMapMode.READ).then(() => {
      const s = new Uint32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      return { frames: s[1], skipped: s[2], lastDiff: s[6] };
    }).finally(() => { this.reading = undefined; });
    return this.reading;
  }

  updateParam(param: string, value: any): void {
    if (param === 'threshold') {
      this.setThreshold(Number(value));
      return;
    }
    throw new Error(`${this.name} has no param ${param}.`);
  }

  pass(encoder: GPUCommandEncoder): void {
    const { diff, decide, copy } = this.pipelines;
    if (!diff || !decide || !copy) {
      throw new Error(`${this.name}: шейдеры ещё компилируются — дождитесь ready.`);
    }
    encoder.clearBuffer(this.state, 0, 4);
    const pass = encoder.beginComputePass({ label: this.name });
    pass.setPipeline(diff);
    pass.setBindGroup(0, this.bindGroups.diff);
    pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
    pass.setPipeline(decide);
    pass.setBindGroup(0, this.bindGroups.decide);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(copy);
    pass.setBindGroup(0, this.bindGroups.copy);
    this.copyLaunch.dispatch(pass);
    pass.end();
  }

  /** Ворота ничего не выводят — отдают вход как есть. */
  getOutputTexture(): GPUTexture {
    return this.input;
  }
}
