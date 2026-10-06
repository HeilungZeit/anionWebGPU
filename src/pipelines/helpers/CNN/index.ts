import { Anime4KPipeline, CNNPrecision } from '../../interfaces';
import { CNNModel } from './model';
import { DeRingEpilogue } from '../ClampHighlights/stats';
import { FrameGate, Launch, launcher } from '../FrameGate';

export * from './model';

export interface CNNPipelineDescriptor {
  device: GPUDevice;
  inputTexture: GPUTexture;
  model: CNNModel;
  name?: string;
  precision?: CNNPrecision;
  /**
   * Статистика ClampStats: финальная стадия сразу выполняет Clamp Highlights
   * (зажим ореолов) — без отдельного прохода в разрешении выхода.
   */
  deRing?: GPUTexture;
  /** Ворота повторов: стадии запускаются косвенно и на повторе пропускаются. */
  gate?: FrameGate;
}

// Шейдеры генераторов пишут типы через псевдонимы T4/M4/A4/S1.
const PRELUDE = {
  f32: 'alias T4 = vec4f;\nalias M4 = mat4x4f;\nalias A4 = vec4f;\nalias S1 = f32;\n',
  f16: 'enable f16;\nalias T4 = vec4h;\nalias M4 = mat4x4h;\nalias A4 = vec4h;\nalias S1 = f16;\n',
};

// Чтение упакованной rgba32uint (CNN_PACK): половина текселя — 4 канала f16.
const UNPACK = {
  f32: 'fn unpack_half(a: u32, b: u32) -> T4 { return T4(unpack2x16float(a), unpack2x16float(b)); }\n',
  f16: 'fn unpack_half(a: u32, b: u32) -> T4 { return T4(bitcast<vec2h>(a), bitcast<vec2h>(b)); }\n',
};

interface CompiledStage {
  /** Появляется, когда createComputePipelineAsync завершится. */
  pipeline?: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  /** Группа 1 финальной стадии — статистика для deRing(). */
  deRing?: GPUBindGroup;
  launch: Launch;
}

/**
 * Исполнитель CNN-модели из conversion/cnn.py: стадия — один compute-проход,
 * поток — один пиксель входа (финальная стадия x2 пишет 2×2 пикселя выхода).
 */
export class CNN implements Anime4KPipeline {
  name: string;

  precision: CNNPrecision;

  stages: CompiledStage[] = [];

  ready: Promise<void>;

  outputTexture: GPUTexture;

  private block: [number, number];

  private width: number;

  private height: number;

  constructor({
    device, inputTexture, model, name = 'cnn', precision = 'f32', deRing, gate,
  }: CNNPipelineDescriptor) {
    this.name = name;
    if (precision === 'f16' && !device.features.has('shader-f16')) {
      throw new Error(`${name}: precision 'f16' requires a device with 'shader-f16'.`);
    }
    this.precision = precision;
    this.block = model.block;
    this.width = inputTexture.width;
    this.height = inputTexture.height;

    const packed = new Set(model.packed);
    const textures: GPUTexture[] = [inputTexture];
    for (let i = 1; i < model.textures; i += 1) {
      textures.push(device.createTexture({
        label: `${name}: feature ${i}`,
        size: [this.width, this.height, 1],
        format: packed.has(i) ? 'rgba32uint' : 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      }));
    }
    this.outputTexture = device.createTexture({
      label: `${name}: output`,
      size: [this.width * model.scale, this.height * model.scale, 1],
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    });
    const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    const epilogue = new DeRingEpilogue(device, deRing);

    const compiling: Promise<void>[] = [];
    model.stages.forEach((stage, n) => {
      const layoutEntries: GPUBindGroupLayoutEntry[] = [];
      const entries: GPUBindGroupEntry[] = [];
      const addTexture = (texture: GPUTexture) => {
        const binding = layoutEntries.length;
        layoutEntries.push({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          texture: texture.format === 'rgba32uint' ? { sampleType: 'uint' } : {},
        });
        entries.push({ binding, resource: texture.createView() });
      };
      const addStorage = (texture: GPUTexture) => {
        const binding = layoutEntries.length;
        layoutEntries.push({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: texture.format },
        });
        entries.push({ binding, resource: texture.createView() });
      };

      stage.inputs.forEach((i) => addTexture(textures[i]));
      if (stage.final) {
        addTexture(inputTexture);
        const binding = layoutEntries.length;
        layoutEntries.push({ binding, visibility: GPUShaderStage.COMPUTE, sampler: {} });
        entries.push({ binding, resource: sampler });
        addStorage(this.outputTexture);
      } else {
        stage.outputs.forEach((i) => addStorage(textures[i]));
      }

      const layout = device.createBindGroupLayout({ label: `${name}: stage ${n} layout`, entries: layoutEntries });
      const final = Boolean(stage.final);
      const compiled: CompiledStage = {
        bindGroup: device.createBindGroup({ label: `${name}: stage ${n}`, layout, entries }),
        deRing: final ? epilogue.bindGroup : undefined,
        launch: launcher(
          gate,
          Math.ceil(this.width / (8 * this.block[0])),
          Math.ceil(this.height / (8 * this.block[1])),
        ),
      };
      this.stages.push(compiled);
      // Компиляция — в фоне: синхронный createComputePipeline занимал
      // GPU-процесс браузера на секунды, и вся страница замирала.
      compiling.push(device.createComputePipelineAsync({
        label: `${name}: stage ${n}`,
        layout: device.createPipelineLayout({
          bindGroupLayouts: final ? [layout, epilogue.layout] : [layout],
        }),
        compute: {
          module: device.createShaderModule({
            label: `${name}: stage ${n}`,
            // enable-директивы — до любых объявлений, поэтому subgroups первым.
            code: (stage.subgroups ? 'enable subgroups;\n' : '')
              + PRELUDE[this.precision] + (packed.size ? UNPACK[this.precision] : '') + (final ? DeRingEpilogue.wgsl : '') + stage.wgsl,
          }),
          entryPoint: 'computeMain',
          constants: final ? epilogue.constants : {},
        },
      }).then((pipeline) => {
        compiled.pipeline = pipeline;
      }));
    });
    this.ready = Promise.all(compiling).then(() => undefined);
  }

  updateParam(param: string, value: any): void {
    throw new Error(`${this.name} has no param.`);
  }

  pass(encoder: GPUCommandEncoder): void {
    // Один проход: в compute-проходе каждый dispatch — своя область
    // синхронизации, запись стадии видна следующей.
    const pass = encoder.beginComputePass({ label: this.name });
    this.stages.forEach((stage) => {
      if (!stage.pipeline) {
        throw new Error(`${this.name}: шейдеры ещё компилируются — дождитесь ready.`);
      }
      pass.setPipeline(stage.pipeline);
      pass.setBindGroup(0, stage.bindGroup);
      if (stage.deRing) pass.setBindGroup(1, stage.deRing);
      stage.launch.dispatch(pass);
    });
    pass.end();
  }

  getOutputTexture(): GPUTexture {
    return this.outputTexture;
  }
}
