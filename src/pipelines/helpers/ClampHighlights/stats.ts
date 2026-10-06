import { Anime4KPipeline } from '../../interfaces';
import statsWGSL from './shaders/stats.wgsl';
import deringWGSL from './shaders/dering.wgsl';

export interface ClampStatsPipelineDescriptor {
  device: GPUDevice;
  /** Исходный кадр. */
  inputTexture: GPUTexture;
  name?: string;
}

/**
 * Первая половина Clamp Highlights: максимум яркости 5×5 по исходнику
 * (r32float). Вторую — зажим — выполняет последнее звено цепочки, если ему
 * передан `deRing: stats.getOutputTexture()` (CNN-модели, Downscale): так
 * не нужен отдельный проход в разрешении выхода.
 */
export class ClampStats implements Anime4KPipeline {
  name: string;

  pipeline: GPUComputePipeline;

  bindGroup: GPUBindGroup;

  outputTexture: GPUTexture;

  constructor({ device, inputTexture, name = 'clamp stats' }: ClampStatsPipelineDescriptor) {
    this.name = name;
    this.outputTexture = device.createTexture({
      label: `${name}: statsmax_texture`,
      size: [inputTexture.width, inputTexture.height, 1],
      format: 'r32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    });
    this.pipeline = device.createComputePipeline({
      label: `${name} pipeline`,
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ label: `${name}: module`, code: statsWGSL }),
        entryPoint: 'computeMain',
      },
    });
    this.bindGroup = device.createBindGroup({
      label: `${name} bind group`,
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: inputTexture.createView() },
        { binding: 1, resource: this.outputTexture.createView() },
      ],
    });
  }

  updateParam(param: string, value: any): void {
    throw new Error(`${this.name} has no param.`);
  }

  pass(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: this.name });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(this.outputTexture.width / 8),
      Math.ceil(this.outputTexture.height / 8),
    );
    pass.end();
  }

  getOutputTexture(): GPUTexture {
    return this.outputTexture;
  }
}

/**
 * Эпилог зажима для последнего звена: WGSL-функция deRing() и группа
 * привязок 1. Без статистики группа 1 привязана к заглушке, а DERING = false.
 */
export class DeRingEpilogue {
  static readonly wgsl = deringWGSL;

  readonly layout: GPUBindGroupLayout;

  readonly bindGroup: GPUBindGroup;

  readonly constants: Record<string, number>;

  constructor(device: GPUDevice, stats?: GPUTexture) {
    this.layout = device.createBindGroupLayout({
      label: 'de-ring layout',
      entries: [{
        binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' },
      }],
    });
    const texture = stats ?? device.createTexture({
      label: 'de-ring stub',
      size: [1, 1, 1],
      format: 'r32float',
      usage: GPUTextureUsage.TEXTURE_BINDING,
    });
    this.bindGroup = device.createBindGroup({
      label: 'de-ring bind group',
      layout: this.layout,
      entries: [{ binding: 0, resource: texture.createView() }],
    });
    this.constants = { DERING: stats ? 1 : 0 };
  }
}
