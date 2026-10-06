import { Anime4KPipeline, DownscalePipelineDescriptor } from '../../interfaces';
import downscaleWGSL from './shaders/downscale.wgsl';

/**
 * Уменьшение до `targetDimensions`. По умолчанию — Catmull-Rom с ядром,
 * растянутым на коэффициент уменьшения (без алиасинга); 'bilinear' — выборка
 * в точке, как AutoDownscalePre в mpv и 1.0.0.
 */
export class Downscale implements Anime4KPipeline {
  outputTexture: GPUTexture;

  steps: { pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, output: GPUTexture }[];

  name: string;

  constructor({
    device,
    inputTexture,
    targetDimensions,
    filter = 'catmull-rom',
    name = 'downscale',
  }: DownscalePipelineDescriptor) {
    this.name = name;

    this.outputTexture = device.createTexture({
      label: `${name} output texture`,
      size: [targetDimensions.width, targetDimensions.height, 1],
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    });

    const module = device.createShaderModule({ label: `${name} module`, code: downscaleWGSL });
    const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    const step = (constants: Record<string, number>, input: GPUTexture, output: GPUTexture) => {
      const pipeline = device.createComputePipeline({
        label: `${name} pipeline`,
        layout: 'auto',
        compute: { module, entryPoint: 'computeMain', constants },
      });
      return {
        pipeline,
        bindGroup: device.createBindGroup({
          label: `${name} bind group`,
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: input.createView() },
            { binding: 1, resource: sampler },
            { binding: 2, resource: output.createView() },
          ],
        }),
        output,
      };
    };

    if (filter === 'bilinear') {
      this.steps = [step({ FILTER: 0 }, inputTexture, this.outputTexture)];
    } else {
      // Сепарабельно: x (ширина выхода × высота входа), затем y.
      const middle = device.createTexture({
        label: `${name} middle texture`,
        size: [targetDimensions.width, inputTexture.height, 1],
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      });
      this.steps = [
        step({ FILTER: 1, AXIS: 0 }, inputTexture, middle),
        step({ FILTER: 1, AXIS: 1 }, middle, this.outputTexture),
      ];
    }
  }

  updateParam(param: string, value: any): void {
    throw new Error(`${this.name} has no param`);
  }

  pass(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: this.name });
    this.steps.forEach(({ pipeline, bindGroup, output }) => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(output.width / 8), Math.ceil(output.height / 8));
    });
    pass.end();
  }

  getOutputTexture(): GPUTexture {
    return this.outputTexture;
  }
}
