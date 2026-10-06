import { Anime4KPipeline } from '../../interfaces';
import { CNNModel } from './model';

export * from './model';

export interface CNNPipelineDescriptor {
  device: GPUDevice;
  inputTexture: GPUTexture;
  model: CNNModel;
  name?: string;
}

interface CompiledStage {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
}

/**
 * Исполнитель CNN-модели из conversion/cnn.py: стадия — один compute-проход,
 * поток — один пиксель входа (финальная стадия x2 пишет 2×2 пикселя выхода).
 */
export class CNN implements Anime4KPipeline {
  name: string;

  stages: CompiledStage[] = [];

  outputTexture: GPUTexture;

  private width: number;

  private height: number;

  constructor({
    device, inputTexture, model, name = 'cnn',
  }: CNNPipelineDescriptor) {
    this.name = name;
    this.width = inputTexture.width;
    this.height = inputTexture.height;

    const textures: GPUTexture[] = [inputTexture];
    for (let i = 1; i < model.textures; i += 1) {
      textures.push(device.createTexture({
        label: `${name}: feature ${i}`,
        size: [this.width, this.height, 1],
        format: 'rgba16float',
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

    model.stages.forEach((stage, n) => {
      const layoutEntries: GPUBindGroupLayoutEntry[] = [];
      const entries: GPUBindGroupEntry[] = [];
      const addTexture = (texture: GPUTexture) => {
        const binding = layoutEntries.length;
        layoutEntries.push({ binding, visibility: GPUShaderStage.COMPUTE, texture: {} });
        entries.push({ binding, resource: texture.createView() });
      };
      const addStorage = (texture: GPUTexture) => {
        const binding = layoutEntries.length;
        layoutEntries.push({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'rgba16float' },
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
      this.stages.push({
        pipeline: device.createComputePipeline({
          label: `${name}: stage ${n}`,
          layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
          compute: {
            module: device.createShaderModule({ label: `${name}: stage ${n}`, code: stage.wgsl }),
            entryPoint: 'computeMain',
          },
        }),
        bindGroup: device.createBindGroup({ label: `${name}: stage ${n}`, layout, entries }),
      });
    });
  }

  updateParam(param: string, value: any): void {
    throw new Error(`${this.name} has no param.`);
  }

  pass(encoder: GPUCommandEncoder): void {
    // Один проход: в compute-проходе каждый dispatch — своя область
    // синхронизации, запись стадии видна следующей.
    const pass = encoder.beginComputePass({ label: this.name });
    this.stages.forEach((stage) => {
      pass.setPipeline(stage.pipeline);
      pass.setBindGroup(0, stage.bindGroup);
      pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
    });
    pass.end();
  }

  getOutputTexture(): GPUTexture {
    return this.outputTexture;
  }
}
