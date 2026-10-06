import { ClampStats, Downscale } from '../../helpers';
import { Anime4KPipeline, ModeCPresetPipelineDescriptor } from '../../interfaces';
import {
  CNNx2M, DenoiseCNNx2L, DenoiseCNNx2M, DenoiseCNNx2VL,
} from '../../upscale';
import { planModeC } from '../chain';

export class ModeC implements Anime4KPipeline {
  pipelines: Anime4KPipeline[];

  outputTexture: GPUTexture;

  /**
   * Constructs a new instance of the preset class.
   *
   * @param {Anime4KPresetPipelineDescriptor} options - An object containing
   * the following properties:
   * @param {GPUDevice} options.device - The GPU device to use for the pipeline.
   * @param {GPUTexture} options.inputTexture - The input texture to process.
   * @param {Dimensions} options.nativeDimensions - The original dimensions of the input texture.
   * @param {Dimensions} options.targetDimensions - The target dimension for the output texture.
   * @param {DenoiseModelSize} [options.denoiseModel='VL'] - Size of the Upscale-Denoise model.
   * @param {CNNPrecision} [options.precision='f32'] - Arithmetic precision of the CNN stages.
   */
  constructor({
    device,
    inputTexture,
    nativeDimensions,
    targetDimensions,
    denoiseModel = 'VL',
    precision = 'f32',
  }: ModeCPresetPipelineDescriptor) {
    this.pipelines = [];
    const chain = planModeC(nativeDimensions, targetDimensions);
    let currentTexture = inputTexture;

    // Без увеличения цепочка пуста: зажим по статистике того же кадра — тождество.
    if (!chain.upscale1) {
      this.outputTexture = currentTexture;
      return;
    }

    // Clamp Highlights: статистика по исходнику, зажим (deRing) — в последнем
    // звене, в разрешении выхода, как HOOK PREKERNEL в mpv.
    const stats = new ClampStats({ device, inputTexture });
    this.pipelines.push(stats);
    let last = 'upscale1';
    if (chain.downscale2) last = 'downscale2';
    if (chain.downscale4) last = 'downscale4';
    if (chain.upscale2) last = 'upscale2';
    const deRing = (step: string) => (step === last ? stats.getOutputTexture() : undefined);

    const Denoise = { M: DenoiseCNNx2M, L: DenoiseCNNx2L, VL: DenoiseCNNx2VL }[denoiseModel];
    const upscale1 = new Denoise({
      device, inputTexture: currentTexture, precision, deRing: deRing('upscale1'),
    });
    this.pipelines.push(upscale1);
    currentTexture = upscale1.getOutputTexture();

    if (chain.downscale2) {
      const downscale = new Downscale({
        device, inputTexture: currentTexture, targetDimensions, deRing: deRing('downscale2'),
      });
      this.pipelines.push(downscale);
      currentTexture = downscale.getOutputTexture();
    }

    if (chain.downscale4) {
      const downscale = new Downscale({
        device,
        inputTexture: currentTexture,
        targetDimensions: {
          width: Math.ceil(targetDimensions.width / 2),
          height: Math.ceil(targetDimensions.height / 2),
        },
        deRing: deRing('downscale4'),
      });
      this.pipelines.push(downscale);
      currentTexture = downscale.getOutputTexture();
    }

    if (chain.upscale2) {
      const upscale2 = new CNNx2M({
        device, inputTexture: currentTexture, precision, deRing: deRing('upscale2'),
      });
      this.pipelines.push(upscale2);
      currentTexture = upscale2.getOutputTexture();
    }

    this.outputTexture = currentTexture;
  }

  updateParam(param: string, value: any): void {
    throw new Error('Preset has no param');
  }

  pass(encoder: GPUCommandEncoder): void {
    this.pipelines.forEach((pipeline) => pipeline.pass(encoder));
  }

  getOutputTexture(): GPUTexture {
    return this.outputTexture;
  }
}
