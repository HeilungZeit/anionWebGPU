import { ClampStats, Downscale, FrameGate } from '../../helpers';
import { Anime4KPipeline, ModeCPresetPipelineDescriptor, whenReady } from '../../interfaces';
import {
  CNNx2M, DenoiseCNNx2L, DenoiseCNNx2M, DenoiseCNNx2VL,
} from '../../upscale';
import { planModeC } from '../chain';

export class ModeC implements Anime4KPipeline {
  pipelines: Anime4KPipeline[];

  outputTexture: GPUTexture;

  /** Все шейдеры цепочки скомпилированы (в фоне); до этого pass() не вызывать. */
  ready: Promise<void>;

  /** Ворота повторов (skipUnchanged): счётчики — gate.readStats(). */
  gate?: FrameGate;

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
    skipUnchanged = false,
    unchangedThreshold = 0,
  }: ModeCPresetPipelineDescriptor) {
    this.pipelines = [];
    const chain = planModeC(nativeDimensions, targetDimensions);
    let currentTexture = inputTexture;

    // Без увеличения цепочка пуста: зажим по статистике того же кадра — тождество.
    if (!chain.upscale1) {
      this.outputTexture = currentTexture;
      this.ready = Promise.resolve();
      return;
    }

    // Clamp Highlights: статистика по исходнику, зажим (deRing) — в последнем
    // звене, в разрешении выхода, как HOOK PREKERNEL в mpv.
    // Ворота — первыми: остальные звенья запускаются из их буфера.
    const gate = skipUnchanged
      ? new FrameGate({ device, inputTexture, threshold: unchangedThreshold })
      : undefined;
    if (gate) this.pipelines.push(gate);
    this.gate = gate;

    const stats = new ClampStats({ device, inputTexture, gate });
    this.pipelines.push(stats);
    let last = 'upscale1';
    if (chain.downscale2) last = 'downscale2';
    if (chain.downscale4) last = 'downscale4';
    if (chain.upscale2) last = 'upscale2';
    const deRing = (step: string) => (step === last ? stats.getOutputTexture() : undefined);

    const Denoise = { M: DenoiseCNNx2M, L: DenoiseCNNx2L, VL: DenoiseCNNx2VL }[denoiseModel];
    const upscale1 = new Denoise({
      device, inputTexture: currentTexture, precision, deRing: deRing('upscale1'), gate,
    });
    this.pipelines.push(upscale1);
    currentTexture = upscale1.getOutputTexture();

    if (chain.downscale2) {
      const downscale = new Downscale({
        device, inputTexture: currentTexture, targetDimensions, deRing: deRing('downscale2'), gate,
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
        gate,
      });
      this.pipelines.push(downscale);
      currentTexture = downscale.getOutputTexture();
    }

    if (chain.upscale2) {
      const upscale2 = new CNNx2M({
        device, inputTexture: currentTexture, precision, deRing: deRing('upscale2'), gate,
      });
      this.pipelines.push(upscale2);
      currentTexture = upscale2.getOutputTexture();
    }

    this.outputTexture = currentTexture;
    this.ready = whenReady(this.pipelines);
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
