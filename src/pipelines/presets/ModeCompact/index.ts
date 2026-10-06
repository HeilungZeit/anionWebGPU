import { Downscale, FrameGate } from '../../helpers';
import { Anime4KPipeline, Anime4KPresetPipelineDescriptor, whenReady } from '../../interfaces';
import { CompactKernel, CompactModelData, CompactSR } from '../../compact';

export interface ModeCompactPresetPipelineDescriptor extends Anime4KPresetPipelineDescriptor {
  /** Веса и описание из конвертера anion-dl; в пакет не входят (лицензия NC). */
  model: CompactModelData;
  /** Ядро, по умолчанию `fast`. */
  kernel?: CompactKernel;
  /**
   * Не пересчитывать повторяющиеся кадры (FrameGate): на повторе сеть не
   * запускается, выход остаётся прошлым. По умолчанию выключено.
   */
  skipUnchanged?: boolean;
  /** Порог повтора в уровнях 8 бит, по умолчанию 0 — только точные повторы. */
  unchangedThreshold?: number;
}

/**
 * «Детали»: SRVGGNetCompact ×2 (AnimeJaNai V2 SuperUltraCompact), затем
 * уменьшение до цели, если она меньше 2× (Catmull-Rom). При цели ≤ 1.2×
 * исходника не делает ничего — как ModeArtCNN.
 */
export class ModeCompact implements Anime4KPipeline {
  pipelines: Anime4KPipeline[] = [];

  outputTexture: GPUTexture;

  ready: Promise<void>;

  /** Ворота повторов (skipUnchanged): счётчики — gate.readStats(). */
  gate?: FrameGate;

  constructor({
    device, inputTexture, nativeDimensions, targetDimensions, model, kernel,
    skipUnchanged = false, unchangedThreshold = 0,
  }: ModeCompactPresetPipelineDescriptor) {
    let currentTexture = inputTexture;
    if (targetDimensions.width > 1.2 * nativeDimensions.width
        && targetDimensions.height > 1.2 * nativeDimensions.height) {
      // Ворота — первыми: остальные звенья запускаются из их буфера.
      const gate = skipUnchanged
        ? new FrameGate({ device, inputTexture, threshold: unchangedThreshold })
        : undefined;
      if (gate) this.pipelines.push(gate);
      this.gate = gate;

      const upscale = new CompactSR({
        device, inputTexture: currentTexture, model, kernel, gate,
      });
      this.pipelines.push(upscale);
      currentTexture = upscale.getOutputTexture();

      if (targetDimensions.width < 2 * nativeDimensions.width
          && targetDimensions.height < 2 * nativeDimensions.height) {
        const downscale = new Downscale({
          device, inputTexture: currentTexture, targetDimensions, gate,
        });
        this.pipelines.push(downscale);
        currentTexture = downscale.getOutputTexture();
      }
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
