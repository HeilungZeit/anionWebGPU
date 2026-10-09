/// <reference types="@webgpu/types" />
import { FrameGate } from "../../helpers/FrameGate/index.js";
import { Anime4KPipeline, Anime4KPresetPipelineDescriptor } from "../../interfaces.js";
import { CompactKernel } from "../../compact/shaders.js";
import { CompactModelData } from "../../compact/index.js";
//#region src/pipelines/presets/ModeCompact/index.d.ts
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
  /**
   * Цветность выхода: σ сглаживания цветности кадра, по умолчанию 2 — от
   * сети берётся только яркость (цветовые блоки сжатия сеть рисует пятнами).
   * 0 — цветность сети, как до Э12. См. CompactPipelineDescriptor.
   */
  chromaSigma?: number;
}
/**
 * «Детали»: SRVGGNetCompact ×2 (AnimeJaNai V2 SuperUltraCompact), затем
 * уменьшение до цели, если она меньше 2× (Catmull-Rom). При цели ≤ 1.2×
 * исходника не делает ничего — как ModeArtCNN.
 */
export declare class ModeCompact implements Anime4KPipeline {
  pipelines: Anime4KPipeline[];
  outputTexture: GPUTexture;
  ready: Promise<void>;
  /** Ворота повторов (skipUnchanged): счётчики — gate.readStats(). */
  gate?: FrameGate;
  constructor({ device, inputTexture, nativeDimensions, targetDimensions, model, kernel, skipUnchanged, unchangedThreshold, chromaSigma }: ModeCompactPresetPipelineDescriptor);
  updateParam(param: string, value: any): void;
  pass(encoder: GPUCommandEncoder): void;
  getOutputTexture(): GPUTexture;
}
//#endregion