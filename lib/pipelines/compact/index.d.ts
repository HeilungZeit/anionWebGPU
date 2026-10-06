/// <reference types="@webgpu/types" />
import { FrameGate } from "../helpers/FrameGate/index.js";
import { Anime4KPipeline } from "../interfaces.js";
import { CompactKernel, CompactLayerMeta, CompactModelMeta } from "./shaders.js";
//#region src/pipelines/compact/index.d.ts
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
/**
 * SRVGGNetCompact ×2 (AnimeJaNai V2 SuperUltraCompact, режим «Детали»):
 * слой — compute-проход, признаки между слоями — планарные буферы f16
 * (6 × vec4<f16> на пиксель), выход — текстура rgba16float ×2.
 */
export declare class CompactSR implements Anime4KPipeline {
  name: string;
  ready: Promise<void>;
  outputTexture: GPUTexture;
  private steps;
  private launch;
  constructor({ device, inputTexture, model, kernel, gate, name }: CompactPipelineDescriptor);
  updateParam(param: string, value: any): void;
  pass(encoder: GPUCommandEncoder): void;
  getOutputTexture(): GPUTexture;
}
//#endregion
export type { CompactKernel, CompactLayerMeta, CompactModelMeta };