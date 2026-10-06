/// <reference types="@webgpu/types" />
import { Anime4KPipeline, DownscalePipelineDescriptor } from "../../interfaces.js";
//#region src/pipelines/helpers/Downscale/index.d.ts
/**
 * Уменьшение до `targetDimensions`. По умолчанию — Catmull-Rom с ядром,
 * растянутым на коэффициент уменьшения (без алиасинга); 'bilinear' — выборка
 * в точке, как AutoDownscalePre в mpv и 1.0.0.
 */
export declare class Downscale implements Anime4KPipeline {
  outputTexture: GPUTexture;
  steps: {
    pipeline: GPUComputePipeline;
    bindGroup: GPUBindGroup;
    output: GPUTexture;
  }[];
  name: string;
  constructor({ device, inputTexture, targetDimensions, filter, name }: DownscalePipelineDescriptor);
  updateParam(param: string, value: any): void;
  pass(encoder: GPUCommandEncoder): void;
  getOutputTexture(): GPUTexture;
}
//#endregion