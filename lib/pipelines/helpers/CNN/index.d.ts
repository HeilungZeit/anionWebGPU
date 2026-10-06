/// <reference types="@webgpu/types" />
import { Anime4KPipeline, CNNPrecision } from "../../interfaces.js";
import { CNNModel, CNNStage } from "./model.js";
//#region src/pipelines/helpers/CNN/index.d.ts
export interface CNNPipelineDescriptor {
  device: GPUDevice;
  inputTexture: GPUTexture;
  model: CNNModel;
  name?: string;
  precision?: CNNPrecision;
}
interface CompiledStage {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
}
/**
 * Исполнитель CNN-модели из conversion/cnn.py: стадия — один compute-проход,
 * поток — один пиксель входа (финальная стадия x2 пишет 2×2 пикселя выхода).
 */
export declare class CNN implements Anime4KPipeline {
  name: string;
  precision: CNNPrecision;
  stages: CompiledStage[];
  outputTexture: GPUTexture;
  private block;
  private width;
  private height;
  constructor({ device, inputTexture, model, name, precision }: CNNPipelineDescriptor);
  updateParam(param: string, value: any): void;
  pass(encoder: GPUCommandEncoder): void;
  getOutputTexture(): GPUTexture;
}
//#endregion
export { CNNModel, CNNStage };