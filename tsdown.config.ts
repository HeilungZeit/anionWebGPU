import { defineConfig } from 'tsdown';

// Модуль на файл (unbundle): сборщик потребителя берёт только нужные пресеты
// и их шейдеры (sideEffects: false в package.json). Шейдеры .wgsl — текстом.
export default defineConfig({
  entry: ['src/index.ts', 'src/pipelines/presets/*/index.ts'],
  outDir: 'lib',
  unbundle: true,
  root: 'src',
  platform: 'browser',
  target: 'es2022',
  loader: { '.wgsl': 'text' },
  dts: true,
  // Типы WebGPU (GPUDevice, GPUTextureUsage…) приходят потребителю вместе с
  // пакетом, как в 1.0.0: tsc сам дописывал эту ссылку в .d.ts, tsdown — нет.
  banner: { dts: '/// <reference types="@webgpu/types" />' },
});
