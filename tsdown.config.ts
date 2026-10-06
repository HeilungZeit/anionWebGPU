import { readFile } from 'node:fs/promises';
import { defineConfig } from 'tsdown';

/**
 * Минификация WGSL: без комментариев и лишних пробелов. Значения чисел не
 * меняются (убирается только ведущий ноль) — математика та же. Пробелы
 * убираются только вокруг пунктуации, которая не склеивается в другой токен;
 * минус и угловые скобки не трогаем (`a - -b`, `vec2<vec4<f32> >`).
 */
function minifyWgsl(code: string): string {
  return code
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\/\/[^\n]*/g, '')
    .replace(/\s+/g, ' ')
    .replace(/ ?([(){}[\],;:=+*/]) ?/g, '$1')
    // 0.0124 → .0124: то же число, WGSL допускает запись без ведущего нуля.
    .replace(/(^|[^\w.])0\.(\d)/g, '$1.$2')
    .trim();
}

// Модуль на файл (unbundle): сборщик потребителя берёт только нужные пресеты
// и их шейдеры (sideEffects: false в package.json). Шейдеры .wgsl — строкой.
export default defineConfig({
  entry: ['src/index.ts', 'src/pipelines/presets/*/index.ts'],
  outDir: 'lib',
  unbundle: true,
  root: 'src',
  platform: 'browser',
  target: 'es2022',
  dts: true,
  // Типы WebGPU (GPUDevice, GPUTextureUsage…) приходят потребителю вместе с
  // пакетом, как в 1.0.0: tsc сам дописывал эту ссылку в .d.ts, tsdown — нет.
  banner: { dts: '/// <reference types="@webgpu/types" />' },
  plugins: [{
    name: 'wgsl-minify',
    async load(id) {
      if (!id.endsWith('.wgsl')) return null;
      const code = minifyWgsl(await readFile(id, 'utf8'));
      return { code: `export default ${JSON.stringify(code)};`, moduleType: 'js' };
    },
  }],
});
