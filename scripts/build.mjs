// Сборка ESM: файл на модуль, чтобы сборщик потребителя брал только нужные
// пресеты и их шейдеры (sideEffects: false в package.json).
//
// 1. tsc → lib/ (ESM + .d.ts).
// 2. Каждый .wgsl → lib/<путь>.wgsl.js с `export default "<код>"`.
// 3. Относительные импорты в lib/ дописываются до полного пути
//    (`./x` → `./x.js` или `./x/index.js`, `./a.wgsl` → `./a.wgsl.js`):
//    ESM без расширений не понимают Node и webpack в строгом режиме.
import { execFileSync } from 'node:child_process';
import {
  existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync,
} from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const src = join(root, 'src');
const lib = join(root, 'lib');

function walk(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    return statSync(path).isDirectory() ? walk(path) : [path];
  });
}

rmSync(lib, { recursive: true, force: true });
execFileSync(join(root, 'node_modules/.bin/tsc'), ['-p', join(root, 'tsconfig.json')], { stdio: 'inherit' });

const shaders = walk(src).filter((f) => f.endsWith('.wgsl'));
for (const file of shaders) {
  const out = join(lib, `${relative(src, file)}.js`);
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, `export default ${JSON.stringify(readFileSync(file, 'utf8'))};\n`);
}

function fullySpecified(fromFile, spec) {
  if (!spec.startsWith('.')) return spec;
  if (spec.endsWith('.wgsl')) {
    // tsc .wgsl не проверяет (declare module '*.wgsl'): без этой проверки
    // пропавший шейдер всплыл бы только в браузере.
    if (!existsSync(resolve(dirname(fromFile), `${spec}.js`))) {
      throw new Error(`${relative(root, fromFile)}: нет шейдера ${spec}`);
    }
    return `${spec}.js`;
  }
  const base = resolve(dirname(fromFile), spec);
  if (existsSync(`${base}.js`) || existsSync(`${base}.d.ts`)) return `${spec}.js`;
  if (existsSync(join(base, 'index.js')) || existsSync(join(base, 'index.d.ts'))) return `${spec}/index.js`;
  throw new Error(`${relative(root, fromFile)}: не найден модуль ${spec}`);
}

const specifier = /(\bfrom\s*|\bimport\s*\(\s*|\bimport\s+)(['"])([^'"]+)\2/g;
const emitted = walk(lib).filter((f) => /\.(js|d\.ts)$/.test(f) && !f.endsWith('.wgsl.js'));
for (const file of emitted) {
  const code = readFileSync(file, 'utf8');
  const next = code.replace(specifier, (m, head, q, spec) => `${head}${q}${fullySpecified(file, spec)}${q}`);
  if (next !== code) writeFileSync(file, next);
}

console.log(`lib/: ${emitted.length} модулей, ${shaders.length} шейдеров`);
