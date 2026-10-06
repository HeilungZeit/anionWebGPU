// Сгенерировано conversion/artcnn.py — не править руками.
// Источник: upstream-artcnn/GLSL/ArtCNN_C4F32_DS.glsl
// 8 шейдеров mpv → 14 стадий.
import stage0 from './shaders/stage0.wgsl';
import stage1 from './shaders/stage1.wgsl';
import stage2 from './shaders/stage2.wgsl';
import stage3 from './shaders/stage3.wgsl';
import stage4 from './shaders/stage4.wgsl';
import stage5 from './shaders/stage5.wgsl';
import stage6 from './shaders/stage6.wgsl';
import stage7 from './shaders/stage7.wgsl';
import stage8 from './shaders/stage8.wgsl';
import stage9 from './shaders/stage9.wgsl';
import stage10 from './shaders/stage10.wgsl';
import stage11 from './shaders/stage11.wgsl';
import stage12 from './shaders/stage12.wgsl';
import stage13 from './shaders/stage13.wgsl';
import { CNNModel } from '../../helpers/CNN/model';

const model: CNNModel = {
  scale: 2,
  block: [1, 1],
  textures: 50,
  stages: [
    {
      wgsl: stage0,
      inputs: [0],
      outputs: [1, 2, 3, 4],
    },
    {
      wgsl: stage1,
      inputs: [0],
      outputs: [5, 6, 7, 8],
    },
    {
      wgsl: stage2,
      inputs: [1, 2, 3, 4, 5, 6, 7, 8],
      outputs: [9, 10, 11, 12],
    },
    {
      wgsl: stage3,
      inputs: [1, 2, 3, 4, 5, 6, 7, 8],
      outputs: [13, 14, 15, 16],
    },
    {
      wgsl: stage4,
      inputs: [9, 10, 11, 12, 13, 14, 15, 16],
      outputs: [17, 18, 19, 20],
    },
    {
      wgsl: stage5,
      inputs: [9, 10, 11, 12, 13, 14, 15, 16],
      outputs: [21, 22, 23, 24],
    },
    {
      wgsl: stage6,
      inputs: [17, 18, 19, 20, 21, 22, 23, 24],
      outputs: [25, 26, 27, 28],
    },
    {
      wgsl: stage7,
      inputs: [17, 18, 19, 20, 21, 22, 23, 24],
      outputs: [29, 30, 31, 32],
    },
    {
      wgsl: stage8,
      inputs: [25, 26, 27, 28, 29, 30, 31, 32],
      outputs: [33, 34, 35, 36],
    },
    {
      wgsl: stage9,
      inputs: [25, 26, 27, 28, 29, 30, 31, 32],
      outputs: [37, 38, 39, 40],
    },
    {
      wgsl: stage10,
      inputs: [33, 34, 35, 36, 37, 38, 39, 40],
      outputs: [41, 42, 43, 44],
    },
    {
      wgsl: stage11,
      inputs: [33, 34, 35, 36, 37, 38, 39, 40],
      outputs: [45, 46, 47, 48],
    },
    {
      wgsl: stage12,
      inputs: [41, 1, 42, 2, 43, 3, 44, 4, 45, 5, 46, 6, 47, 7, 48, 8],
      outputs: [49],
    },
    {
      wgsl: stage13,
      inputs: [49],
      outputs: [],
      final: true,
    },
  ],
};

export default model;
