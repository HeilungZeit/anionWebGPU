// Сгенерировано conversion/artcnn.py — не править руками.
// Источник: upstream-artcnn/GLSL/ArtCNN_C4F16_DS.glsl
// 8 шейдеров mpv → 7 стадий.
import stage0 from './shaders/stage0.wgsl';
import stage1 from './shaders/stage1.wgsl';
import stage2 from './shaders/stage2.wgsl';
import stage3 from './shaders/stage3.wgsl';
import stage4 from './shaders/stage4.wgsl';
import stage5 from './shaders/stage5.wgsl';
import stage6 from './shaders/stage6.wgsl';
import { CNNModel } from '../../helpers/CNN/model';

const model: CNNModel = {
  scale: 2,
  block: [1, 1],
  textures: 25,
  stages: [
    {
      wgsl: stage0,
      inputs: [0],
      outputs: [1, 2, 3, 4],
    },
    {
      wgsl: stage1,
      inputs: [1, 2, 3, 4],
      outputs: [5, 6, 7, 8],
    },
    {
      wgsl: stage2,
      inputs: [5, 6, 7, 8],
      outputs: [9, 10, 11, 12],
    },
    {
      wgsl: stage3,
      inputs: [9, 10, 11, 12],
      outputs: [13, 14, 15, 16],
    },
    {
      wgsl: stage4,
      inputs: [13, 14, 15, 16],
      outputs: [17, 18, 19, 20],
    },
    {
      wgsl: stage5,
      inputs: [17, 18, 19, 20],
      outputs: [21, 22, 23, 24],
    },
    {
      wgsl: stage6,
      inputs: [21, 1, 22, 2, 23, 3, 24, 4],
      outputs: [],
      final: true,
    },
  ],
};

export default model;
