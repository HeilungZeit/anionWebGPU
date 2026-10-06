// Сгенерировано conversion/cnn.py — не править руками.
// Источник: upstream-anime4k/glsl/Upscale+Denoise/Anime4K_Upscale_Denoise_CNN_x2_L.glsl
// 9 слоёв mpv → 4 стадий.
import stage0 from './shaders/stage0.wgsl';
import stage1 from './shaders/stage1.wgsl';
import stage2 from './shaders/stage2.wgsl';
import stage3 from './shaders/stage3.wgsl';
import { CNNModel } from '../../helpers/CNN/model';

const model: CNNModel = {
  scale: 2,
  textures: 7,
  stages: [
    {
      wgsl: stage0,
      inputs: [0],
      outputs: [1, 2],
    },
    {
      wgsl: stage1,
      inputs: [1, 2],
      outputs: [3, 4],
    },
    {
      wgsl: stage2,
      inputs: [3, 4],
      outputs: [5, 6],
    },
    {
      wgsl: stage3,
      inputs: [5, 6],
      outputs: [],
      final: true,
    },
  ],
};

export default model;
