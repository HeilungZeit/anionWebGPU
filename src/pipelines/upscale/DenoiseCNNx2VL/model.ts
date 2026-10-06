// Сгенерировано conversion/cnn.py — не править руками.
// Источник: conversion/glsl/anime4k/Anime4K_Upscale_Denoise_CNN_x2_VL.glsl
// 17 слоёв mpv → 8 стадий.
import stage0 from './shaders/stage0.wgsl';
import stage1 from './shaders/stage1.wgsl';
import stage2 from './shaders/stage2.wgsl';
import stage3 from './shaders/stage3.wgsl';
import stage4 from './shaders/stage4.wgsl';
import stage5 from './shaders/stage5.wgsl';
import stage6 from './shaders/stage6.wgsl';
import stage7 from './shaders/stage7.wgsl';
import { CNNModel } from '../../helpers/CNN/model';

const model: CNNModel = {
  scale: 2,
  block: [1, 1],
  textures: 15,
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
      outputs: [7, 8],
    },
    {
      wgsl: stage4,
      inputs: [7, 8],
      outputs: [9, 10],
    },
    {
      wgsl: stage5,
      inputs: [9, 10],
      outputs: [11, 12],
    },
    {
      wgsl: stage6,
      inputs: [11, 12],
      outputs: [13, 14],
    },
    {
      wgsl: stage7,
      inputs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
      outputs: [],
      final: true,
    },
  ],
};

export default model;
