// Сгенерировано conversion/cnn.py — не править руками.
// Источник: conversion/glsl/anime4k/Anime4K_Upscale_Denoise_CNN_x2_L.glsl
// 9 слоёв mpv → 4 стадий.
import stage0 from './shaders/stage0.wgsl';
import stage1 from './shaders/stage1.wgsl';
import stage2 from './shaders/stage2.wgsl';
import stage3 from './shaders/stage3.wgsl';
import { CNNModel } from '../../helpers/CNN/model';

const model: CNNModel = {
  scale: 2,
  block: [1, 1],
  textures: 4,
  packed: [1, 2, 3],
  stages: [
    {
      wgsl: stage0,
      inputs: [0],
      outputs: [1],
    },
    {
      wgsl: stage1,
      inputs: [1],
      outputs: [2],
    },
    {
      wgsl: stage2,
      inputs: [2],
      outputs: [3],
    },
    {
      wgsl: stage3,
      inputs: [3],
      outputs: [],
      final: true,
    },
  ],
};

export default model;
