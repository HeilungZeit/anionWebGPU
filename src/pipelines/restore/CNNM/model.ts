// Сгенерировано conversion/cnn.py — не править руками.
// Источник: conversion/glsl/anime4k/Anime4K_Restore_CNN_M.glsl
// 8 слоёв mpv → 8 стадий.
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
  scale: 1,
  block: [1, 1],
  textures: 8,
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
      outputs: [4],
    },
    {
      wgsl: stage4,
      inputs: [4],
      outputs: [5],
    },
    {
      wgsl: stage5,
      inputs: [5],
      outputs: [6],
    },
    {
      wgsl: stage6,
      inputs: [6],
      outputs: [7],
    },
    {
      wgsl: stage7,
      inputs: [1, 2, 3, 4, 5, 6, 7],
      outputs: [],
      final: true,
    },
  ],
};

export default model;
