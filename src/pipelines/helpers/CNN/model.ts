/**
 * Граф CNN-модели, который выпускает conversion/cnn.py.
 *
 * Текстуры нумеруются: 0 — вход пайплайна, 1..N — промежуточные rgba16float в
 * разрешении входа. Стадия читает `inputs` и пишет `outputs`; финальная
 * стадия вместо `outputs` пишет выход модели (вход × `scale`) и дополнительно
 * получает вход и билинейный сэмплер для сложения с исходником.
 */
export interface CNNStage {
  wgsl: string;
  inputs: number[];
  outputs: number[];
  final?: boolean;
  /** Стадия использует subgroupShuffle: нужна фича 'subgroups' устройства. */
  subgroups?: boolean;
}

export interface CNNModel {
  scale: number;
  /** Пикселей входа на поток [x, y]; группа — 8×8 потоков. */
  block: [number, number];
  textures: number;
  stages: CNNStage[];
}
