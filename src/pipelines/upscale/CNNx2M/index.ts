// Сгенерировано conversion/cnn.py — не править руками.
import { CNNModelPipelineDescriptor } from '../../interfaces';
import { CNN } from '../../helpers/CNN';
import model from './model';

export class CNNx2M extends CNN {
  constructor({
    device, inputTexture, precision, deRing, gate,
  }: CNNModelPipelineDescriptor) {
    super({
      device, inputTexture, model, name: 'CNNx2M', precision, deRing, gate,
    });
  }
}
