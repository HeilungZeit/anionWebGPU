// Сгенерировано conversion/cnn.py — не править руками.
import { CNNModelPipelineDescriptor } from '../../interfaces';
import { CNN } from '../../helpers/CNN';
import model from './model';

export class CNNM extends CNN {
  constructor({
    device, inputTexture, precision, deRing,
  }: CNNModelPipelineDescriptor) {
    super({
      device, inputTexture, model, name: 'CNNM', precision, deRing,
    });
  }
}
