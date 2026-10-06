// Сгенерировано conversion/cnn.py — не править руками.
import { CNNModelPipelineDescriptor } from '../../interfaces';
import { CNN } from '../../helpers/CNN';
import model from './model';

export class DenoiseCNNx2VL extends CNN {
  constructor({
    device, inputTexture, precision, deRing, gate,
  }: CNNModelPipelineDescriptor) {
    super({
      device, inputTexture, model, name: 'DenoiseCNNx2VL', precision, deRing, gate,
    });
  }
}
