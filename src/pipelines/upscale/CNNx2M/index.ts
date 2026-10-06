// Сгенерировано conversion/cnn.py — не править руками.
import { Anime4KPipelineDescriptor } from '../../interfaces';
import { CNN } from '../../helpers/CNN';
import model from './model';

export class CNNx2M extends CNN {
  constructor({ device, inputTexture }: Anime4KPipelineDescriptor) {
    super({
      device, inputTexture, model, name: 'CNNx2M',
    });
  }
}
