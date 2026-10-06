// Сгенерировано conversion/artcnn.py — не править руками.
// ArtCNN (Artoriuz, MIT): увеличение ×2 по яркости, цветность — билинейно.
import { CNNModelPipelineDescriptor } from '../../interfaces';
import { CNN } from '../../helpers/CNN';
import model from './model';

export class ArtCNNC4F32DS extends CNN {
  constructor({
    device, inputTexture, precision, deRing,
  }: CNNModelPipelineDescriptor) {
    super({
      device, inputTexture, model, name: 'ArtCNNC4F32DS', precision, deRing,
    });
  }
}
