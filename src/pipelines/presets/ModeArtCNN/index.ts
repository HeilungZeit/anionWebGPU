import { Downscale } from '../../helpers';
import { Anime4KPipeline, ModeArtCNNPresetPipelineDescriptor } from '../../interfaces';
import { ArtCNNC4F16DS, ArtCNNC4F32DS } from '../../artcnn';

/**
 * ArtCNN DS (denoise + sharpen) ×2 по яркости, затем уменьшение до цели,
 * если она меньше 2× (Catmull-Rom). При цели ≤ 1.2× исходника не делает
 * ничего — выход равен входу, как у ModeC.
 */
export class ModeArtCNN implements Anime4KPipeline {
  pipelines: Anime4KPipeline[] = [];

  outputTexture: GPUTexture;

  constructor({
    device,
    inputTexture,
    nativeDimensions,
    targetDimensions,
    model = 'C4F16',
    precision = 'f32',
  }: ModeArtCNNPresetPipelineDescriptor) {
    let currentTexture = inputTexture;
    if (targetDimensions.width > 1.2 * nativeDimensions.width
        && targetDimensions.height > 1.2 * nativeDimensions.height) {
      const ArtCNN = model === 'C4F32' ? ArtCNNC4F32DS : ArtCNNC4F16DS;
      const upscale = new ArtCNN({ device, inputTexture: currentTexture, precision });
      this.pipelines.push(upscale);
      currentTexture = upscale.getOutputTexture();

      if (targetDimensions.width < 2 * nativeDimensions.width
          && targetDimensions.height < 2 * nativeDimensions.height) {
        const downscale = new Downscale({ device, inputTexture: currentTexture, targetDimensions });
        this.pipelines.push(downscale);
        currentTexture = downscale.getOutputTexture();
      }
    }
    this.outputTexture = currentTexture;
  }

  updateParam(param: string, value: any): void {
    throw new Error('Preset has no param');
  }

  pass(encoder: GPUCommandEncoder): void {
    this.pipelines.forEach((pipeline) => pipeline.pass(encoder));
  }

  getOutputTexture(): GPUTexture {
    return this.outputTexture;
  }
}
