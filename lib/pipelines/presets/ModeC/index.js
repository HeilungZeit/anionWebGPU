
import { Downscale } from "../../helpers/Downscale/index.js";
import { ClampHighlights } from "../../helpers/ClampHighlights/index.js";
import { CNNx2M } from "../../upscale/CNNx2M/index.js";
import { DenoiseCNNx2VL } from "../../upscale/DenoiseCNNx2VL/index.js";
import { DenoiseCNNx2M } from "../../upscale/DenoiseCNNx2M/index.js";
import { DenoiseCNNx2L } from "../../upscale/DenoiseCNNx2L/index.js";
//#region src/pipelines/presets/ModeC/index.ts
var ModeC = class {
	pipelines;
	outputTexture;
	/**
	* Constructs a new instance of the preset class.
	*
	* @param {Anime4KPresetPipelineDescriptor} options - An object containing
	* the following properties:
	* @param {GPUDevice} options.device - The GPU device to use for the pipeline.
	* @param {GPUTexture} options.inputTexture - The input texture to process.
	* @param {Dimensions} options.nativeDimensions - The original dimensions of the input texture.
	* @param {Dimensions} options.targetDimensions - The target dimension for the output texture.
	* @param {DenoiseModelSize} [options.denoiseModel='VL'] - Size of the Upscale-Denoise model.
	* @param {CNNPrecision} [options.precision='f32'] - Arithmetic precision of the CNN stages.
	*/
	constructor({ device, inputTexture, nativeDimensions, targetDimensions, denoiseModel = "VL", precision = "f32" }) {
		let curWidth = nativeDimensions.width;
		let curHeight = nativeDimensions.height;
		this.pipelines = [];
		let currentTexture = inputTexture;
		if (targetDimensions.width > 1.2 * curWidth && targetDimensions.height > 1.2 * curHeight) {
			const Denoise = {
				M: DenoiseCNNx2M,
				L: DenoiseCNNx2L,
				VL: DenoiseCNNx2VL
			}[denoiseModel];
			const upscale1 = new Denoise({
				device,
				inputTexture: currentTexture,
				precision
			});
			this.pipelines.push(upscale1);
			currentTexture = upscale1.getOutputTexture();
			curWidth *= 2;
			curHeight *= 2;
		}
		if (targetDimensions.width > 1.2 * nativeDimensions.width && targetDimensions.height > 1.2 * nativeDimensions.height && targetDimensions.width < 2 * nativeDimensions.width && targetDimensions.height < 2 * nativeDimensions.height) {
			const autoDownscalex2 = new Downscale({
				device,
				inputTexture: currentTexture,
				targetDimensions
			});
			this.pipelines.push(autoDownscalex2);
			currentTexture = autoDownscalex2.getOutputTexture();
			curWidth = targetDimensions.width;
			curHeight = targetDimensions.height;
		}
		if (targetDimensions.width > 2.4 * nativeDimensions.width && targetDimensions.height > 2.4 * nativeDimensions.height && targetDimensions.width < 4 * nativeDimensions.width && targetDimensions.height < 4 * nativeDimensions.height) {
			const autoDownscalex4 = new Downscale({
				device,
				inputTexture: currentTexture,
				targetDimensions: {
					width: Math.ceil(targetDimensions.width / 2),
					height: Math.ceil(targetDimensions.height / 2)
				}
			});
			this.pipelines.push(autoDownscalex4);
			currentTexture = autoDownscalex4.getOutputTexture();
			curWidth = Math.ceil(targetDimensions.width / 2);
			curHeight = Math.ceil(targetDimensions.height / 2);
		}
		if (targetDimensions.width > 1.2 * curWidth && targetDimensions.height > 1.2 * curHeight) {
			const upscale2 = new CNNx2M({
				device,
				inputTexture: currentTexture,
				precision
			});
			this.pipelines.push(upscale2);
			currentTexture = upscale2.getOutputTexture();
			curWidth *= 2;
			curHeight *= 2;
		}
		const clampHighlights = new ClampHighlights({
			device,
			inputTexture: currentTexture,
			statsTexture: inputTexture
		});
		this.pipelines.push(clampHighlights);
		currentTexture = clampHighlights.getOutputTexture();
		this.outputTexture = currentTexture;
	}
	updateParam(param, value) {
		throw new Error("Preset has no param");
	}
	pass(encoder) {
		for (let i = 0; i < this.pipelines.length; i += 1) this.pipelines[i].pass(encoder);
	}
	getOutputTexture() {
		return this.outputTexture;
	}
};
//#endregion
export { ModeC };
