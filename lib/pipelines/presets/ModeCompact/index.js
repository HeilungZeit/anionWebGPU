
import { whenReady } from "../../interfaces.js";
import { FrameGate } from "../../helpers/FrameGate/index.js";
import { Downscale } from "../../helpers/Downscale/index.js";
import { CompactSR } from "../../compact/index.js";
//#region src/pipelines/presets/ModeCompact/index.ts
/**
* «Детали»: SRVGGNetCompact ×2 (AnimeJaNai V2 SuperUltraCompact), затем
* уменьшение до цели, если она меньше 2× (Catmull-Rom). При цели ≤ 1.2×
* исходника не делает ничего — как ModeArtCNN.
*/
var ModeCompact = class {
	pipelines = [];
	outputTexture;
	ready;
	/** Ворота повторов (skipUnchanged): счётчики — gate.readStats(). */
	gate;
	constructor({ device, inputTexture, nativeDimensions, targetDimensions, model, kernel, skipUnchanged = false, unchangedThreshold = 0 }) {
		let currentTexture = inputTexture;
		if (targetDimensions.width > 1.2 * nativeDimensions.width && targetDimensions.height > 1.2 * nativeDimensions.height) {
			const gate = skipUnchanged ? new FrameGate({
				device,
				inputTexture,
				threshold: unchangedThreshold
			}) : void 0;
			if (gate) this.pipelines.push(gate);
			this.gate = gate;
			const upscale = new CompactSR({
				device,
				inputTexture: currentTexture,
				model,
				kernel,
				gate
			});
			this.pipelines.push(upscale);
			currentTexture = upscale.getOutputTexture();
			if (targetDimensions.width < 2 * nativeDimensions.width && targetDimensions.height < 2 * nativeDimensions.height) {
				const downscale = new Downscale({
					device,
					inputTexture: currentTexture,
					targetDimensions,
					gate
				});
				this.pipelines.push(downscale);
				currentTexture = downscale.getOutputTexture();
			}
		}
		this.outputTexture = currentTexture;
		this.ready = whenReady(this.pipelines);
	}
	updateParam(param, value) {
		throw new Error("Preset has no param");
	}
	pass(encoder) {
		this.pipelines.forEach((pipeline) => pipeline.pass(encoder));
	}
	getOutputTexture() {
		return this.outputTexture;
	}
};
//#endregion
export { ModeCompact };
