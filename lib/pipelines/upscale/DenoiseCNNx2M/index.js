
import { CNN } from "../../helpers/CNN/index.js";
import model from "./model.js";
//#region src/pipelines/upscale/DenoiseCNNx2M/index.ts
var DenoiseCNNx2M = class extends CNN {
	constructor({ device, inputTexture, precision, deRing }) {
		super({
			device,
			inputTexture,
			model,
			name: "DenoiseCNNx2M",
			precision,
			deRing
		});
	}
};
//#endregion
export { DenoiseCNNx2M };
