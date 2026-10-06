
import { CNN } from "../../helpers/CNN/index.js";
import model from "./model.js";
//#region src/pipelines/upscale/DenoiseCNNx2L/index.ts
var DenoiseCNNx2L = class extends CNN {
	constructor({ device, inputTexture, precision }) {
		super({
			device,
			inputTexture,
			model,
			name: "DenoiseCNNx2L",
			precision
		});
	}
};
//#endregion
export { DenoiseCNNx2L };
