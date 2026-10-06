
import { CNN } from "../../helpers/CNN/index.js";
import model from "./model.js";
//#region src/pipelines/upscale/CNNx2M/index.ts
var CNNx2M = class extends CNN {
	constructor({ device, inputTexture, precision }) {
		super({
			device,
			inputTexture,
			model,
			name: "CNNx2M",
			precision
		});
	}
};
//#endregion
export { CNNx2M };
