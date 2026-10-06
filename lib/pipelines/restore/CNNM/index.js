
import { CNN } from "../../helpers/CNN/index.js";
import model from "./model.js";
//#region src/pipelines/restore/CNNM/index.ts
var CNNM = class extends CNN {
	constructor({ device, inputTexture, precision }) {
		super({
			device,
			inputTexture,
			model,
			name: "CNNM",
			precision
		});
	}
};
//#endregion
export { CNNM };
