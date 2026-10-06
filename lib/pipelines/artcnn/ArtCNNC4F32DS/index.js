
import { CNN } from "../../helpers/CNN/index.js";
import model from "./model.js";
//#region src/pipelines/artcnn/ArtCNNC4F32DS/index.ts
var ArtCNNC4F32DS = class extends CNN {
	constructor({ device, inputTexture, precision }) {
		super({
			device,
			inputTexture,
			model,
			name: "ArtCNNC4F32DS",
			precision
		});
	}
};
//#endregion
export { ArtCNNC4F32DS };
