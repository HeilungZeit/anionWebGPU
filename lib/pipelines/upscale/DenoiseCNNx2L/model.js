
import stage0_default from "./shaders/stage0.js";
import stage1_default from "./shaders/stage1.js";
import stage2_default from "./shaders/stage2.js";
import stage3_default from "./shaders/stage3.js";
//#region src/pipelines/upscale/DenoiseCNNx2L/model.ts
const model = {
	scale: 2,
	block: [1, 1],
	textures: 7,
	stages: [
		{
			wgsl: stage0_default,
			inputs: [0],
			outputs: [1, 2]
		},
		{
			wgsl: stage1_default,
			inputs: [1, 2],
			outputs: [3, 4]
		},
		{
			wgsl: stage2_default,
			inputs: [3, 4],
			outputs: [5, 6]
		},
		{
			wgsl: stage3_default,
			inputs: [5, 6],
			outputs: [],
			final: true
		}
	]
};
//#endregion
export { model as default };
