
import downscale_default from "./shaders/downscale.js";
//#region src/pipelines/helpers/Downscale/index.ts
/**
* Уменьшение до `targetDimensions`. По умолчанию — Catmull-Rom с ядром,
* растянутым на коэффициент уменьшения (без алиасинга); 'bilinear' — выборка
* в точке, как AutoDownscalePre в mpv и 1.0.0.
*/
var Downscale = class {
	outputTexture;
	steps;
	name;
	constructor({ device, inputTexture, targetDimensions, filter = "catmull-rom", name = "downscale" }) {
		this.name = name;
		this.outputTexture = device.createTexture({
			label: `${name} output texture`,
			size: [
				targetDimensions.width,
				targetDimensions.height,
				1
			],
			format: "rgba16float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
		});
		const module = device.createShaderModule({
			label: `${name} module`,
			code: downscale_default
		});
		const sampler = device.createSampler({
			magFilter: "linear",
			minFilter: "linear"
		});
		const step = (constants, input, output) => {
			const pipeline = device.createComputePipeline({
				label: `${name} pipeline`,
				layout: "auto",
				compute: {
					module,
					entryPoint: "computeMain",
					constants
				}
			});
			return {
				pipeline,
				bindGroup: device.createBindGroup({
					label: `${name} bind group`,
					layout: pipeline.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: input.createView()
						},
						{
							binding: 1,
							resource: sampler
						},
						{
							binding: 2,
							resource: output.createView()
						}
					]
				}),
				output
			};
		};
		if (filter === "bilinear") this.steps = [step({ FILTER: 0 }, inputTexture, this.outputTexture)];
		else {
			const middle = device.createTexture({
				label: `${name} middle texture`,
				size: [
					targetDimensions.width,
					inputTexture.height,
					1
				],
				format: "rgba16float",
				usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
			});
			this.steps = [step({
				FILTER: 1,
				AXIS: 0
			}, inputTexture, middle), step({
				FILTER: 1,
				AXIS: 1
			}, middle, this.outputTexture)];
		}
	}
	updateParam(param, value) {
		throw new Error(`${this.name} has no param`);
	}
	pass(encoder) {
		const pass = encoder.beginComputePass({ label: this.name });
		this.steps.forEach(({ pipeline, bindGroup, output }) => {
			pass.setPipeline(pipeline);
			pass.setBindGroup(0, bindGroup);
			pass.dispatchWorkgroups(Math.ceil(output.width / 8), Math.ceil(output.height / 8));
		});
		pass.end();
	}
	getOutputTexture() {
		return this.outputTexture;
	}
};
//#endregion
export { Downscale };
