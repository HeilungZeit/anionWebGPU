
import { DeRingEpilogue } from "../ClampHighlights/stats.js";
//#region src/pipelines/helpers/CNN/index.ts
const PRELUDE = {
	f32: "alias T4 = vec4f;\nalias M4 = mat4x4f;\nalias A4 = vec4f;\nalias S1 = f32;\n",
	f16: "enable f16;\nalias T4 = vec4h;\nalias M4 = mat4x4h;\nalias A4 = vec4h;\nalias S1 = f16;\n"
};
/**
* Исполнитель CNN-модели из conversion/cnn.py: стадия — один compute-проход,
* поток — один пиксель входа (финальная стадия x2 пишет 2×2 пикселя выхода).
*/
var CNN = class {
	name;
	precision;
	stages = [];
	outputTexture;
	block;
	width;
	height;
	constructor({ device, inputTexture, model, name = "cnn", precision = "f32", deRing }) {
		this.name = name;
		if (precision === "f16" && !device.features.has("shader-f16")) throw new Error(`${name}: precision 'f16' requires a device with 'shader-f16'.`);
		this.precision = precision;
		this.block = model.block;
		this.width = inputTexture.width;
		this.height = inputTexture.height;
		const textures = [inputTexture];
		for (let i = 1; i < model.textures; i += 1) textures.push(device.createTexture({
			label: `${name}: feature ${i}`,
			size: [
				this.width,
				this.height,
				1
			],
			format: "rgba16float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
		}));
		this.outputTexture = device.createTexture({
			label: `${name}: output`,
			size: [
				this.width * model.scale,
				this.height * model.scale,
				1
			],
			format: "rgba16float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
		});
		const sampler = device.createSampler({
			magFilter: "linear",
			minFilter: "linear"
		});
		const epilogue = new DeRingEpilogue(device, deRing);
		model.stages.forEach((stage, n) => {
			const layoutEntries = [];
			const entries = [];
			const addTexture = (texture) => {
				const binding = layoutEntries.length;
				layoutEntries.push({
					binding,
					visibility: GPUShaderStage.COMPUTE,
					texture: {}
				});
				entries.push({
					binding,
					resource: texture.createView()
				});
			};
			const addStorage = (texture) => {
				const binding = layoutEntries.length;
				layoutEntries.push({
					binding,
					visibility: GPUShaderStage.COMPUTE,
					storageTexture: {
						access: "write-only",
						format: "rgba16float"
					}
				});
				entries.push({
					binding,
					resource: texture.createView()
				});
			};
			stage.inputs.forEach((i) => addTexture(textures[i]));
			if (stage.final) {
				addTexture(inputTexture);
				const binding = layoutEntries.length;
				layoutEntries.push({
					binding,
					visibility: GPUShaderStage.COMPUTE,
					sampler: {}
				});
				entries.push({
					binding,
					resource: sampler
				});
				addStorage(this.outputTexture);
			} else stage.outputs.forEach((i) => addStorage(textures[i]));
			const layout = device.createBindGroupLayout({
				label: `${name}: stage ${n} layout`,
				entries: layoutEntries
			});
			const final = Boolean(stage.final);
			this.stages.push({
				pipeline: device.createComputePipeline({
					label: `${name}: stage ${n}`,
					layout: device.createPipelineLayout({ bindGroupLayouts: final ? [layout, epilogue.layout] : [layout] }),
					compute: {
						module: device.createShaderModule({
							label: `${name}: stage ${n}`,
							code: (stage.subgroups ? "enable subgroups;\n" : "") + PRELUDE[this.precision] + (final ? DeRingEpilogue.wgsl : "") + stage.wgsl
						}),
						entryPoint: "computeMain",
						constants: final ? epilogue.constants : {}
					}
				}),
				bindGroup: device.createBindGroup({
					label: `${name}: stage ${n}`,
					layout,
					entries
				}),
				deRing: final ? epilogue.bindGroup : void 0
			});
		});
	}
	updateParam(param, value) {
		throw new Error(`${this.name} has no param.`);
	}
	pass(encoder) {
		const pass = encoder.beginComputePass({ label: this.name });
		this.stages.forEach((stage) => {
			pass.setPipeline(stage.pipeline);
			pass.setBindGroup(0, stage.bindGroup);
			if (stage.deRing) pass.setBindGroup(1, stage.deRing);
			pass.dispatchWorkgroups(Math.ceil(this.width / (8 * this.block[0])), Math.ceil(this.height / (8 * this.block[1])));
		});
		pass.end();
	}
	getOutputTexture() {
		return this.outputTexture;
	}
};
//#endregion
export { CNN };
