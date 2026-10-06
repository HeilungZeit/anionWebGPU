
//#region src/pipelines/helpers/Overlay/shaders/overlay2.wgsl
var overlay2_default = "@group(0) @binding(0) var mySampler: sampler;\n@group(0) @binding(1) var tex_diff: texture_2d<f32>;\n@group(0) @binding(2) var tex_origin: texture_2d<f32>;\n\n@fragment\nfn main(@location(0) fragUV: vec2<f32>) -> @location(0) vec4<f32> {\n    let color_bilinear: vec4f = textureSample(tex_origin, mySampler, fragUV);\n    let color_addon: vec4f = textureSample(tex_diff, mySampler, fragUV);\n    return clamp(color_bilinear + color_addon, vec4<f32>(0., 0., 0., 0.), vec4<f32>(1., 1., 1., 1.));\n}\n";
//#endregion
export { overlay2_default as default };
