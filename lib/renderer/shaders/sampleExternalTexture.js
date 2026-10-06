
//#region src/renderer/shaders/sampleExternalTexture.wgsl
var sampleExternalTexture_default = "@group(0) @binding(1) var mySampler: sampler;\n@group(0) @binding(2) var myTexture: texture_2d<f32>;\n\n@fragment\nfn main(@location(0) fragUV : vec2f) -> @location(0) vec4f {\n  return textureSampleBaseClampToEdge(myTexture, mySampler, fragUV);\n}\n";
//#endregion
export { sampleExternalTexture_default as default };
