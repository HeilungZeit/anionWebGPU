
//#region src/pipelines/deblur/DoG/shaders/lumination.wgsl
var lumination_default = "@group(0) @binding(0) var tex_in: texture_2d<f32>;\n@group(0) @binding(1) var tex_out: texture_storage_2d<rgba16float, write>;\n\nfn colorAt(x: u32, y: u32) -> vec4f {\n  return textureLoad(tex_in, vec2u(x, y), 0);\n}\n\n// Function to calculate luminance\nfn get_luma(rgba: vec4<f32>) -> f32 {\n    return dot(rgba, vec4<f32>(0.299, 0.587, 0.114, 0.0));\n}\n\n@compute\n@workgroup_size(8, 8)\nfn computeMain(@builtin(global_invocation_id) pixel: vec3u) {\n  // OOB check\n  let dim_in: vec2u = textureDimensions(tex_in);\n  let dim_out: vec2u = textureDimensions(tex_out);\n  if (pixel.x >= dim_out.x || pixel.y >= dim_out.y) {\n    return;\n  }\n\n  var color: vec4f = colorAt(pixel.x, pixel.y);\n  textureStore(tex_out, vec2u(pixel.x, pixel.y), vec4f(get_luma(color), 0, 0, 1));\n}\n";
//#endregion
export { lumination_default as default };
