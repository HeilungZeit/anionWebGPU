
//#region src/pipelines/helpers/FrameGate/shaders/decide.wgsl
var decide_default = "@group(0)@binding(0)var<storage,read_write> state:array<u32,8>;@group(0)@binding(1)var<storage,read> template_args:array<u32>;@group(0)@binding(2)var<storage,read_write> args:array<u32>;@compute @workgroup_size(1)fn computeMain(){let changed=state[0]> state[3]|| state[5]!=0u;let count=state[4]*3u;for(var i:u32=0u;i < count;i=i+1u){args[i]=select(0u,template_args[i],changed);}state[1]=state[1]+1u;state[2]=state[2]+select(1u,0u,changed);state[5]=0u;state[6]=state[0];}";
//#endregion
export { decide_default as default };
