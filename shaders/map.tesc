#version 460 core
layout(vertices=3) out;
in vec3 vsPosition[],vsNormal[];
out vec3 tcPosition[],tcNormal[];
uniform float uTessellation;
void main(){
    tcPosition[gl_InvocationID]=vsPosition[gl_InvocationID];
    tcNormal[gl_InvocationID]=vsNormal[gl_InvocationID];
    if(gl_InvocationID==0){
        gl_TessLevelOuter[0]=uTessellation;gl_TessLevelOuter[1]=uTessellation;
        gl_TessLevelOuter[2]=uTessellation;gl_TessLevelInner[0]=uTessellation;
    }
}
