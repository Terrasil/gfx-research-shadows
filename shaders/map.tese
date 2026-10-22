#version 460 core
#include "warp.glsl"
layout(triangles,equal_spacing,ccw) in;
in vec3 tcPosition[],tcNormal[];
flat out vec4 tePlane;
uniform mat4 uLightVp;
void main(){
    vec3 p=tcPosition[0]*gl_TessCoord.x+tcPosition[1]*gl_TessCoord.y+tcPosition[2]*gl_TessCoord.z;
    vec3 n=tcNormal[0];tePlane=vec4(n,-dot(n,tcPosition[0]));
    vec4 clip=uLightVp*vec4(p,1);
    clip.xy=warpPoint(clip.xy/clip.w)*clip.w;gl_Position=clip;
}
