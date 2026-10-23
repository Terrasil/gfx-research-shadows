#version 460 core
#include "warp.glsl"
layout(vertices=3) out;
in vec3 vsPosition[],vsNormal[];
out vec3 tcPosition[],tcNormal[];
uniform float uTessellation;
uniform int uAdaptiveTessellation,uMapResolution;
uniform mat4 uLightVp;
vec2 warpedPosition(vec3 p){vec4 clip=uLightVp*vec4(p,1);return warpPoint(clip.xy/clip.w);}
float edgeLevel(vec3 a,vec3 b){
    if(uAdaptiveTessellation==0||uWarp<=0.0)return uTessellation;
    vec2 wa=warpedPosition(a),wb=warpedPosition(b);
    float error=0.0;
    for(int i=1;i<=3;++i){
        float t=float(i)*0.25;
        error=max(error,length(warpedPosition(mix(a,b,t))-mix(wa,wb,t)));
    }
    // Curvature heuristic targeting quarter-texel chord error. The endpoints
    // are shared by adjacent patches; this is not a conservative error bound.
    return clamp(ceil(sqrt(error*float(uMapResolution)*2.0)),1.0,uTessellation);
}
void main(){
    tcPosition[gl_InvocationID]=vsPosition[gl_InvocationID];
    tcNormal[gl_InvocationID]=vsNormal[gl_InvocationID];
    if(gl_InvocationID==0){
        gl_TessLevelOuter[0]=edgeLevel(vsPosition[1],vsPosition[2]);
        gl_TessLevelOuter[1]=edgeLevel(vsPosition[2],vsPosition[0]);
        gl_TessLevelOuter[2]=edgeLevel(vsPosition[0],vsPosition[1]);
        gl_TessLevelInner[0]=max(gl_TessLevelOuter[0],max(gl_TessLevelOuter[1],gl_TessLevelOuter[2]));
    }
}
