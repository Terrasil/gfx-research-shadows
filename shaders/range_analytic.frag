#version 460 core
#include "warp.glsl"
#include "boxes.glsl"
in vec2 vUv;
layout(location=0) out float outNearest;
layout(location=1) out float outFarthest;
uniform mat4 uLightVp,uInverseLightVp;
void main(){
    vec2 p=unwarpPoint(vUv*2.0-1.0);
    vec4 a=uInverseLightVp*vec4(p,-1,1),b=uInverseLightVp*vec4(p,1,1);
    vec3 origin=a.xyz/a.w,direction=normalize(b.xyz/b.w-origin);
    outNearest=1.0;outFarthest=0.0;
    for(int i=0;i<uBoxCount;++i){
        if(int(uBoxLo[i].w+0.5)==uDroppedGroup)continue;
        float t;
        if(!boxHit(origin,direction,uBoxLo[i].xyz,uBoxHi[i].xyz,t))continue;
        vec4 clip=uLightVp*vec4(origin+direction*t,1);
        float depth=clip.z/clip.w*0.5+0.5;
        if(depth<=0.0||depth>=1.0)continue;
        outNearest=min(outNearest,depth);outFarthest=max(outFarthest,depth);
    }
}
