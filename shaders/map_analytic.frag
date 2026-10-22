#version 460 core
#include "warp.glsl"
#include "boxes.glsl"
in vec2 vUv;
uniform mat4 uInverseLightVp,uLightVp;
uniform int uGroup;
void main(){
    vec2 p=unwarpPoint(vUv*2.0-1.0);
    vec4 a=uInverseLightVp*vec4(p,-1,1),b=uInverseLightVp*vec4(p,1,1);
    vec3 origin=a.xyz/a.w,direction=normalize(b.xyz/b.w-origin);
    float nearest=1e29;
    for(int i=0;i<uBoxCount;++i){
        int g=int(uBoxLo[i].w+0.5);if(g==uDroppedGroup||(uGroup>=0&&g!=uGroup))continue;
        float t;if(boxHit(origin,direction,uBoxLo[i].xyz,uBoxHi[i].xyz,t))nearest=min(nearest,t);
    }
    if(nearest>1e28)discard;
    vec4 clip=uLightVp*vec4(origin+direction*nearest,1);gl_FragDepth=clip.z/clip.w*0.5+0.5;
}
