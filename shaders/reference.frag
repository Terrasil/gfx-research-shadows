#version 460 core
#include "screen.glsl"
#include "boxes.glsl"
in vec2 vUv;
layout(location=0) out vec2 outSum;
uniform vec3 uDirection;
uniform float uWeight;
void main(){
    float depth=texture(uDepth,vUv).r;
    if(depth>=1.0){outSum=vec2(0);return;}
    vec3 normal=normalize(texture(uNormal,vUv).xyz);
    vec3 position=worldPosition(vUv,depth)+normal*uNormalBias;
    float w=uWeight*max(dot(normal,uDirection),0.0);
    outSum=vec2(blocked(position,uDirection)?0.0:w,w);
}
