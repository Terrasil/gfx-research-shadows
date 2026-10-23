#version 460 core
#include "screen.glsl"
in vec2 vUv;
layout(location=0) out vec2 outSum;
layout(binding=2) uniform sampler2DArray uMaps;
uniform mat4 uLightVp;
uniform vec3 uDirection;
uniform float uWeight,uDepthBias;
vec2 receiverGradient(vec3 p){
    vec3 dx=dFdx(p),dy=dFdy(p);
    float determinant=dx.x*dy.y-dx.y*dy.x;
    if(abs(determinant)<1e-12)return vec2(0);
    return vec2(dy.y*dx.z-dx.y*dy.z,dx.x*dy.z-dy.x*dx.z)/determinant;
}
void main(){
    float depth=texture(uDepth,vUv).r;
    vec3 rawNormal=texture(uNormal,vUv).xyz;
    vec3 normal=rawNormal/max(length(rawNormal),1e-12);
    vec3 p=worldPosition(vUv,depth)+normal*uNormalBias;
    vec4 clip=uLightVp*vec4(p,1);
    vec3 q=clip.xyz/clip.w*0.5+0.5;
    vec2 gradient=receiverGradient(q);
    if(depth>=1.0){outSum=vec2(0);return;}
    float visible=1.0;
    if(all(greaterThanEqual(q,vec3(0)))&&all(lessThan(q,vec3(1)))){
        ivec2 size=textureSize(uMaps,0).xy;
        ivec2 pixel=clamp(ivec2(q.xy*vec2(size)),ivec2(0),size-1);
        vec2 center=(vec2(pixel)+0.5)/vec2(size);
        float corrected=q.z+dot(gradient,center-q.xy)-uDepthBias;
        visible=corrected<=texelFetch(uMaps,ivec3(pixel,0),0).r?1.0:0.0;
    }
    float w=uWeight*max(dot(normal,uDirection),0.0);
    outSum=vec2(visible*w,w);
}
