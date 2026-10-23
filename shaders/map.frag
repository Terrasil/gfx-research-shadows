#version 460 core
#include "warp.glsl"
flat in vec4 tePlane;
uniform mat4 uLightVp,uInverseLightVp;
uniform int uMapResolution;
void main(){
    vec2 p=unwarpPoint(gl_FragCoord.xy/float(uMapResolution)*2.0-1.0);
    vec4 a=uInverseLightVp*vec4(p,-1,1),b=uInverseLightVp*vec4(p,1,1);
    vec3 origin=a.xyz/a.w,direction=normalize(b.xyz/b.w-origin);
    float den=dot(tePlane.xyz,direction);if(abs(den)<1e-8)discard;
    float t=-(dot(tePlane.xyz,origin)+tePlane.w)/den;
    vec4 clip=uLightVp*vec4(origin+direction*t,1);
    float d=clip.z/clip.w*0.5+0.5;
    gl_FragDepth=d;
}
