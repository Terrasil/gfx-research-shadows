#version 460 core
#include "warp.glsl"
#include "boxes.glsl"
in vec2 vUv;
uniform mat4 uInverseLightVp,uLightVp;
uniform int uIncludeReceiver;
uniform vec3 uReceiverNormal;
#ifdef PACKED_INTERVALS
#define INTERVAL_WRITER 1
#include "packed_depth.glsl"
bool boxEndpoints(vec3 o,vec3 d,vec3 lo,vec3 hi,out vec2 t){
    t=vec2(0,1e29);
    for(int k=0;k<3;++k){
        if(abs(d[k])<1e-8){if(o[k]<lo[k]||o[k]>hi[k])return false;}
        else{vec2 a=vec2(lo[k]-o[k],hi[k]-o[k])/d[k];t.x=max(t.x,min(a.x,a.y));t.y=min(t.y,max(a.x,a.y));}
    }
    return t.y>=t.x&&t.y>0.0;
}
#endif
void main(){
    vec2 p=unwarpPoint(vUv*2.0-1.0);
    vec4 a=uInverseLightVp*vec4(p,-1,1),b=uInverseLightVp*vec4(p,1,1);
    vec3 origin=a.xyz/a.w,direction=normalize(b.xyz/b.w-origin);
    float nearest=1e29;
#ifdef PACKED_INTERVALS
    float farthest=-1e29;
    for(int i=0;i<uBoxCount;++i){
        vec4 lo,hi;sceneBox(i,lo,hi);
        if(int(lo.w+0.5)==uDroppedGroup)continue;
        vec2 t;if(boxEndpoints(origin,direction,lo.xyz,hi.xyz,t)){
            nearest=min(nearest,t.x);farthest=max(farthest,t.y);
        }
    }
    if(farthest<0.0)discard;
    vec4 aClip=uLightVp*vec4(origin+direction*nearest,1),bClip=uLightVp*vec4(origin+direction*farthest,1);
    uint bits=(quantizeDepth(aClip.z/aClip.w*0.5+0.5)<<16u)|quantizeDepth(bClip.z/bClip.w*0.5+0.5);
    imageStore(uDepthIntervals,ivec3(ivec2(gl_FragCoord.xy),0),uvec4(bits));
    return;
#endif
    for(int i=0;i<uBoxCount;++i){
        vec4 lo,hi;sceneBox(i,lo,hi);
        int g=int(lo.w+0.5);if(g==uDroppedGroup)continue;
        float t;if(boxHit(origin,direction,lo.xyz,hi.xyz,t)){
            nearest=min(nearest,t);
        }
    }
    if(uIncludeReceiver!=0){
        float den=dot(uReceiverNormal,direction);
        if(abs(den)>1e-8){
            float t=-dot(uReceiverNormal,origin)/den;
            vec3 q=origin+direction*t;
            vec3 local=vec3(uReceiverNormal.y*q.x-uReceiverNormal.x*q.y,0,q.z);
            if(t>0.0&&abs(local.x)<=15.0&&abs(local.z)<=15.0)nearest=min(nearest,t);
        }
    }
    if(nearest>1e28)discard;
    vec4 clip=uLightVp*vec4(origin+direction*nearest,1);gl_FragDepth=clip.z/clip.w*0.5+0.5;
}
