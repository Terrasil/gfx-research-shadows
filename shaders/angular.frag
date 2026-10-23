#version 460 core
#include "screen.glsl"
#include "boxes.glsl"
in vec2 vUv;
#if ARR_ACCUMULATE
layout(location=0) out vec2 outSum;
void writeVisibility(float lit,float total){outSum=vec2(lit,total);}
#else
layout(location=0) out float outVisibility;
void writeVisibility(float lit,float total){outVisibility=total>1e-12?clamp(lit/total,0.0,1.0):1.0;}
#endif
uniform vec3 uSun;
uniform float uTanAngle,uTriRadius,uTriRotation;
uniform int uRaySamples,uSignedMargin,uTriResultMasks;
#if ARR_TRIANGULATION
#include "tri_reconstruction.glsl"
#include "tri_geometry.glsl"
#else
layout(std430,binding=7) readonly buffer AngularDirections {vec4 raySamples[];};
#endif
void main(){
    float depth=texture(uDepth,vUv).r;
    if(depth>=1.0){writeVisibility(1.0,1.0);return;}
    vec3 normal=normalize(texture(uNormal,vUv).xyz);
    vec3 position=worldPosition(vUv,depth)+normal*uNormalBias;
#if ARR_TRIANGULATION
    vec2 points[TRI_COUNT];float values[TRI_COUNT];
    trianglePoints(uTriRadius,uTriRotation,points);
    if(uTriResultMasks!=0){
        vec2 sums=rayMaskSums(position,points,normal);
        writeVisibility(sums.x,sums.y);return;
    }
    if(uSignedMargin!=0)coneMargins(position,points,values);
    else{
        vec3 u=normalize(cross(vec3(0,0,1),uSun)),v=cross(uSun,u);
        for(int i=0;i<TRI_COUNT;++i){vec3 d=normalize(uSun+uTanAngle*(u*points[i].x+v*points[i].y));values[i]=blocked(position,d)?-1.0:1.0;}
    }
    writeVisibility(triangulatedVisibility(points,values,normal),1.0);
#else
    float lit=0.0,total=0.0;
    for(int i=0;i<uRaySamples;++i){
        vec4 sampleValue=raySamples[i];float w=sampleValue.w*max(dot(normal,sampleValue.xyz),0.0);
        total+=w;if(w>0.0&&!blocked(position,sampleValue.xyz))lit+=w;
    }
    writeVisibility(lit,total);
#endif
}
