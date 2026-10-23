#version 460 core
#include "warp.glsl"
#include "screen.glsl"
in vec2 vUv;
layout(location=0) out float outVisibility;
layout(location=1) out vec4 outDiagnostic;
layout(binding=2) uniform sampler2DArray uMaps;
uniform mat4 uLightVp;
uniform vec3 uSun;
uniform int uMapResolution,uFilterSamples,uBlockerSamples,uRotateSamples,uDpcfTaps,uSamplePattern,uSampleSeed;
const int uMethod=SHADOW_METHOD;
const int uChsKernel=CHS_FILTER_SIZE;
uniform float uTanAngle,uDepthRange,uExtent,uDepthBias,uMaxRadiusWorld,uFixedPcfRadius;
uniform float uDpcfRadius,uDpcfRatioScale,uChsSunSize;
#if (SHADOW_METHOD == 1 || SHADOW_METHOD == 2 || SHADOW_METHOD == 4) && SHADOW_SMOOTHING > 0
#include "smooth_depth.glsl"
#endif
vec2 receiverNormalGradient(vec3 normal){
    vec3 tangent=normalize(cross(vec3(0,0,1),uSun)),bitangent=cross(uSun,tangent);
    float denominator=dot(normal,uSun);if(abs(denominator)<1e-5)return vec2(0);
    return (uExtent/uDepthRange)*vec2(dot(normal,tangent),dot(normal,bitangent))/denominator;
}
#include "pcss_sampling.glsl"
float depthGap(vec2 p,float receiverDepth,vec2 origin,vec2 gradient){
    vec2 uv=warpPoint(p)*0.5+0.5;
    if(any(lessThan(uv,vec2(0)))||any(greaterThanEqual(uv,vec2(1))))return -1.0;
    ivec2 texel=clamp(ivec2(uv*float(uMapResolution)),ivec2(0),ivec2(uMapResolution-1));
    float d=texelFetch(uMaps,ivec3(texel,0),0).r;if(d>=1.0)return -1.0;
    vec2 center=unwarpPoint((vec2(texel)+0.5)/float(uMapResolution)*2.0-1.0);
    return receiverDepth+dot(gradient,center-origin)-d;
}
float contactUv(vec2 uv,float depth,vec2 origin,vec2 gradient){
    vec2 texel=uv*float(uMapResolution)-0.5,w=fract(texel),base=floor(texel);float q[4];
    for(int y=0;y<2;++y)for(int x=0;x<2;++x){
        vec2 p=unwarpPoint((base+vec2(x,y)+0.5)/float(uMapResolution)*2.0-1.0);
        q[y*2+x]=depthGap(p,depth,origin,gradient)>uDepthBias?0.0:1.0;
    }
    return mix(mix(q[0],q[1],w.x),mix(q[2],q[3],w.x),w.y);
}
float filtered(vec2 p,float depth,vec2 gradient,float radius,float angle,uint filterKey){
    vec2 warped=warpPoint(p);float step=2.0/float(uMapResolution);
    float footprint=0.5*min(length(unwarpPoint(warped+vec2(step,0))-p),length(unwarpPoint(warped+vec2(0,step))-p));
#if (SHADOW_METHOD == 2 || SHADOW_METHOD == 4) && SHADOW_SMOOTHING > 0
    if(radius<=footprint*0.5)return smoothVisibility(warped*0.5+0.5,depth,p,gradient);
    float visible=0.0;
    for(int i=0;i<uFilterSamples;++i){
        vec2 uv=warpPoint(p+diskSample(i,uFilterSamples,angle,uSamplePattern,filterKey)*radius)*0.5+0.5;
        visible+=smoothVisibility(uv,depth,p,gradient);
    }
    visible/=float(uFilterSamples);
    if(radius<footprint*1.5){
        float contact=smoothVisibility(warped*0.5+0.5,depth,p,gradient);
        return mix(contact,visible,smoothstep(footprint*0.5,footprint*1.5,radius));
    }
    return visible;
#else
    if(radius<footprint)return contactUv(warped*0.5+0.5,depth,p,gradient);
    float visible=0.0;
    for(int i=0;i<uFilterSamples;++i)
        visible+=depthGap(p+diskSample(i,uFilterSamples,angle,uSamplePattern,filterKey)*radius,depth,p,gradient)>uDepthBias?0.0:1.0;
    return visible/float(uFilterSamples);
#endif
}
float pcss(vec3 light,vec2 gradient,float angle,bool geometric){
    float scale=uDepthRange*uTanAngle/uExtent,maxRadius=uMaxRadiusWorld/uExtent;
    float searchRadius=min(max(light.z,0.0)*scale,maxRadius),sum=0.0,logs=0.0,count=0.0;
    uint blockerKey=0u,filterKey=0u;
    if(uSamplePattern!=0){
        uvec2 pixel=uvec2(gl_FragCoord.xy);
        blockerKey=diskSeed(pixel,uint(uSampleSeed),0u);
        filterKey=diskSeed(pixel,uint(uSampleSeed),1u);
    }
    for(int i=0;i<uBlockerSamples;++i){
#if (SHADOW_METHOD == 2 || SHADOW_METHOD == 4) && (SHADOW_SMOOTHING == 2 || SHADOW_SMOOTHING == 3)
        vec2 uv=warpPoint(light.xy+diskSample(i,uBlockerSamples,angle,uSamplePattern,blockerKey)*searchRadius)*0.5+0.5;
        vec3 stats=smoothBlockers(uv,light.z,light.xy,gradient);
        logs+=stats.x;sum+=stats.y;count+=stats.z;
#else
        float gap=depthGap(light.xy+diskSample(i,uBlockerSamples,angle,uSamplePattern,blockerKey)*searchRadius,light.z,light.xy,gradient);
        if(gap>uDepthBias){sum+=gap;if(geometric)logs+=log(max(gap,1e-8));++count;}
#endif
    }
    if(count<=0.0)return 1.0;
    float mean=sum/count,gap=geometric?min(mean,exp(logs/count)):mean;
    float radius=min(gap*scale,maxRadius);
    outDiagnostic=vec4(radius*uExtent,min(mean*scale,maxRadius)*uExtent,0,1);
    return filtered(light.xy,light.z,gradient,radius,angle,filterKey);
}
#if SHADOW_METHOD == 5
#include "dpcf.glsl"
#elif SHADOW_METHOD == 6
#include "chs.glsl"
#endif
void main(){
    outDiagnostic=vec4(0);
    float depth=texture(uDepth,vUv).r;
    if(depth>=1.0){outVisibility=1.0;return;}
    vec3 normal=normalize(texture(uNormal,vUv).xyz);
    vec3 position=worldPosition(vUv,depth)+normal*uNormalBias;
    vec4 clip=uLightVp*vec4(position,1);
    vec3 light=vec3(clip.xy/clip.w,clip.z/clip.w*0.5+0.5);
    if(any(greaterThan(abs(light.xy),vec2(1)))||light.z<=0||light.z>=1){outVisibility=1.0;return;}
    vec2 gradient=receiverNormalGradient(normal),uv=warpPoint(light.xy)*0.5+0.5;
#if SHADOW_METHOD == 0
    outVisibility=depthGap(light.xy,light.z,light.xy,gradient)<=uDepthBias?1.0:0.0;
#elif SHADOW_METHOD == 1
        float sum=0.0;int radius=int(uFixedPcfRadius+0.5);
        for(int y=-radius;y<=radius;++y)for(int x=-radius;x<=radius;++x){
#if SHADOW_SMOOTHING == 0
            vec2 tap=uv+vec2(x,y)/float(uMapResolution);
            if(any(lessThan(tap,vec2(0)))||any(greaterThanEqual(tap,vec2(1)))){sum+=1.0;continue;}
            vec2 p=unwarpPoint(tap*2.0-1.0);
            sum+=depthGap(p,light.z,light.xy,gradient)>uDepthBias?0.0:1.0;
#elif SHADOW_SMOOTHING == 1
            sum+=contactUv(uv+vec2(x,y)/float(uMapResolution),light.z,light.xy,gradient);
#else
            sum+=smoothVisibility(uv+vec2(x,y)/float(uMapResolution),light.z,light.xy,gradient);
#endif
        }
        outVisibility=sum/float((radius*2+1)*(radius*2+1));return;
#else
    float angle=0.0;
    if(uRotateSamples!=0||uMethod==5)angle=6.28318530718*fract(52.9829189*fract(dot(gl_FragCoord.xy,vec2(0.06711056,0.00583715))));
#if SHADOW_METHOD == 2 || SHADOW_METHOD == 4
    outVisibility=pcss(light,gradient,angle,uMethod==4);
#else
    float lightScale=uTanAngle/tan(radians(1.5));
#if SHADOW_METHOD == 5
        vec2 t=1.0+uWarp*abs(light.xy-uFocus);
        vec2 slope=2.0*gradient*t*t/warpGain();
        float radius=uDpcfRadius*lightScale;
        outVisibility=dpcfFilter(vec3(uv,light.z-uDepthBias),slope,radius,uDpcfTaps,uDpcfRatioScale,angle);
        outDiagnostic=vec4(radius*2.0*uExtent/float(uMapResolution),0,1.0-outVisibility,1);return;
#elif SHADOW_METHOD == 6
        float sunSize=uChsSunSize*lightScale;
        outVisibility=chsFilter(vec3(uv,light.z),uDepthBias,sunSize*sunSize,uChsKernel);
        outDiagnostic=vec4(float(uChsKernel/2)*2.0*uExtent/float(uMapResolution),0,1.0-outVisibility,1);return;
#endif
#endif
#endif
}
