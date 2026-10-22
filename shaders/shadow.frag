#version 460 core
#include "warp.glsl"
#include "screen.glsl"
#include "boxes.glsl"
#include "source.glsl"
in vec2 vUv;
layout(location=0) out float outVisibility;
layout(binding=2) uniform sampler2DArray uMaps;
uniform mat4 uLightVp;
uniform int uMethod,uMapResolution,uFilterSamples,uBlockerSamples,uRotateSamples;
uniform float uDepthRange,uExtent,uDepthBias,uMaxRadiusWorld,uFixedPcfRadius;

vec2 receiverGradient(vec3 p){
    vec3 dx=dFdx(p),dy=dFdy(p);
    float d=dx.x*dy.y-dx.y*dy.x;
    if(abs(d)<1e-12)return vec2(0);
    return vec2(dy.y*dx.z-dx.y*dy.z,dx.x*dy.z-dy.x*dx.z)/d;
}
vec2 vogel(int i,int count,float angle){
    float r=sqrt((float(i)+0.5)/float(count)),a=float(i)*2.39996322972865+angle;
    return vec2(cos(a),sin(a))*r;
}
float depthGap(vec2 p,float receiverDepth,vec2 origin,vec2 gradient,int layer){
    vec2 uv=warpPoint(p)*0.5+0.5;
    if(any(lessThan(uv,vec2(0)))||any(greaterThanEqual(uv,vec2(1))))return -1.0;
    ivec2 texel=clamp(ivec2(uv*float(uMapResolution)),ivec2(0),ivec2(uMapResolution-1));
    float d=texelFetch(uMaps,ivec3(texel,layer),0).r;
    if(d>=1.0)return -1.0;
    vec2 center=unwarpPoint((vec2(texel)+0.5)/float(uMapResolution)*2.0-1.0);
    return receiverDepth+dot(gradient,center-origin)-d;
}
float contact(vec2 p,float receiverDepth,vec2 gradient,int layer){
    vec2 uv=warpPoint(p)*0.5+0.5,texel=uv*float(uMapResolution)-0.5;
    vec2 w=fract(texel),base=floor(texel);
    float q[4];
    for(int y=0;y<2;++y)for(int x=0;x<2;++x){
        vec2 point=unwarpPoint((base+vec2(x,y)+0.5)/float(uMapResolution)*2.0-1.0);
        q[y*2+x]=depthGap(point,receiverDepth,p,gradient,layer)<=uDepthBias?1.0:0.0;
    }
    return mix(mix(q[0],q[1],w.x),mix(q[2],q[3],w.x),w.y);
}
float filtered(vec2 p,float d,vec2 gradient,int layer,float radius,float angle,int bin,float threshold){
    if(radius<0.5/float(uMapResolution)&&bin<0)return contact(p,d,gradient,layer);
    float visible=0.0;
    for(int i=0;i<uFilterSamples;++i){
        float gap=depthGap(p+vogel(i,uFilterSamples,angle)*radius,d,p,gradient,layer);
        bool blockedHere=gap>uDepthBias;
        if(bin==0)blockedHere=blockedHere&&gap<=threshold;
        if(bin==1)blockedHere=blockedHere&&gap>threshold;
        visible+=blockedHere?0.0:1.0;
    }
    return visible/float(uFilterSamples);
}
float pcss(vec2 p,float d,vec2 gradient,int layer,float angle,bool minimumGap,bool binned){
    float scale=uDepthRange*uTanAngle/uExtent;
    float maxRadius=uMaxRadiusWorld/uExtent;
    float searchRadius=min(max(d,0.0)*scale,maxRadius);
    float total=0.0,count=0.0,lo=1e20,hi=0.0;
    for(int i=0;i<uBlockerSamples;++i){
        float gap=depthGap(p+vogel(i,uBlockerSamples,angle)*searchRadius,d,p,gradient,layer);
        if(gap>uDepthBias){total+=gap;count+=1;lo=min(lo,gap);hi=max(hi,gap);}
    }
    if(count==0.0)return 1.0;
    if(binned){
        float threshold=(lo+hi)*0.5,sum0=0.0,sum1=0.0,n0=0.0,n1=0.0;
        for(int i=0;i<uBlockerSamples;++i){
            float gap=depthGap(p+vogel(i,uBlockerSamples,angle)*searchRadius,d,p,gradient,layer);
            if(gap>uDepthBias){if(gap<=threshold){sum0+=gap;n0+=1;}else{sum1+=gap;n1+=1;}}
        }
        float a=n0>0?filtered(p,d,gradient,layer,min(sum0/max(n0,1.0)*scale,maxRadius),angle,0,threshold):1.0;
        float b=n1>0?filtered(p,d,gradient,layer,min(sum1/max(n1,1.0)*scale,maxRadius),angle,1,threshold):1.0;
        return min(a,b);
    }
    float radius=min((minimumGap?lo:total/count)*scale,maxRadius);
    return filtered(p,d,gradient,layer,radius,angle,-1,0.0);
}
void main(){
    float depth=texture(uDepth,vUv).r;
    vec3 rawNormal=texture(uNormal,vUv).xyz;
    vec3 normal=rawNormal/max(length(rawNormal),1e-12);
    vec3 position=worldPosition(vUv,depth);
    vec3 shifted=position+normal*uNormalBias;
    vec4 clip=uLightVp*vec4(shifted,1);
    vec3 light=vec3(clip.xy/clip.w,clip.z/clip.w*0.5+0.5);
    vec2 gradient=receiverGradient(light);
    if(depth>=1.0){outVisibility=1.0;return;}
    if(uMethod==8){
        float lit=0.0,total=0.0;
        for(int i=0;i<16;++i){
            float w;vec3 direction=sourceDirection(i,w);w*=max(dot(normal,direction),0.0);
            total+=w;if(!blocked(shifted,direction))lit+=w;
        }
        outVisibility=total>1e-12?lit/total:1.0;return;
    }
    if(any(greaterThan(abs(light.xy),vec2(1)))||light.z<=0||light.z>=1){outVisibility=1.0;return;}
    if(uMethod==0){outVisibility=depthGap(light.xy,light.z,light.xy,gradient,0)<=uDepthBias?1.0:0.0;return;}
    if(uMethod==1){
        float sum=0.0;int radius=int(uFixedPcfRadius);
        for(int y=-radius;y<=radius;++y)for(int x=-radius;x<=radius;++x){
            vec2 offset=vec2(x,y)*2.0/float(uMapResolution);
            sum+=contact(light.xy+offset,light.z+dot(gradient,offset),gradient,0);
        }
        outVisibility=sum/float((radius*2+1)*(radius*2+1));return;
    }
    float angle=0.0;
    if(uRotateSamples!=0)angle=6.28318530718*fract(sin(dot(floor(gl_FragCoord.xy),vec2(12.9898,78.233)))*43758.5453);
    float a=pcss(light.xy,light.z,gradient,0,angle,uMethod==3,uMethod==7);
    if(uMethod==5||uMethod==6){
        float b=pcss(light.xy,light.z,gradient,1,angle,false,false);
        a=uMethod==5?min(a,b):a*b;
    }
    outVisibility=a;
}
