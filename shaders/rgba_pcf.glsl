uniform int uPcfMode,uPcfSamples;
uniform float uPcfStrength,uPcfRadius,uPcfMaxRadius;
uniform int uPcfWorldUnits,uPcfMomentGap;
uniform float uPcfWorldRadius;
vec2 rgbaPcfDisk(int i,int count){
    float r=sqrt((float(i)+0.5)/float(count)),phi=float(i)*2.39996322973;
    return r*vec2(cos(phi),sin(phi));
}
bool rgbaPcfCovered(vec3 light){
    vec2 uv=warpPoint(light.xy)*0.5+0.5;
    return all(greaterThanEqual(uv,vec2(0)))&&all(lessThan(uv,vec2(1)))&&light.z>=0.0&&light.z<=1.0;
}
float rgbaPcfGap(vec2 p,vec3 origin,int channel,vec2 gradient){
    vec2 uv=warpPoint(p)*0.5+0.5;
    if(any(lessThan(uv,vec2(0)))||any(greaterThanEqual(uv,vec2(1))))return 0.0;
    ivec2 t=ivec2(uv*float(uMapResolution));float d=rgbaDepthAt(t,channel);
    if(d>=1.0)return 0.0;
    vec2 center=unwarpPoint((vec2(t)+0.5)/float(uMapResolution)*2.0-1.0);
    float gap=origin.z+dot(gradient,center-origin.xy)-d-uDepthBias*uBiasScale[channel];
    return max(gap,0.0)*uDepthRanges[channel];
}
#include "rgba_pcss.glsl"
float rgbaPcfEstimateGap(vec3 light,int channel,vec2 gradient,float cap){
    float gap=rgbaPcfGap(light.xy,light,channel,gradient);
    // Center blocker wins locally. A lit center searches a separate 8-point disk.
    if(gap<=0.0&&cap>0.0){
        float radius=min(max(light.z,0.0)*uDepthRanges[channel]*uTanAngle/uExtents[channel],cap);
        float logs=0.0,count=0.0;
        for(int i=0;i<8;++i){
            float g=rgbaPcfGap(light.xy+rgbaPcfDisk(i,8)*radius,light,channel,gradient);
            if(g>0.0){logs+=log(max(g,1e-8));count+=1.0;}
        }
        if(count<=0.0)return 0.0;
        gap=exp(logs/count);
    }
    return gap;
}
float rgbaPcfRadiusInfo(vec3 light,int channel,vec2 gradient,out float gap,out float footprint){
    gap=0.0;footprint=0.0;
    if(uPcfMode==0||uPcfStrength<=0.0||!rgbaPcfCovered(light))return 0.0;
    vec2 warped=warpPoint(light.xy);float step=2.0/float(uMapResolution);
    // Unwarped light-plane distance corresponding to one local map texel.
    footprint=min(length(unwarpPoint(warped+vec2(step,0))-light.xy),length(unwarpPoint(warped+vec2(0,step))-light.xy));
    if(uPcfMode==1){
        if(uPcfMomentGap!=0)gap=rgbaPcfEstimateGap(light,channel,gradient,footprint*uPcfMaxRadius);
        float radius=uPcfWorldUnits!=0?uPcfWorldRadius/uExtents[channel]:footprint*uPcfRadius;
        return radius*uPcfStrength;
    }
    if(uPcfMaxRadius<=0.0)return 0.0;
    float cap=footprint*uPcfMaxRadius;
    gap=uPcfMode==4?rgbaPcssBlockers(light,channel,gradient).x:rgbaPcfEstimateGap(light,channel,gradient,cap);
    return min(gap*uTanAngle/uExtents[channel]*uPcfStrength,cap);
}
float rgbaPcfRadius(vec3 light,int channel,vec2 gradient){
    float gap,footprint;return rgbaPcfRadiusInfo(light,channel,gradient,gap,footprint);
}
float rgbaPcfVisibilityAtRadius(vec3 light,int channel,vec2 gradient,float radius){
    if(radius<=1e-8)return visibility(light,channel,gradient);
    float sum=0.0,count=0.0;
    for(int i=0;i<uPcfSamples;++i){
        vec2 delta=rgbaPcfDisk(i,uPcfSamples)*radius;
        vec3 sampleLight=vec3(light.xy+delta,light.z+dot(gradient,delta));
        if(!rgbaPcfCovered(sampleLight))continue;
        sum+=visibility(sampleLight,channel,gradient);count+=1.0;
    }
    return count>0.0?sum/count:visibility(light,channel,gradient);
}
float rgbaChannelVisibilityAtRadius(vec3 light,int channel,vec2 gradient,float radius,float gap){
    if(uPcfMode==4&&uPcfStrength>0.0&&uPcfMaxRadius>0.0&&gap<=0.0)return 1.0;
    return rgbaPcfVisibilityAtRadius(light,channel,gradient,radius);
}
float rgbaPcfVisibility(vec3 light,int channel,vec2 gradient){
    float gap,footprint;float radius=rgbaPcfRadiusInfo(light,channel,gradient,gap,footprint);
    return rgbaChannelVisibilityAtRadius(light,channel,gradient,radius,gap);
}

float rgbaPcfSharpSupport(float base,float gap,float extent,float pixelFootprint){
    if((uPcfMode!=2&&uPcfMode!=4)||uPcfStrength<=0.0||uPcfMaxRadius<=0.0||gap<=0.0)return 0.0;
    float radius=gap*uTanAngle/extent;
    return (1.0-base)*(1.0-smoothstep(0.75,1.5,radius/max(pixelFootprint,1e-7)));
}
