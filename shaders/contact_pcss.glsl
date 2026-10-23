#include "contact_math.glsl"
uniform float uSharpPriority,uBlockerQuantile,uContactSupportWorld,uContactDistanceWorld;

// DPCF-inspired: one fixed-support PCF loop also measures blocker distance.
// Our two-sided curve is an explicit adaptation, not the author's unpublished equation.
float dpcfVisibility(vec2 p,float d,vec2 gradient,float angle,out vec4 diagnostic){
    float radiusWorld=min(uContactSupportWorld*uTanAngle,uMaxRadiusWorld);
    float radius=radiusWorld/uExtent,visible=0.0,sum=0.0;
    int count=0;
    for(int i=0;i<uFilterSamples;++i){
        float gap=depthGap(p+vogel(i,uFilterSamples,angle)*radius,d,p,gradient,0);
        bool blockedHere=gap>uDepthBias;
        visible+=blockedHere?0.0:1.0;
        if(blockedHere){sum+=gap;++count;}
    }
    visible/=float(uFilterSamples);
    float strength=count>0?contactStrength(sum/float(count)*uDepthRange,uContactDistanceWorld,uSharpPriority):0.0;
    diagnostic=vec4(radiusWorld,radiusWorld,strength,1);
    return mix(visible,contactCurve(visible),strength);
}

// CHS-inspired: fixed support, mean-gap-controlled Gaussian weights.
// Orthographic directional-light gaps replace ShadowFX's perspective depth ratio.
float chsVisibility(vec2 p,float d,vec2 gradient,float angle,out vec4 diagnostic){
    float radiusWorld=min(uContactSupportWorld*uTanAngle,uMaxRadiusWorld);
    float radius=radiusWorld/uExtent,sum=0.0;
    int count=0;
    for(int i=0;i<uBlockerSamples;++i){
        float gap=depthGap(p+vogel(i,uBlockerSamples,angle)*radius,d,p,gradient,0);
        if(gap>uDepthBias){sum+=gap;++count;}
    }
    diagnostic=vec4(radiusWorld,radiusWorld,0,1);
    if(count==0)return 1.0;
    float desiredWorld=sum/float(count)*uDepthRange*uTanAngle;
    float ratio=sqrt(clamp(desiredWorld/max(radiusWorld,1e-8),0.0,1.0));
    float visible=0.0,total=0.0;
    for(int i=0;i<uFilterSamples;++i){
        vec2 offset=vogel(i,uFilterSamples,angle);
        float weight=mix(1.0,chsWeight(dot(offset,offset),ratio),uSharpPriority);
        float gap=depthGap(p+offset*radius,d,p,gradient,0);
        visible+=(gap>uDepthBias?0.0:1.0)*weight;
        total+=weight;
    }
    diagnostic.z=uSharpPriority*(1.0-ratio);
    return visible/max(total,1e-12);
}

// Same search and PCF support as NDC PCSS; only the blocker statistic changes.
float statisticPcss(vec2 p,float d,vec2 gradient,float angle,bool quantile,out vec4 diagnostic){
    float scale=uDepthRange*uTanAngle/uExtent,maxRadius=uMaxRadiusWorld/uExtent;
    float searchRadius=min(max(d,0.0)*scale,maxRadius);
    float gaps[256],sum=0.0,logs=0.0;
    int count=0;
    for(int i=0;i<uBlockerSamples;++i){
        float gap=depthGap(p+vogel(i,uBlockerSamples,angle)*searchRadius,d,p,gradient,0);
        if(gap>uDepthBias){
            if(quantile)gaps[count]=gap;else logs+=log(max(gap,1e-8));
            ++count;sum+=gap;
        }
    }
    diagnostic=vec4(0,0,0,1);
    if(count==0)return 1.0;
    float meanGap=sum/float(count);
    float candidate=quantile?quantileGap(gaps,count,uBlockerQuantile):geometricGap(logs,count);
    float gap=preferredGap(meanGap,candidate,uSharpPriority);
    float meanRadius=min(meanGap*scale,maxRadius),radius=min(gap*scale,maxRadius);
    diagnostic=vec4(radius*uExtent,meanRadius*uExtent,clamp(1.0-radius/max(meanRadius,1e-8),0.0,1.0),1);
    return filtered(p,d,gradient,0,radius,angle,-1,0.0);
}
