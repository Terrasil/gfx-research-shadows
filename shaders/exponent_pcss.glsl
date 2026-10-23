#include "exponent_math.glsl"
uniform float uExponent,uAutoYMin,uAutoYMax,uAutoNearWorld,uAutoFarWorld;
float exponentPcss(vec2 p,float d,vec2 gradient,float angle,out vec4 diagnostic){
    float scale=uDepthRange*uTanAngle/uExtent,maxRadius=uMaxRadiusWorld/uExtent;
    float searchRadius=min(max(d,0.0)*scale,maxRadius);
    float gaps[256];int count=0;
    for(int i=0;i<uBlockerSamples;++i){
        float gap=depthGap(p+vogel(i,uBlockerSamples,angle)*searchRadius,d,p,gradient,0);
        if(gap>uDepthBias)gaps[count++]=gap;
    }
    diagnostic=vec4(0);
    if(count==0)return 1.0;
    vec3 stats=exponentStatistics(gaps,count);
    bool inverse=uMethod==27||uMethod==29;
    bool automatic=uMethod==28||uMethod==29;
    float y=uExponent;
    if(automatic){
        float characteristicGap=exp(stats.y)*uDepthRange;
        y=distanceExponent(characteristicGap,uAutoNearWorld,uAutoFarWorld,uAutoYMin,uAutoYMax);
    }
    float candidate=exponentGap(gaps,count,stats,y,inverse);
    float gap=preferredGap(stats.x,candidate,uSharpPriority);
    float radius=min(gap*scale,maxRadius),meanRadius=min(stats.x*scale,maxRadius);
    diagnostic=vec4(radius*uExtent,y,clamp(1.0-radius/max(meanRadius,1e-8),0.0,1.0),1);
    return filtered(p,d,gradient,0,radius,angle,-1,0.0);
}
