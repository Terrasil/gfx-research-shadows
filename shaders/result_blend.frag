#version 460 core
layout(location=0) out float outVisibility;
layout(binding=0) uniform sampler2D uVisibility;
layout(binding=1) uniform sampler2D uDepth;
layout(binding=2) uniform sampler2D uNormal;
layout(binding=3) uniform sampler2D uGuide;
uniform mat4 uInverseVp;
uniform float uBlendStrength,uBlendSigma;
uniform int uBlendRadius,uContactEnabled,uInnerPenumbra;
vec3 blendPositionAtDepth(ivec2 pixel,float depth){
    vec2 uv=(vec2(pixel)+0.5)/vec2(textureSize(uDepth,0));
    vec4 p=uInverseVp*vec4(uv*2.0-1.0,depth*2.0-1.0,1);
    return p.xyz/p.w;
}
vec3 blendPosition(ivec2 pixel){return blendPositionAtDepth(pixel,texelFetch(uDepth,pixel,0).r);}
float spatialWeight(int offset){
    int q=abs(offset);
    return uBlendRadius==1?(q==0?2.0:1.0):(q==0?6.0:q==1?4.0:1.0);
}
void main(){
    ivec2 pixel=ivec2(gl_FragCoord.xy),size=textureSize(uDepth,0);
    float center=texelFetch(uVisibility,pixel,0).r;
    if(texelFetch(uDepth,pixel,0).r>=1.0||uBlendStrength<=0.0){outVisibility=center;return;}
    vec3 origin=blendPosition(pixel),normal=normalize(texelFetch(uNormal,pixel,0).xyz);
    // Use the center depth for the footprint, so a geometry discontinuity
    // cannot inflate the plane rejection tolerance or invalidate derivatives.
    float depth=texelFetch(uDepth,pixel,0).r;
    float tolerance=max(0.0001,0.25*max(length(blendPositionAtDepth(pixel+ivec2(1,0),depth)-origin),
                                        length(blendPositionAtDepth(pixel+ivec2(0,1),depth)-origin)));
    vec4 guide=texelFetch(uGuide,pixel,0);
    float sum=0.0,total=0.0;
    for(int y=-2;y<=2;++y)for(int x=-2;x<=2;++x){
        if(abs(x)>uBlendRadius||abs(y)>uBlendRadius)continue;
        ivec2 q=pixel+ivec2(x,y);
        if(any(lessThan(q,ivec2(0)))||any(greaterThanEqual(q,size))||texelFetch(uDepth,q,0).r>=1.0)continue;
        vec3 n=normalize(texelFetch(uNormal,q,0).xyz);
        if(dot(n,normal)<0.95)continue;
        vec3 delta=blendPosition(q)-origin;
        if(max(abs(dot(delta,normal)),abs(dot(delta,n)))>tolerance)continue;
        float value=texelFetch(uVisibility,q,0).r;
        vec4 neighbor=texelFetch(uGuide,q,0);
        float dv=(value-center)/max(uBlendSigma,0.001);
        float radiusDelta=abs(log((max(neighbor.x,0.0)+0.001)/(max(guide.x,0.0)+0.001)));
        float dc=uContactEnabled!=0?(neighbor.z-guide.z)/0.12:0.0;
        float weight=spatialWeight(x)*spatialWeight(y)*exp(-0.5*dv*dv-2.0*radiusDelta-0.5*dc*dc);
        sum+=weight*value;total+=weight;
    }
    float value=mix(center,total>1e-8?sum/total:center,uBlendStrength);
    if(uContactEnabled!=0)value=min(value,1.0-clamp(guide.z,0.0,1.0));
    if(uInnerPenumbra!=0&&center>=0.999999)value=center;
    outVisibility=clamp(value,0.0,1.0);
}
