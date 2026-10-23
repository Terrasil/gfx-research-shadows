#version 460 core
in vec2 vUv;
layout(location=0) out float outVisibility;
layout(binding=0) uniform sampler2D uAccum;
layout(binding=1) uniform sampler2D uDepth;
layout(binding=2) uniform sampler2D uNormal;
uniform mat4 uInverseVp;
uniform int uArrSmoothing;
uniform float uArrSmoothingRadius;
vec3 positionAt(ivec2 pixel){
    vec2 uv=(vec2(pixel)+0.5)/vec2(textureSize(uDepth,0));
    vec4 p=uInverseVp*vec4(uv*2.0-1.0,texelFetch(uDepth,pixel,0).r*2.0-1.0,1);
    return p.xyz/p.w;
}
float visibilityAt(ivec2 pixel){vec2 q=texelFetch(uAccum,pixel,0).rg;return q.y>1e-12?clamp(q.x/q.y,0.0,1.0):1.0;}
vec4 arrCubicWeights(float t){
    float q=1.0-t,t2=t*t,t3=t2*t;
    if(uArrSmoothing==5)return vec4(-0.5*t3+t2-0.5*t,1.5*t3-2.5*t2+1.0,-1.5*t3+2.0*t2+0.5*t,0.5*t3-0.5*t2);
    return vec4(q*q*q,3.0*t3-6.0*t2+4.0,-3.0*t3+3.0*t2+3.0*t+1.0,t3)/6.0;
}
void addTexel(ivec2 p,float w,vec3 origin,vec3 normal,float tolerance,inout float sum,inout float weight){
    p=clamp(p,ivec2(0),textureSize(uDepth,0)-1);
    if(texelFetch(uDepth,p,0).r>=1.0)return;
    vec3 n=normalize(texelFetch(uNormal,p,0).xyz);
    if(dot(n,normal)<0.95)return;
    vec3 delta=positionAt(p)-origin;
    if(max(abs(dot(delta,normal)),abs(dot(delta,n)))>tolerance)return;
    sum+=w*visibilityAt(p);weight+=w;
}
void reconstruct(vec2 p,vec3 origin,vec3 normal,float tolerance,inout float sum,inout float weight){
    ivec2 base=ivec2(floor(p));vec2 f=fract(p);
    if(uArrSmoothing<4){
        if(uArrSmoothing==3)f=f*f*(3.0-2.0*f);
        for(int y=0;y<2;++y)for(int x=0;x<2;++x){
            float w=(x==0?1.0-f.x:f.x)*(y==0?1.0-f.y:f.y);
            addTexel(base+ivec2(x,y),w,origin,normal,tolerance,sum,weight);
        }
    }else{
        vec4 wx=arrCubicWeights(f.x),wy=arrCubicWeights(f.y);
        for(int y=0;y<4;++y)for(int x=0;x<4;++x)
            addTexel(base+ivec2(x-1,y-1),wx[x]*wy[y],origin,normal,tolerance,sum,weight);
    }
}
void main(){
    ivec2 pixel=ivec2(gl_FragCoord.xy);
    if(texelFetch(uDepth,pixel,0).r>=1.0){outVisibility=1.0;return;}
    vec3 p=positionAt(pixel),n=normalize(texelFetch(uNormal,pixel,0).xyz);
    float tolerance=max(0.0001,0.25*max(length(dFdx(p)),length(dFdy(p))));
    float sum=0.0,weight=0.0;
    // Four symmetric subpixel reconstruction locations; no half-pixel image shift.
    for(int y=-1;y<=1;y+=2)for(int x=-1;x<=1;x+=2)
        reconstruct(vec2(pixel)+vec2(x,y)*0.5*uArrSmoothingRadius,p,n,tolerance,sum,weight);
    outVisibility=abs(weight)>1e-6?clamp(sum/weight,0.0,1.0):visibilityAt(pixel);
}
