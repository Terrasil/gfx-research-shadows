#version 460 core
layout(location=0) out float outVisibility;
layout(binding=0) uniform sampler2D uVisibility;
layout(binding=1) uniform sampler2D uDepth;
layout(binding=2) uniform sampler2D uNormal;
layout(binding=3) uniform sampler2D uGuide;
uniform mat4 uInverseVp;
uniform float uBlendStrength,uBlendSigma;
uniform int uBlendRadius;
vec3 positionAt(ivec2 pixel,float depth){
    vec2 uv=(vec2(pixel)+0.5)/vec2(textureSize(uDepth,0));
    vec4 p=uInverseVp*vec4(uv*2.0-1.0,depth*2.0-1.0,1);
    return p.xyz/p.w;
}
void main(){
    ivec2 pixel=ivec2(gl_FragCoord.xy),size=textureSize(uDepth,0);
    float center=texelFetch(uVisibility,pixel,0).r,depth=texelFetch(uDepth,pixel,0).r;
    vec4 guide=texelFetch(uGuide,pixel,0);
    float strength=uBlendStrength*(4.0*center*(1.0-center))*smoothstep(0.75,3.0,guide.x)*(1.0-clamp(guide.y,0.0,1.0));
    if(depth>=1.0||guide.a<0.5||strength<=1e-6){outVisibility=center;return;}
    vec3 origin=positionAt(pixel,depth),normal=normalize(texelFetch(uNormal,pixel,0).xyz);
    float tolerance=max(0.0001,0.25*max(length(positionAt(pixel+ivec2(1,0),depth)-origin),
                                        length(positionAt(pixel+ivec2(0,1),depth)-origin)));
    float sigma=clamp(guide.x*0.35,0.65,float(uBlendRadius)*0.8);
    float sum=0.0,total=0.0;
    for(int y=-2;y<=2;++y)for(int x=-2;x<=2;++x){
        if(abs(x)>uBlendRadius||abs(y)>uBlendRadius)continue;
        ivec2 q=pixel+ivec2(x,y);
        if(any(lessThan(q,ivec2(0)))||any(greaterThanEqual(q,size)))continue;
        float d=texelFetch(uDepth,q,0).r;
        vec4 other=texelFetch(uGuide,q,0);
        // Do not borrow a different cascade coverage or a sharp near shadow.
        if(d>=1.0||other.a<0.5||other.z!=guide.z||other.y>=0.999)continue;
        vec3 n=normalize(texelFetch(uNormal,q,0).xyz),delta=positionAt(q,d)-origin;
        if(dot(normal,n)<0.95||max(abs(dot(delta,normal)),abs(dot(delta,n)))>tolerance)continue;
        float value=texelFetch(uVisibility,q,0).r;
        float dv=(value-center)/max(uBlendSigma,0.001);
        float ds=log((other.x+1.0)/(guide.x+1.0));
        float dc=(other.y-guide.y)/0.15;
        float weight=exp(-0.5*float(x*x+y*y)/(sigma*sigma)-0.5*dv*dv-0.5*ds*ds-0.5*dc*dc);
        sum+=weight*value;total+=weight;
    }
    outVisibility=clamp(mix(center,total>1e-8?sum/total:center,strength),0.0,1.0);
}
