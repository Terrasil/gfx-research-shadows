// Port of AMD ShadowFX contactFixedGather4, uniformBlockerSearch,
// percentageCloserSoftShadows, cubicBezierCurve and original fixed weight tables.
// Copyright (c) 2016 Advanced Micro Devices, Inc. MIT license: upstream/shadowfx/LICENSE.txt.
// One full atlas region, one depth layer; same forward-Z comparison and gather component order.
#include "chs_tables.glsl"
float chsWeight(int fs,int row,int col,vec4 coefficients){return dot(chsControls(fs,row,col),coefficients);}
float chsFilter(vec3 position,float bias,float sunArea,int requestedFS){
#ifdef CHS_FILTER_SIZE
    const int fs=CHS_FILTER_SIZE;
#else
    int fs=requestedFS;
#endif
    int fr=fs/2;
    vec2 size=vec2(textureSize(uMaps,0).xy),invSize=1.0/size;
    vec3 uvz=vec3(position.xy,position.z-bias);
    vec2 coord=size*uvz.xy+0.5,frac=fract(coord);
    uvz.xy=floor(coord)*invSize;
    float count=0.0,sum=0.0;
    for(int row=-fr;row<=fr;row+=2)for(int col=-fr;col<=fr;col+=2){
        vec4 depth=textureGather(uMaps,vec3(uvz.xy+vec2(col,row)*invSize,0),0);
        vec4 blocker=vec4(lessThan(depth,vec4(uvz.z)));
        count+=dot(blocker,vec4(1));sum+=dot(depth,blocker);
    }
    float fraction=count/float((fs+1)*(fs+1));
    if(count==0.0||fraction==1.0)return 1.0-fraction;
    float blocker=sum/count;
    float ratio=sqrt(clamp((position.z-blocker)*sunArea/max(blocker,1e-8),0.0,1.0));
    float q=1.0-ratio;
    vec4 coefficients=vec4(q*q*q,3.0*q*q*ratio,3.0*q*ratio*ratio,ratio*ratio*ratio);
    float shadowSum=0.0;
    for(int row=-fr;row<=fr;row+=2){
        int r=row+fr;
        for(int col=-fr;col<=fr;col+=2){
            int c=col+fr;
            vec4 depth=textureGather(uMaps,vec3(uvz.xy+vec2(col,row)*invSize,0),0);
            vec4 curr=vec4(greaterThanEqual(depth,vec4(uvz.z)));
            vec3 a=vec3(chsWeight(fs,r-1,c-1,coefficients),chsWeight(fs,r-1,c,coefficients),chsWeight(fs,r-1,c+1,coefficients));
            vec3 b=vec3(chsWeight(fs,r,c-1,coefficients),chsWeight(fs,r,c,coefficients),chsWeight(fs,r,c+1,coefficients));
            vec3 d=vec3(chsWeight(fs,r+1,c-1,coefficients),chsWeight(fs,r+1,c,coefficients),chsWeight(fs,r+1,c+1,coefficients));
            vec2 above=mix(a.yz,a.xy,frac.x),middle=mix(b.yz,b.xy,frac.x),below=mix(d.yz,d.xy,frac.x);
            vec2 lower=mix(middle,above,frac.y),upper=mix(below,middle,frac.y);
            shadowSum+=dot(curr,vec4(upper,lower.y,lower.x));
        }
    }
    return shadowSum/dot(chsControlSum(fs),coefficients);
}
