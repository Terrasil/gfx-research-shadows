uniform int uDepthLayers,uSourceSamples,uBitmaskGrid;
// Backproject piecewise constant depth patches and union source-sample masks.
// The neighborhood is sampled on a bounded grid, not the paper's full search hierarchy.
vec2 patchGradient(ivec2 p,int layer,float depth){
    vec2 gradient=vec2(0);
    for(int axis=0;axis<2;++axis){
        ivec2 offset=axis==0?ivec2(1,0):ivec2(0,1);
        float left=1.0,right=1.0;
        if(p[axis]>0)left=texelFetch(uMaps,ivec3(p-offset,layer),0).r;
        if(p[axis]+1<uMapResolution)right=texelFetch(uMaps,ivec3(p+offset,layer),0).r;
        bool hasLeft=left<1.0,hasRight=right<1.0;
        float a=depth-left,b=right-depth;
        if(hasLeft&&hasRight)gradient[axis]=abs(a)<abs(b)?a:b;
        else if(hasLeft)gradient[axis]=a;
        else if(hasRight)gradient[axis]=b;
    }
    return gradient*float(uMapResolution)*0.5;
}
float bitmaskVisibility(vec3 light,vec3 normal){
    vec2 points[64];float weights[64];float total=0.0;
    for(int i=0;i<uSourceSamples;++i){
        float w;vec3 direction=sourceDirection(i,w);
        points[i]=sourceDisk(i);weights[i]=w*max(dot(normal,direction),0.0);total+=weights[i];
    }
    float scale=uDepthRange*uTanAngle/uExtent;
    float searchRadius=min(max(light.z,0.0)*scale,uMaxRadiusWorld/uExtent);
    int radius=int(ceil(searchRadius*float(uMapResolution)*0.5))+1;
    ivec2 pixel=ivec2(floor((light.xy*0.5+0.5)*float(uMapResolution)));
    ivec2 begin=max(pixel-ivec2(radius),ivec2(0)),end=min(pixel+ivec2(radius),ivec2(uMapResolution-1));
    int stride=max(1,(2*radius+1+uBitmaskGrid-1)/uBitmaskGrid);
    uvec2 blockedMask=uvec2(0);
    for(int y=begin.y;y<=end.y;y+=stride)for(int x=begin.x;x<=end.x;x+=stride){
        ivec2 lo=ivec2(x,y),hi=min(lo+ivec2(stride-1),end),q=(lo+hi)/2;
        vec2 patchCenter=(vec2(lo+hi)+1.0)/float(uMapResolution)-1.0;
        vec2 halfSize=vec2(hi-lo+1)/float(uMapResolution);
        for(int layer=0;layer<uDepthLayers;++layer){
            float d=texelFetch(uMaps,ivec3(q,layer),0).r;
            if(d>=1.0)break;
            vec2 gradient=patchGradient(q,layer,d);
            vec2 sampleCenter=(vec2(q)+0.5)/float(uMapResolution)*2.0-1.0;
            d+=dot(gradient,patchCenter-sampleCenter);
            float gap=light.z-d;
            if(gap+dot(abs(gradient),halfSize)<=uDepthBias)continue;
            // Perspective bounds of the tilted patch on the angular source plane.
            vec2 boundLo=vec2(1e20),boundHi=vec2(-1e20);bool bounded=true;
            for(int corner=0;corner<4;++corner){
                vec2 offset=vec2((corner&1)==0?-1.0:1.0,(corner&2)==0?-1.0:1.0)*halfSize;
                float cornerGap=gap-dot(gradient,offset);
                if(cornerGap<=uDepthBias){bounded=false;break;}
                vec2 projected=(patchCenter+offset-light.xy)/(cornerGap*scale);
                boundLo=min(boundLo,projected);boundHi=max(boundHi,projected);
            }
            if(bounded&&(any(greaterThan(boundLo,vec2(1)))||any(lessThan(boundHi,vec2(-1)))))continue;
            for(int i=0;i<uSourceSamples;++i){
                int word=i/32;uint bit=1u<<uint(i&31);
                if((blockedMask[word]&bit)!=0u)continue;
                vec2 slope=points[i]*scale;
                float denominator=1.0+dot(gradient,slope);
                if(abs(denominator)<1e-8)continue;
                float distance=(gap+dot(gradient,patchCenter-light.xy))/denominator;
                vec2 hit=light.xy+distance*slope;
                if(distance>uDepthBias&&all(lessThanEqual(abs(hit-patchCenter),halfSize)))blockedMask[word]|=bit;
            }
        }
    }
    float lit=0.0;
    for(int i=0;i<uSourceSamples;++i)
        if((blockedMask[i/32]&(1u<<uint(i&31)))==0u)lit+=weights[i];
    return total>1e-12?lit/total:1.0;
}
