// Separate depth bands; one camera and one map. The optional R32UI map keeps
// only the outer endpoints of all surfaces, not an ordered list of layers.
uniform float uLayerMinFilterTexels;
float minimumLayerRadius(vec2 p){
    if(uLayerMinFilterTexels<=0.0)return 0.0;
    vec2 warped=warpPoint(p);float step=2.0/float(uMapResolution);
    // Local texel size in the unwarped light plane, including NDC focusing.
    float texel=min(length(unwarpPoint(warped+vec2(step,0))-p),
                    length(unwarpPoint(warped+vec2(0,step))-p));
    return min(uLayerMinFilterTexels*texel,uMaxRadiusWorld/uExtent);
}
void layerStatistic(float gap,float weight,float split,inout vec2 logs,inout vec2 sums,inout vec2 counts,inout float lo,inout float hi){
    if(gap<=uDepthBias||weight<=0.0)return;
    int band=gap>split?1:0;logs[band]+=weight*log(max(gap,1e-8));
    sums[band]+=weight*gap;counts[band]+=weight;lo=min(lo,gap);hi=max(hi,gap);
}
void layerPairStatistic(vec2 gaps,float weight,float split,inout vec2 logs,inout vec2 sums,inout vec2 counts,inout float lo,inout float hi){
    layerStatistic(gaps.x,weight,split,logs,sums,counts,lo,hi);
#ifdef PACKED_INTERVALS
    // Both endpoints establish the initial range; then keep the frontmost in each band.
    if(abs(gaps.y-gaps.x)>0.5/65535.0&&(split>1e20||(gaps.x>split&&gaps.y<=split)))
        layerStatistic(gaps.y,weight,split,logs,sums,counts,lo,hi);
#endif
}
void layerSearch(vec2 p,float depth,vec2 origin,vec2 gradient,float split,
                 inout vec2 logs,inout vec2 sums,inout vec2 counts,inout float lo,inout float hi){
#if SHADOW_SMOOTHING == 2 || SHADOW_SMOOTHING == 3
    vec4 gaps,backGaps,weights;vec2 uv=warpPoint(p)*0.5+0.5;
    if(!gatheredLayerGaps(uv,depth,origin,gradient,gaps,backGaps,weights))return;
    for(int c=0;c<4;++c)layerPairStatistic(vec2(gaps[c],backGaps[c]),weights[c],split,logs,sums,counts,lo,hi);
#else
    vec2 uv=warpPoint(p)*0.5+0.5;
    if(any(lessThan(uv,vec2(0)))||any(greaterThanEqual(uv,vec2(1))))return;
    ivec2 texel=clamp(ivec2(uv*float(uMapResolution)),ivec2(0),ivec2(uMapResolution-1));
    vec2 depths=mapDepthPair(texel);
    vec2 center=unwarpPoint((vec2(texel)+0.5)/float(uMapResolution)*2.0-1.0);
    vec2 gaps=depth+dot(gradient,center-origin)-depths;
    gaps=mix(gaps,vec2(-1),greaterThanEqual(depths,vec2(1)));
    layerPairStatistic(gaps,1.0,split,logs,sums,counts,lo,hi);
#endif
}
vec4 bandMask(vec4 gaps,float split,bool farBand){
    return vec4(greaterThan(gaps,vec4(uDepthBias)))*
           (farBand?vec4(greaterThan(gaps,vec4(split))):vec4(lessThanEqual(gaps,vec4(split))));
}
// Near occlusion, far occlusion, and far depth hiding an unknown near layer.
vec3 bandCoverage(vec4 gaps,vec4 backGaps,vec4 weights,float split){
    vec4 nearMask=bandMask(gaps,split,false),farMask=bandMask(gaps,split,true);
#ifdef PACKED_INTERVALS
    nearMask=max(nearMask,bandMask(backGaps,split,false));
    farMask=max(farMask,bandMask(backGaps,split,true));
#endif
    return vec3(dot(weights,nearMask),dot(weights,farMask),dot(weights,farMask*(1.0-nearMask)));
}
#if SHADOW_SMOOTHING == 4 || SHADOW_SMOOTHING == 5
#ifdef PACKED_INTERVALS
vec3 bandQuad(uvec4 bits,vec2 x,vec2 y,vec2 wx,vec2 wy,float depth,vec2 origin,vec2 gradient,float split){
    vec4 depths=vec4(bits>>16u)*(1.0/65535.0),backDepths=vec4(bits&65535u)*(1.0/65535.0);
    backDepths=mix(backDepths,vec4(1),equal(bits,uvec4(emptyInterval)));
#else
vec3 bandQuad(vec4 depths,vec2 x,vec2 y,vec2 wx,vec2 wy,float depth,vec2 origin,vec2 gradient,float split){
    vec4 backDepths=depths;
#endif
    vec4 receiver=depth+gradient.x*(vec4(x.x,x.y,x.y,x.x)-origin.x)+gradient.y*(vec4(y.y,y.y,y.x,y.x)-origin.y);
    vec4 gaps=mix(receiver-depths,vec4(-1),greaterThanEqual(depths,vec4(1)));
    vec4 backGaps=mix(receiver-backDepths,vec4(-1),greaterThanEqual(backDepths,vec4(1)));
    return bandCoverage(gaps,backGaps,vec4(wx.x*wy.y,wx.y*wy.y,wx.y*wy.x,wx.x*wy.x),split);
}
#endif
vec3 bandSample(vec2 p,float depth,vec2 origin,vec2 gradient,float split,bool contact){
    vec2 uv=warpPoint(p)*0.5+0.5;
    if(any(lessThan(uv,vec2(0)))||any(greaterThanEqual(uv,vec2(1))))return vec3(0);
#if SHADOW_SMOOTHING == 0
    if(!contact){
        ivec2 texel=clamp(ivec2(uv*float(uMapResolution)),ivec2(0),ivec2(uMapResolution-1));
        vec2 depths=mapDepthPair(texel),center=unwarpPoint((vec2(texel)+0.5)/float(uMapResolution)*2.0-1.0);
        vec2 gaps=mix(vec2(depth+dot(gradient,center-origin))-depths,vec2(-1),greaterThanEqual(depths,vec2(1)));
        return bandCoverage(vec4(gaps.x),vec4(gaps.y),vec4(1,0,0,0),split);
    }
#endif
#if SHADOW_SMOOTHING == 4 || SHADOW_SMOOTHING == 5
    float size=float(uMapResolution);vec2 st=uv*size-0.5,base=floor(st),f=fract(st);
    vec4 wx=cubicWeights(f.x),wy=cubicWeights(f.y);
    vec2 p0=unwarpPoint(clamp((base-0.5)/size*2.0-1.0,vec2(-1),vec2(1)));
    vec2 p1=unwarpPoint(clamp((base+0.5)/size*2.0-1.0,vec2(-1),vec2(1)));
    vec2 p2=unwarpPoint(clamp((base+1.5)/size*2.0-1.0,vec2(-1),vec2(1)));
    vec2 p3=unwarpPoint(clamp((base+2.5)/size*2.0-1.0,vec2(-1),vec2(1)));vec3 lookup=vec3(uv,0);
    vec3 o=bandQuad(layerGatherOffset(lookup,ivec2(-1,-1)),vec2(p0.x,p1.x),vec2(p0.y,p1.y),wx.xy,wy.xy,depth,origin,gradient,split);
    o+=bandQuad(layerGatherOffset(lookup,ivec2(1,-1)),vec2(p2.x,p3.x),vec2(p0.y,p1.y),wx.zw,wy.xy,depth,origin,gradient,split);
    o+=bandQuad(layerGatherOffset(lookup,ivec2(-1,1)),vec2(p0.x,p1.x),vec2(p2.y,p3.y),wx.xy,wy.zw,depth,origin,gradient,split);
    o+=bandQuad(layerGatherOffset(lookup,ivec2(1,1)),vec2(p2.x,p3.x),vec2(p2.y,p3.y),wx.zw,wy.zw,depth,origin,gradient,split);
    o=clamp(o,vec3(0),vec3(1));float total=max(1.0,o.x+o.z);o.xz/=total;return o;
#else
    vec4 gaps,backGaps,weights;if(!gatheredLayerGaps(uv,depth,origin,gradient,gaps,backGaps,weights))return vec3(0);
    return bandCoverage(gaps,backGaps,weights,split);
#endif
}
float geometricLayers(vec3 light,vec2 gradient,vec2 angles){
    bool inner=uInnerPenumbra!=0;
    if(inner&&depthGap(light.xy,light.z,light.xy,gradient)<=uDepthBias)return 1.0;
    float scale=uDepthRange*uTanAngle/uExtent,maxRadius=uMaxRadiusWorld/uExtent;
    float searchRadius=min(max(light.z,0.0)*scale,maxRadius),lo=1e30,hi=0.0;
    uint bk=0u,fk=0u;
    if(uSamplePattern==1||uSamplePattern==2){
        // A shared randomized quadrature avoids decorrelation grain without a
        // screen-space blur. The two bands still use paired filter directions.
        uvec2 pixel=uSharedSamples!=0?uvec2(0):uvec2(gl_FragCoord.xy);
        bk=diskSeed(pixel,uint(uSampleSeed),0u);fk=diskSeed(pixel,uint(uSampleSeed),1u);
    }
    vec2 logs=vec2(0),sums=vec2(0),counts=vec2(0);
    for(int i=0;i<uBlockerSamples;++i)
        layerSearch(light.xy+diskSample(i,uBlockerSamples,angles.x,uSamplePattern,bk)*searchRadius,light.z,light.xy,gradient,1e30,logs,sums,counts,lo,hi);
    if(counts.x<=0.0)return 1.0;
    float minimumRadius=minimumLayerRadius(light.xy);
    float globalRadius=max(minimumRadius,min(min(sums.x/counts.x,exp(logs.x/counts.x))*scale,maxRadius));
    if(hi<lo*uLayerSeparation){
        outDiagnostic=vec4(globalRadius*uExtent,globalRadius*uExtent,0,1);
        float v=filtered(light.xy,light.z,gradient,globalRadius,angles.y,fk);return inner?clamp(2.0*v,0.0,1.0):v;
    }
    float split=sqrt(lo*hi);logs=vec2(0);sums=vec2(0);counts=vec2(0);
    for(int i=0;i<uBlockerSamples;++i)
        layerSearch(light.xy+diskSample(i,uBlockerSamples,angles.x,uSamplePattern,bk)*searchRadius,light.z,light.xy,gradient,split,logs,sums,counts,lo,hi);
    vec2 radii=max(vec2(minimumRadius),min(min(sums/max(counts,vec2(1e-8)),exp(logs/max(counts,vec2(1e-8))))*scale,vec2(maxRadius)));
    outDiagnostic=vec4(radii*uExtent,counts.x/(counts.x+counts.y),1);
    vec2 warped=warpPoint(light.xy);float step=2.0/float(uMapResolution);
    float footprint=0.5*min(length(unwarpPoint(warped+vec2(step,0))-light.xy),length(unwarpPoint(warped+vec2(0,step))-light.xy));
    vec2 blend=smoothstep(vec2(footprint*0.5),vec2(footprint*1.5),radii);
    vec3 contactBands=bandSample(light.xy,light.z,light.xy,gradient,split,true);
    float known=0.0,unknown=0.0,nearSum=0.0,openSum=0.0;
    for(int i=0;i<uFilterSamples;++i){
        vec2 omega=diskSample(i,uFilterSamples,angles.y,uSamplePattern,fk);
        vec3 nearBands=mix(contactBands,bandSample(light.xy+omega*radii.x,light.z,light.xy,gradient,split,false),blend.x);
        float vf=1.0-mix(contactBands.y,bandSample(light.xy+omega*radii.y,light.z,light.xy,gradient,split,false).y,blend.y);
        float open=max(0.0,1.0-nearBands.x-nearBands.z);
        nearSum+=nearBands.x;openSum+=open;
        known+=open*vf;unknown+=nearBands.z*vf;
    }
    float conditionalLit=(openSum+1e-6)/(openSum+nearSum+1e-6);
    float visible=(known+unknown*conditionalLit)/float(uFilterSamples);
    return inner?clamp(2.0*visible,0.0,1.0):visible;
}
