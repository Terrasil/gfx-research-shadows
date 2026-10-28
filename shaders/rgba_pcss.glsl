uniform int uPcssBlockerSamples,uPcssMean;
uniform float uPcssSearchRadius;
uniform int uPcssUseBounds;
uniform vec4 uPcssCasterNear;

// Orthographic directional-light PCSS. Distances and the search limit are in
// world units; each channel projects them with its own half extent and range.
// Return mean gap, blocker count, normalized search radius.
vec3 rgbaPcssBlockers(vec3 light,int channel,vec2 gradient){
    if(!rgbaPcfCovered(light))return vec3(0);
    float receiverToNear=max(light.z,0.0);
    if(uPcssUseBounds!=0)receiverToNear=max(light.z-uPcssCasterNear[channel],0.0);
    // A sun camera's arbitrary translation must not enlarge the search kernel.
    float radius=min(receiverToNear*uDepthRanges[channel]*uTanAngle,uPcssSearchRadius)/uExtents[channel];
    float sum=0.0,logs=0.0,count=0.0;
    int samples=radius>1e-8?uPcssBlockerSamples:1;
    // The center is one equally weighted sample. It never terminates the search.
    for(int i=0;i<samples;++i){
        vec2 offset=vec2(0);
        if(i>0)offset=rgbaPcfDisk(i-1,samples-1)*radius;
        float gap=rgbaPcfGap(light.xy+offset,light,channel,gradient);
        if(gap>0.0){sum+=gap;if(uPcssMean==1)logs+=log(max(gap,1e-8));count+=1.0;}
    }
    float gap=count>0.0?(uPcssMean==1?exp(logs/count):sum/count):0.0;
    return vec3(gap,count,radius);
}
