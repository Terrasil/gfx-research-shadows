// A near-only local bound. Missing depths never count as near occlusion.
// Packed endpoints can reveal some hidden casters, but not intermediate layers.
uniform float uContactGuard,uContactMaxGap;
float validNearGap(vec2 depths,float receiver){
    vec2 gaps=vec2(receiver)-depths;
    float gap=1e30;
    for(int c=0;c<2;++c)
        if(depths[c]<1.0&&gaps[c]>uDepthBias&&gaps[c]*uDepthRange<uContactMaxGap)
            gap=min(gap,gaps[c]);
    return gap;
}
vec2 localNearStats(vec2 p,float depth,vec2 origin,vec2 gradient){
    vec2 uv=warpPoint(p)*0.5+0.5,st=uv*float(uMapResolution)-0.5;
    ivec2 base=ivec2(floor(st));vec2 f=fract(st);
    float logs=0.0,coverage=0.0;
    for(int y=0;y<2;++y)for(int x=0;x<2;++x){
        ivec2 t=base+ivec2(x,y);
        if(any(lessThan(t,ivec2(0)))||any(greaterThanEqual(t,ivec2(uMapResolution))))continue;
        vec2 center=unwarpPoint((vec2(t)+0.5)/float(uMapResolution)*2.0-1.0);
        float receiver=depth+dot(gradient,center-origin);
        float gap=validNearGap(mapDepthPair(t),receiver);
        if(gap>1e20)continue;
        float fade=1.0-smoothstep(0.75*uContactMaxGap,uContactMaxGap,gap*uDepthRange);
        float weight=(x==0?1.0-f.x:f.x)*(y==0?1.0-f.y:f.y)*fade;
        logs+=weight*log(max(gap,1e-8));coverage+=weight;
    }
    return vec2(logs,coverage);
}
vec4 contactProtection(vec3 light,vec2 gradient){
    vec2 local=localNearStats(light.xy,light.z,light.xy,gradient);
    if(local.y<=1e-6)return vec4(0);
    float gap=exp(local.x/local.y);
    float radius=min(gap*uDepthRange*uTanAngle/uExtent,uMaxRadiusWorld/uExtent);
    vec2 warped=warpPoint(light.xy);float step=2.0/float(uMapResolution);
    float footprint=0.5*min(length(unwarpPoint(warped+vec2(step,0))-light.xy),
                             length(unwarpPoint(warped+vec2(0,step))-light.xy));
    float coverage=local.y;
    if(radius>0.5*footprint){
        coverage=0.0;
        for(int i=0;i<12;++i)
            coverage+=localNearStats(light.xy+vogel(i,12,0.0)*radius,light.z,light.xy,gradient).y;
        coverage/=12.0;
    }
    // Constrain the supported core; do not replace the full penumbra by a hard mask.
    float alpha=uContactGuard*min(local.y,coverage);
    return vec4(alpha,radius*uExtent,local.y,gap*uDepthRange);
}
