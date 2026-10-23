// Tensor-product cubic reconstruction of visibility, not raw depth.
// Sigg and Hadwiger, GPU Gems 2 chapter 20. Four gathers cover 4x4 texels.
vec4 cubicWeights(float t){
    float q=1.0-t,t2=t*t,t3=t2*t;
#if SHADOW_SMOOTHING == 5
    // Interpolating Catmull-Rom (Keys cubic convolution, a=-0.5).
    return vec4(-0.5*t3+t2-0.5*t,1.5*t3-2.5*t2+1.0,-1.5*t3+2.0*t2+0.5*t,0.5*t3-0.5*t2);
#else
    return vec4(q*q*q,3.0*t3-6.0*t2+4.0,-3.0*t3+3.0*t2+3.0*t+1.0,t3)/6.0;
#endif
}
float cubicQuad(vec4 depths,vec2 x,vec2 y,vec2 wx,vec2 wy,float depth,vec2 origin,vec2 gradient){
    // Gather order: (01, 11, 10, 00).
    vec4 gaps=depth+gradient.x*(vec4(x.x,x.y,x.y,x.x)-origin.x)
                   +gradient.y*(vec4(y.y,y.y,y.x,y.x)-origin.y)-depths;
    vec4 blocked=vec4(greaterThan(gaps,vec4(uDepthBias)))*vec4(lessThan(depths,vec4(1)));
    return dot(blocked,vec4(wx.x*wy.y,wx.y*wy.y,wx.y*wy.x,wx.x*wy.x));
}
float cubicVisibility(vec2 uv,float depth,vec2 origin,vec2 gradient){
    if(any(lessThan(uv,vec2(0)))||any(greaterThanEqual(uv,vec2(1))))return 1.0;
    float size=float(uMapResolution);
    vec2 st=uv*size-0.5,base=floor(st),f=fract(st);
    vec4 wx=cubicWeights(f.x),wy=cubicWeights(f.y);
    // Outside-map texels have depth 1 and contribute no occlusion. Clamp their
    // unwarp arguments to avoid evaluating the rational NDC warp beyond its domain.
    vec2 p0=unwarpPoint(clamp((base-0.5)/size*2.0-1.0,vec2(-1),vec2(1)));
    vec2 p1=unwarpPoint(clamp((base+0.5)/size*2.0-1.0,vec2(-1),vec2(1)));
    vec2 p2=unwarpPoint(clamp((base+1.5)/size*2.0-1.0,vec2(-1),vec2(1)));
    vec2 p3=unwarpPoint(clamp((base+2.5)/size*2.0-1.0,vec2(-1),vec2(1)));
    vec3 lookup=vec3(uv,0);
    float occluded=cubicQuad(shadowGatherOffset(lookup,ivec2(-1,-1)),vec2(p0.x,p1.x),vec2(p0.y,p1.y),wx.xy,wy.xy,depth,origin,gradient);
    occluded+=cubicQuad(shadowGatherOffset(lookup,ivec2(1,-1)),vec2(p2.x,p3.x),vec2(p0.y,p1.y),wx.zw,wy.xy,depth,origin,gradient);
    occluded+=cubicQuad(shadowGatherOffset(lookup,ivec2(-1,1)),vec2(p0.x,p1.x),vec2(p2.y,p3.y),wx.xy,wy.zw,depth,origin,gradient);
    occluded+=cubicQuad(shadowGatherOffset(lookup,ivec2(1,1)),vec2(p2.x,p3.x),vec2(p2.y,p3.y),wx.zw,wy.zw,depth,origin,gradient);
    return clamp(1.0-occluded,0.0,1.0);
}
