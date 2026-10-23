// Filter comparison results and blocker statistics, never interpolated raw depth.
// textureGather components are (01, 11, 10, 00).
#if defined(SHADOW_SMOOTHING) && (SHADOW_SMOOTHING == 4 || SHADOW_SMOOTHING == 5)
#include "cubic_visibility.glsl"
#endif
bool gatheredGaps(vec2 uv,float receiverDepth,vec2 origin,vec2 gradient,out vec4 gaps,out vec4 weights){
    gaps=vec4(-1);weights=vec4(0);
    if(any(lessThan(uv,vec2(0)))||any(greaterThanEqual(uv,vec2(1))))return false;
    vec2 st=uv*float(uMapResolution)-0.5,base=floor(st),f=fract(st);
#if defined(SHADOW_SMOOTHING) && SHADOW_SMOOTHING == 3
    // Monotone Hermite interpolation, as in texture reconstruction for terrain.
    f=f*f*(3.0-2.0*f);
#endif
    vec4 depths=textureGather(uMaps,vec3(uv,0),0);
    vec2 a=unwarpPoint((base+0.5)/float(uMapResolution)*2.0-1.0);
    vec2 b=unwarpPoint((base+1.5)/float(uMapResolution)*2.0-1.0);
    vec4 x=vec4(a.x,b.x,b.x,a.x),y=vec4(b.y,b.y,a.y,a.y);
    gaps=receiverDepth+gradient.x*(x-origin.x)+gradient.y*(y-origin.y)-depths;
    gaps=mix(gaps,vec4(-1),greaterThanEqual(depths,vec4(1)));
    weights=vec4((1.0-f.x)*f.y,f.x*f.y,f.x*(1.0-f.y),(1.0-f.x)*(1.0-f.y));
    return true;
}
float smoothVisibility(vec2 uv,float depth,vec2 origin,vec2 gradient){
#if defined(SHADOW_SMOOTHING) && (SHADOW_SMOOTHING == 4 || SHADOW_SMOOTHING == 5)
    return cubicVisibility(uv,depth,origin,gradient);
#else
    vec4 gaps,weights;
    if(!gatheredGaps(uv,depth,origin,gradient,gaps,weights))return 1.0;
    return clamp(1.0-dot(weights,vec4(greaterThan(gaps,vec4(uDepthBias)))),0.0,1.0);
#endif
}
vec3 smoothBlockers(vec2 uv,float depth,vec2 origin,vec2 gradient){
    vec4 gaps,weights;
    if(!gatheredGaps(uv,depth,origin,gradient,gaps,weights))return vec3(0);
    weights*=vec4(greaterThan(gaps,vec4(uDepthBias)));
    return vec3(dot(weights,log(max(gaps,vec4(1e-8)))),dot(weights,gaps),dot(weights,vec4(1)));
}
