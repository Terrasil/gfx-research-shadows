// Directional-light, disk-source version of depth-texel backprojection (Guennebaud et al. 2006).
// Exhaustive base-level scan. No HSM, geometry lookup, depth peeling or source-ray marching.
#include "disk_area.glsl"
float backprojection(vec3 light,vec2 gradient,out vec4 diagnostic){
    float scale=uDepthRange*uTanAngle/uExtent;
    float search=min(max(light.z,0.0)*scale,uMaxRadiusWorld/uExtent);
    vec2 uvLo=warpPoint(clamp(light.xy-vec2(search),-1.0,1.0))*0.5+0.5;
    vec2 uvHi=warpPoint(clamp(light.xy+vec2(search),-1.0,1.0))*0.5+0.5;
    ivec2 lo=clamp(ivec2(floor(uvLo*float(uMapResolution)))-1,ivec2(0),ivec2(uMapResolution-1));
    ivec2 hi=clamp(ivec2(ceil(uvHi*float(uMapResolution)))+1,ivec2(0),ivec2(uMapResolution-1));
    float occluded=0.0,maxRadius=0.0;
    for(int y=lo.y;y<=hi.y;++y)for(int x=lo.x;x<=hi.x;++x){
        ivec2 cell=ivec2(x,y);
        float depth=texelFetch(uMaps,ivec3(cell,0),0).r;if(depth>=1.0)continue;
        vec2 a=unwarpPoint(vec2(cell)/float(uMapResolution)*2.0-1.0);
        vec2 b=unwarpPoint(vec2(cell+1)/float(uMapResolution)*2.0-1.0);
        vec2 center=unwarpPoint((vec2(cell)+0.5)/float(uMapResolution)*2.0-1.0);
        float gap=light.z+dot(gradient,center-light.xy)-depth;
        if(gap<=uDepthBias)continue;
        float radius=min(gap*scale,uMaxRadiusWorld/uExtent);
        if(radius<1e-9)continue;
        vec2 qLo=(a-light.xy)/radius,qHi=(b-light.xy)/radius;
        // Join a shared projected boundary to its immediate left/bottom neighbour.
        // Equal-depth cells are unchanged. As in the original gap treatment, overlaps remain possible.
        for(int k=0;k<2;++k){
            ivec2 neighbour=cell-(k==0?ivec2(1,0):ivec2(0,1));
            if(any(lessThan(neighbour,ivec2(0))))continue;
            float nd=texelFetch(uMaps,ivec3(neighbour,0),0).r;
            vec2 nc=unwarpPoint((vec2(neighbour)+0.5)/float(uMapResolution)*2.0-1.0);
            float ng=light.z+dot(gradient,nc-light.xy)-nd;
            if(nd<1.0&&ng>uDepthBias){
                float nr=min(ng*scale,uMaxRadiusWorld/uExtent);
                qLo[k]=min(qLo[k],(a[k]-light.xy[k])/max(nr,1e-9));
            }
        }
        occluded+=diskRectangle(qLo,qHi)/3.141592653589793;
        maxRadius=max(maxRadius,radius*uExtent);
        if(occluded>=1.0){diagnostic=vec4(maxRadius,0,1,1);return 0.0;}
    }
    diagnostic=vec4(maxRadius,0,occluded,1);
    return clamp(1.0-occluded,0.0,1.0);
}
