// All reconstruction/filter code uses this one depth access operation.
// Backend selection is a compile-time variant, not a per-sample uniform branch.
#ifndef MULTIVIEW_STORAGE
#define MULTIVIEW_STORAGE 0
#endif
#if MULTIVIEW_STORAGE == 1
layout(binding=2) uniform sampler2DArray uViewDepthArray;
float rgbaDepthAt(ivec2 p,int channel){
    return texelFetch(uViewDepthArray,ivec3(p,channel),0).r;
}
#elif MULTIVIEW_STORAGE == 2
layout(binding=2) uniform sampler2D uViewDepth0;
layout(binding=3) uniform sampler2D uViewDepth1;
layout(binding=4) uniform sampler2D uViewDepth2;
layout(binding=5) uniform sampler2D uViewDepth3;
float rgbaDepthAt(ivec2 p,int channel){
    // Explicit sampler selection also permits nonuniform channel queries.
    if(channel==0)return texelFetch(uViewDepth0,p,0).r;
    if(channel==1)return texelFetch(uViewDepth1,p,0).r;
    if(channel==2)return texelFetch(uViewDepth2,p,0).r;
    return texelFetch(uViewDepth3,p,0).r;
}
#else
layout(binding=2) uniform sampler2D uRgbaDepth;
float rgbaDepthAt(ivec2 p,int channel){
    return texelFetch(uRgbaDepth,p,0)[channel];
}
#endif
