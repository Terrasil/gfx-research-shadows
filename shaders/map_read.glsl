#ifndef SHADOW_MAP_READ
#define SHADOW_MAP_READ 1
#ifdef PACKED_INTERVALS
#include "packed_depth.glsl"
layout(binding=2) uniform usampler2DArray uPackedMaps;
#define shadowGatherOffset(uv,offset) (vec4(textureGatherOffset(uPackedMaps,uv,offset,0)>>16u)*(1.0/65535.0))
#define layerGatherOffset(uv,offset) textureGatherOffset(uPackedMaps,uv,offset,0)
#else
#define shadowGatherOffset(uv,offset) textureGatherOffset(uMaps,uv,offset,0)
#define layerGatherOffset(uv,offset) textureGatherOffset(uMaps,uv,offset,0)
#endif
float nearestMapDepth(ivec2 pixel){
#ifdef PACKED_INTERVALS
    return unpackDepthInterval(texelFetch(uPackedMaps,ivec3(pixel,0),0).r).x;
#else
    return texelFetch(uMaps,ivec3(pixel,0),0).r;
#endif
}
vec2 mapDepthPair(ivec2 pixel){
#ifdef PACKED_INTERVALS
    return unpackDepthInterval(texelFetch(uPackedMaps,ivec3(pixel,0),0).r);
#else
    return vec2(nearestMapDepth(pixel));
#endif
}
void mapDepthGather(vec2 uv,out vec4 front,out vec4 back){
#ifdef PACKED_INTERVALS
    uvec4 q=textureGather(uPackedMaps,vec3(uv,0),0);
    front=vec4(q>>16u)*(1.0/65535.0);
    back=vec4(q&65535u)*(1.0/65535.0);
    back=mix(back,vec4(1),equal(q,uvec4(emptyInterval)));
#else
    front=textureGather(uMaps,vec3(uv,0),0);back=front;
#endif
}
#endif
