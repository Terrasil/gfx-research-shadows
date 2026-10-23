// One R32UI stores two UNORM16 endpoints. Empty is (65535,0).
const uint emptyInterval=0xffff0000u;
uint quantizeDepth(float d){return uint(round(clamp(d,0.0,1.0)*65535.0));}
vec2 unpackDepthInterval(uint bits){
    return bits==emptyInterval?vec2(1):vec2(bits>>16u,bits&65535u)*(1.0/65535.0);
}
#if defined(INTERVAL_WRITER)
layout(r32ui,binding=0) uniform uimage2DArray uDepthIntervals;
void recordDepthAt(ivec3 pixel,float depth){
    if(depth<0.0||depth>1.0)return;
    uint q=quantizeDepth(depth),previous=imageLoad(uDepthIntervals,pixel).r;
    for(;;){
        uint next=(min(previous>>16u,q)<<16u)|max(previous&65535u,q);
        if(next==previous)break;
        uint observed=imageAtomicCompSwap(uDepthIntervals,pixel,previous,next);
        if(observed==previous)break;
        previous=observed;
    }
}
#endif
