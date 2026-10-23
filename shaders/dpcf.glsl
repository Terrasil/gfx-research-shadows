// Port of Filament v1.23.1, ShadowSample_DPCF and hardenedKernel.
// Apache-2.0, original sources and license in upstream/filament.
// Changes: forward Z, external projection/radius/rotation, explicit empty-map handling.
#include "dpcf_disk.glsl"
float hardenedKernel(float x){
    x=2.0*x-1.0;float s=sign(x);x=1.0-s*x;x=x*x*x;x=s-x*s;
    return 0.5*x+0.5;
}
float dpcfFilter(vec3 position,vec2 dz_duv,float radius,int taps,float ratioScale,float angle){
    vec2 texelSize=1.0/vec2(textureSize(uMaps,0).xy);
    mat2 R=mat2(cos(angle),sin(angle),-sin(angle),cos(angle));
    float count=0.0,sum=0.0;
    for(int i=0;i<taps;++i){
        vec2 duv=R*(poissonDisk[i]*texelSize*radius);
        float z=textureLod(uMaps,vec3(position.xy+duv,0),0.0).r;
        float occluded=step(-dot(dz_duv,duv),position.z-z);
        count+=occluded;sum+=z*occluded;
    }
    if(count==0.0)return 1.0;
    float blocker=sum/count;
    // Forward-Z equivalent of (z_blocker-z_receiver)/(1-z_blocker) in reverse Z.
    float ratio=clamp((position.z-blocker)/max(blocker,1e-8)*ratioScale,0.0,1.0);
    float occluded=count/float(taps);
    return 1.0-mix(hardenedKernel(occluded),occluded,ratio);
}
