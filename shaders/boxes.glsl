#include "stress_instances.glsl"
uniform int uBoxCount,uDroppedGroup;
uniform vec4 uBoxLo[32],uBoxHi[32];
void sceneBox(int index,out vec4 lo,out vec4 hi){
    if(uStressMode!=0){
        vec3 a,b,color;int group;stressInstance(index,a,b,color,group);
        lo=vec4(a,float(group));hi=vec4(b,0);
    }else{lo=uBoxLo[index];hi=uBoxHi[index];}
}
bool boxHit(vec3 origin,vec3 direction,vec3 lo,vec3 hi,out float hit){
    float nearT=0.0,farT=1e30;
    for(int k=0;k<3;++k){
        if(abs(direction[k])<1e-10){if(origin[k]<lo[k]||origin[k]>hi[k])return false;}
        else{
            float a=(lo[k]-origin[k])/direction[k],b=(hi[k]-origin[k])/direction[k];
            nearT=max(nearT,min(a,b));farT=min(farT,max(a,b));
            if(nearT>farT)return false;
        }
    }
    hit=nearT;return farT>1e-6;
}
bool blocked(vec3 origin,vec3 direction){
    for(int b=0;b<uBoxCount;++b){
        vec4 lo,hi;sceneBox(b,lo,hi);
        if(int(lo.w+0.5)==uDroppedGroup)continue;
        float t;if(boxHit(origin,direction,lo.xyz,hi.xyz,t))return true;
    }
    return false;
}
