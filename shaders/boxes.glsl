uniform int uBoxCount,uDroppedGroup;
uniform vec4 uBoxLo[32],uBoxHi[32];
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
        if(int(uBoxLo[b].w+0.5)==uDroppedGroup)continue;
        float t;if(boxHit(origin,direction,uBoxLo[b].xyz,uBoxHi[b].xyz,t))return true;
    }
    return false;
}
