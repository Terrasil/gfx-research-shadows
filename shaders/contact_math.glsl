// Scalar helpers shared by the rendering shader and independent GPU checks.
float preferredGap(float meanGap,float candidate,float priority){
    if(priority<=0.0)return meanGap;
    return mix(meanGap,min(meanGap,candidate),clamp(priority,0.0,1.0));
}
float geometricGap(float logSum,int count){
    return count>0?exp(logSum/float(count)):0.0;
}
// Iterative quickselect, no additional depth reads. The bounded input has at most 256 gaps.
float selectGap(inout float gaps[256],int count,int rank){
    int left=0,right=count-1;
    for(int iteration=0;iteration<256&&left<right;++iteration){
        float pivot=gaps[(left+right)/2];
        int i=left,j=right;
        while(i<=j){
            while(i<=right&&gaps[i]<pivot)++i;
            while(j>=left&&gaps[j]>pivot)--j;
            if(i<=j){float value=gaps[i];gaps[i]=gaps[j];gaps[j]=value;++i;--j;}
        }
        if(rank<=j)right=j;
        else if(rank>=i)left=i;
        else break;
    }
    return gaps[rank];
}
float quantileGap(inout float gaps[256],int count,float quantile){
    if(count==0)return 0.0;
    float index=clamp(quantile,0.0,1.0)*float(count-1);
    int lower=int(floor(index)),upper=int(ceil(index));
    float a=selectGap(gaps,count,lower);
    if(upper==lower)return a;
    return mix(a,selectGap(gaps,count,upper),fract(index));
}
float contactCurve(float visibility){
    float v=clamp(visibility,0.0,1.0),a=v*v,b=(1.0-v)*(1.0-v);
    return a/(a+b);
}
float contactStrength(float gapWorld,float distanceWorld,float priority){
    return clamp(priority,0.0,1.0)*(1.0-smoothstep(0.0,max(distanceWorld,1e-6),gapWorld));
}
// CHS-inspired cubic mixture of four Gaussian widths, evaluated on a unit disk.
// See AMD ShadowFX's calculateFilterWeight; the sampling/projection here are adapted.
float chsWeight(float radiusSquared,float ratio){
    vec4 gaussian=exp(-max(radiusSquared,0.0)/vec4(0.0625,0.25,0.5625,1.0));
    float t=clamp(ratio,0.0,1.0),s=1.0-t;
    return dot(gaussian,vec4(s*s*s,3.0*s*s*t,3.0*s*t*t,t*t*t));
}
