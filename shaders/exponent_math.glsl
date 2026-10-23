// x: arithmetic gap, y: mean log gap, z: minimum gap.
vec3 exponentStatistics(in float gaps[256],int count){
    if(count==0)return vec3(0);
    float sum=0.0,logs=0.0,minimum=1e30;
    for(int i=0;i<count;++i){
        float g=max(gaps[i],1e-8);
        sum+=g;logs+=log(g);minimum=min(minimum,g);
    }
    return vec3(sum/float(count),logs/float(count),minimum);
}
float distanceExponent(float gapWorld,float nearWorld,float farWorld,float low,float high){
    if(low==high)return low;
    return mix(high,low,smoothstep(nearWorld,farWorld,gapWorld));
}
float exponentExpm1(float x){
    return abs(x)<0.01?x*(1.0+x*(0.5+x*(1.0/6.0+x/24.0))):exp(x)-1.0;
}
float exponentLog1p(float x){
    return abs(x)<0.01?x*(1.0+x*(-0.5+x*(1.0/3.0-x/4.0))):log(1.0+x);
}
// Power: (mean(g^-y))^(-1/y); inverse: sum(g*g^-y)/sum(g^-y).
// Normalize powers by the minimum gap, so weights never exceed one.
float exponentGap(in float gaps[256],int count,vec3 stats,float y,bool inverse){
    if(count==0)return 0.0;
    if(y<=0.0)return inverse?stats.x:exp(stats.y);
    if(stats.z==stats.x)return stats.x;
    float weights=0.0,numerator=0.0,change=0.0;
    for(int i=0;i<count;++i){
        float g=max(gaps[i],1e-8),ratio=stats.z/g;
        if(inverse||y==1.0||y==2.0){
            float w=y==1.0?ratio:y==2.0?ratio*ratio:pow(ratio,y);
            weights+=w;
            if(inverse)numerator+=g*w;
        }else change+=exponentExpm1(y*log(ratio));
    }
    float result;
    if(y==1.0)result=stats.z*float(count)/weights;
    else if(inverse)result=numerator/weights;
    else if(y==2.0)result=stats.z*sqrt(float(count)/weights);
    else result=stats.z*exp(-exponentLog1p(change/float(count))/y);
    return clamp(result,stats.z,max(stats.z,stats.x));
}
