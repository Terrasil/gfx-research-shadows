const float basisPi=3.14159265358979323846;
float varianceVisibility(vec2 m,float t,float minimumVariance,float bleeding){
    if(t<=m.x)return 1.0;
    float variance=max(m.y-m.x*m.x,minimumVariance),delta=t-m.x;
    float p=variance/(variance+delta*delta);
    return clamp((p-bleeding)/(1.0-bleeding),0.0,1.0);
}
float fourierTerm(vec2 cosineSine,int k,float t){
    float frequency=basisPi*float(2*k+1);
    return (2.0/frequency)*(cosineSine.y*cos(frequency*t)-cosineSine.x*sin(frequency*t));
}
float exponentialLogResponse(float value,float t,float exponent){
    return log(max(value,1e-38))+exponent*(1.0-t);
}
float fourierVisibility(sampler2DArray basis,vec2 uv,float t,int terms,float bias){
    // Restrict depth differences to (-1,1), away from the periodic jump.
    float receiver=0.05+0.9*(t-bias),visible=0.5;
    for(int layer=0;layer<terms/2;++layer){
        vec4 q=texture(basis,vec3(uv,layer));
        visible+=fourierTerm(q.xy,2*layer,receiver)+fourierTerm(q.zw,2*layer+1,receiver);
    }
    return clamp(visible,0.0,1.0);
}

float exponentialFallback(sampler2DArray basis,vec2 uv,float t,float exponent,int size){
    vec2 st=uv*float(size)-0.5,f=fract(st);ivec2 p=ivec2(floor(st));
    float a=exp(min(0.0,exponentialLogResponse(texelFetch(basis,ivec3(clamp(p,ivec2(0),ivec2(size-1)),0),0).r,t,exponent)));
    float b=exp(min(0.0,exponentialLogResponse(texelFetch(basis,ivec3(clamp(p+ivec2(1,0),ivec2(0),ivec2(size-1)),0),0).r,t,exponent)));
    float c=exp(min(0.0,exponentialLogResponse(texelFetch(basis,ivec3(clamp(p+ivec2(0,1),ivec2(0),ivec2(size-1)),0),0).r,t,exponent)));
    float d=exp(min(0.0,exponentialLogResponse(texelFetch(basis,ivec3(clamp(p+ivec2(1,1),ivec2(0),ivec2(size-1)),0),0).r,t,exponent)));
    return mix(mix(a,b,f.x),mix(c,d,f.x),f.y);
}
