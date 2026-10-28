// Spatial moments of the existing PCF samples, not moments of stored depth.
struct RgbaPcfMoment{float visibility;vec2 direction;float margin;float confidence;float gap;float ratio;};
float rgbaDiskCoverage(float x){
    if(x<=-1.0)return 0.0;if(x>=1.0)return 1.0;
    float h=sqrt(max(0.0,1.0-x*x));
    return 0.5+(atan(x,h)+x*h)/3.14159265359;
}
float rgbaInverseCoverage(float value){
    float x=clamp(2.0*value-1.0,-0.98,0.98);
    for(int i=0;i<4;++i){
        float h=sqrt(max(1e-6,1.0-x*x));
        x=clamp(x-(rgbaDiskCoverage(x)-value)*1.57079632679/h,-0.9999,0.9999);
    }
    return x;
}
RgbaPcfMoment rgbaMomentFromSums(float value,vec2 mean,vec2 litMean,vec3 second,float ratio,float gap,float complete){
    RgbaPcfMoment result=RgbaPcfMoment(value,vec2(0),0.0,0.0,gap,ratio);
    if(value<=0.02||value>=0.98||ratio<=0.0||gap<=0.0||complete<0.5)return result;
    vec2 moment=litMean-mean*value;
    vec3 covariance=second-vec3(mean.x*mean.x,mean.x*mean.y,mean.y*mean.y);
    float determinant=covariance.x*covariance.z-covariance.y*covariance.y;
    if(determinant<=1e-7||dot(moment,moment)<=1e-10)return result;
    vec2 direction=vec2(covariance.z*moment.x-covariance.y*moment.y,covariance.x*moment.y-covariance.y*moment.x)/determinant;
    direction=normalize(direction);
    float distance=rgbaInverseCoverage(value);
    float ideal=0.21220659079*pow(max(0.0,1.0-distance*distance),1.5);
    float shape=dot(moment,direction)/max(ideal,1e-6);
    result.direction=direction;result.margin=distance*ratio;
    result.confidence=smoothstep(0.35,0.75,shape)*(1.0-smoothstep(1.3,2.0,shape));
    return result;
}
// Orientations are expressed in one common sun-disk basis before fitting.
// Binary/fully covered views constrain the fit but never invent a distance.
vec2 rgbaMomentReconstruct(vec2 points[4],RgbaPcfMoment observations[4]){
    vec2 sumNormal=vec2(0);float weight=0.0,sumMargin=0.0,minGap=1e20,maxGap=0.0,partial=0.0;
    for(int i=0;i<4;++i){
        RgbaPcfMoment q=observations[i];
        if(q.gap>0.0){minGap=min(minGap,q.gap);maxGap=max(maxGap,q.gap);}
        float w=q.confidence;
        if(w>0.25)partial+=1.0;
        sumNormal+=q.direction*w;weight+=w;
        sumMargin+=(q.margin-dot(q.direction,points[i]))*w;
    }
    if(partial<1.5||weight<0.25||length(sumNormal)<1e-7)return vec2(0);
    vec2 direction=normalize(sumNormal);float center=sumMargin/weight;
    float confidence=smoothstep(0.85,0.98,length(sumNormal)/weight);
    if(maxGap>0.0)confidence*=1.0-smoothstep(1.5,3.0,maxGap/max(minGap,1e-7));
    float disagreement=0.0,residual=0.0;
    for(int i=0;i<4;++i){
        RgbaPcfMoment q=observations[i];float predicted=center+dot(direction,points[i]);
        if(q.confidence>0.0)disagreement=max(disagreement,abs(q.margin-predicted));
        float coverage=q.ratio>1e-6?rgbaDiskCoverage(predicted/q.ratio):(predicted>=0.0?1.0:0.0);
        residual=max(residual,abs(coverage-q.visibility));
    }
    confidence*=1.0-smoothstep(0.08,0.35,disagreement);
    confidence*=1.0-smoothstep(0.08,0.30,residual);
    return vec2(rgbaDiskCoverage(center),clamp(confidence,0.0,1.0));
}
