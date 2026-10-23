// Heuristic locality-weighted geometric blocker gap. One search loop, no gap array.
// Uniform radial spacing concentrates proposals near the receiver projection.
vec2 localGeometricOffset(int i,int count,float angle){
    if(i==0)return vec2(0);
    float radius=(float(i)-0.5)/float(max(count-1,1));
    float a=float(i-1)*2.39996322972865+angle;
    return radius*vec2(cos(a),sin(a));
}
float localGeometricPcss(vec2 p,float d,vec2 gradient,float angle,out vec4 diagnostic){
    float scale=uDepthRange*uTanAngle/uExtent,maxRadius=uMaxRadiusWorld/uExtent;
    float searchRadius=min(max(d,0.0)*scale,maxRadius);
    vec2 wp=warpPoint(p);float texel=2.0/float(uMapResolution);
    float footprint=0.5*max(length(unwarpPoint(wp+vec2(texel,0))-p),
                            length(unwarpPoint(wp+vec2(0,texel))-p));
    float sum=0.0,count=0.0,weights=0.0,weightedLogs=0.0,weightedSum=0.0;
    for(int i=0;i<uBlockerSamples;++i){
        vec2 offset=localGeometricOffset(i,uBlockerSamples,angle)*searchRadius;
        float gap=depthGap(p+offset,d,p,gradient,0);
        if(gap>uDepthBias){
            float radius=gap*scale+footprint,r2=radius*radius;
            float weight=r2/max(r2+dot(offset,offset),1e-20);
            sum+=gap;count+=1.0;
            weights+=weight;weightedLogs+=weight*log(max(gap,1e-8));weightedSum+=weight*gap;
        }
    }
    diagnostic=vec4(0,0,0,1);
    if(count==0.0)return 1.0;
    float meanGap=sum/count;
    float localGap=min(exp(weightedLogs/weights),weightedSum/weights);
    // The weighted statistic may exceed the unweighted arithmetic gap if lateral
    // relevance favors a farther surface; do not clamp it to the unrelated mean.
    float effective=mix(meanGap,localGap,uSharpPriority);
    float radius=min(effective*scale,maxRadius),meanRadius=min(meanGap*scale,maxRadius);
    diagnostic=vec4(radius*uExtent,meanRadius*uExtent,clamp(1.0-radius/max(meanRadius,1e-8),0.0,1.0),1);
    return filtered(p,d,gradient,0,radius,angle,-1,0.0);
}
