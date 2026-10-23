// Proposed single-map experiment: two local blocker scales, no caster annotations.
float adaptivePcss(vec2 p,float d,vec2 gradient,float angle,bool sharpPriority,out vec4 diagnostic){
    float scale=uDepthRange*uTanAngle/uExtent,maxRadius=uMaxRadiusWorld/uExtent;
    float searchRadius=min(max(d,0.0)*scale,maxRadius);
    float gaps[256];int count=0;
    float lo=1e20,hi=0.0,sum=0.0;
    for(int i=0;i<uBlockerSamples;++i){
        // A close ring protects thin contact blockers; a broad ring finds distant blockers.
        vec2 offset=vec2(0);
        if(i>0){
            int n=max(1,(uBlockerSamples-1)/2);
            bool closeRing=i<=n;
            int j=closeRing?i-1:i-n-1;
            int samples=closeRing?n:max(1,uBlockerSamples-1-n);
            offset=vogel(j,samples,angle)*(closeRing?searchRadius*0.2:searchRadius);
        }
        float gap=depthGap(p+offset,d,p,gradient,0);
        if(gap>uDepthBias){
            gaps[count++]=gap;sum+=gap;lo=min(lo,gap);hi=max(hi,gap);
        }
    }
    diagnostic=vec4(0,0,0,1);
    if(count==0)return 1.0;
    float nearGap=lo,farGap=hi,n0=0.0,n1=0.0;
    for(int iteration=0;iteration<4;++iteration){
        float threshold=sqrt(max(nearGap*farGap,1e-20));
        float sum0=0.0,sum1=0.0;n0=0.0;n1=0.0;
        for(int i=0;i<count;++i){
            if(gaps[i]<=threshold){sum0+=gaps[i];n0+=1.0;}
            else{sum1+=gaps[i];n1+=1.0;}
        }
        if(n0>0.0)nearGap=sum0/n0;
        if(n1>0.0)farGap=sum1/n1;
    }
    float threshold=sqrt(max(nearGap*farGap,1e-20));
    float variance0=0.0,variance1=0.0;
    for(int i=0;i<count;++i){
        float delta=log(max(gaps[i],1e-10)/(gaps[i]<=threshold?nearGap:farGap));
        if(gaps[i]<=threshold)variance0+=delta*delta;else variance1+=delta*delta;
    }
    float separation=log(max(farGap/nearGap,1.0));
    float spread=sqrt(variance0/max(n0,1.0))+sqrt(variance1/max(n1,1.0));
    float confidence=(n0>=2.0&&n1>=2.0)?smoothstep(0.45,1.05,separation)*
                     smoothstep(1.0,3.0,separation/max(spread,0.1)):0.0;
    float meanRadius=min(sum/float(count)*scale,maxRadius);
    float r0=min(nearGap*scale,maxRadius),r1=min(farGap*scale,maxRadius);
    float single=filtered(p,d,gradient,0,meanRadius,angle,-1,0.0);
    float result=single;
    float sharp=1.0;
    if(confidence>0.0||sharpPriority){
        sharp=filtered(p,d,gradient,0,r0,angle,0,threshold);
        if(confidence>0.0){
            float soft=filtered(p,d,gradient,0,r1,angle,1,threshold);
            result=mix(single,sharp*soft,confidence);
        }
    }
    // The confidence blend can violate V(all blockers) <= V(near blockers).
    // Restore that upper bound with the filtered near component, not a binary
    // center-depth test that would turn the far penumbra into a hard shadow.
    float previous=result;
    if(sharpPriority)result=min(result,sharp);
    diagnostic=vec4((result<previous?r0:mix(meanRadius,r0,confidence))*uExtent,
                    mix(meanRadius,r1,confidence)*uExtent,
                    sharpPriority?max(previous-result,0.0):confidence,1.0);
    return clamp(result,0.0,1.0);
}
