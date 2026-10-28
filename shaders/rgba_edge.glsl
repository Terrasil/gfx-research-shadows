// Bounded local silhouette search. Only the four rasterized depth views are used.
// A ray here is a 2D search in a shadow texture, not a scene intersection ray.
vec2 rgbaDepthState(vec2 uv,vec3 light,int channel,vec2 gradient){
    ivec2 t=ivec2(uv*float(uMapResolution));
    float d=rgbaDepthAt(t,channel);
    vec2 p=unwarpPoint((vec2(t)+0.5)/float(uMapResolution)*2.0-1.0);
    float gap=light.z+dot(gradient,p-light.xy)-d-uDepthBias*uBiasScale[channel];
    return vec2(d<1.0&&gap>0.0?0.0:1.0,max(0.0,gap)*uDepthRanges[channel]);
}
struct RgbaEdgeInfo{float margin;float nearMargin;vec2 outward;float nearGap;float ambiguous;float softness;};
RgbaEdgeInfo rgbaEdgeInfo(vec3 light,int channel,vec2 gradient){
    vec2 uv=warpPoint(light.xy)*0.5+0.5;
    vec2 center=rgbaDepthState(uv,light,channel,gradient);
    bool lit=center.x>0.5;float best=1e20,gap=0.0;vec2 outward=vec2(0);
    bool crossed[8];for(int k=0;k<8;++k)crossed[k]=false;
    for(int direction=0;direction<8;++direction){
        float phi=6.28318530718*float(direction)/8.0;
        vec2 axis=vec2(cos(phi),sin(phi))/float(uMapResolution);
        float previous=0.0;
        for(int ring=0;ring<8;++ring){
            float radius=min(exp2(float(ring)),float(uEdgeSearch));
            vec2 q=uv+axis*radius;
            if(any(lessThan(q,vec2(0)))||any(greaterThanEqual(q,vec2(1))))break;
            vec2 state=rgbaDepthState(q,light,channel,gradient);
            if((state.x>0.5)!=lit){
                crossed[direction]=true;
                float lo=previous,hi=radius;float blockerGap=lit?state.y:center.y;
                for(int refine=0;refine<4;++refine){
                    float mid=(lo+hi)*0.5;vec2 probe=rgbaDepthState(uv+axis*mid,light,channel,gradient);
                    if((probe.x>0.5)==lit)lo=mid;else hi=mid;
                    if(probe.x<0.5)blockerGap=probe.y;
                }
                vec2 edge=unwarpPoint((uv+axis*((lo+hi)*0.5))*2.0-1.0);
                float distance=length(edge-light.xy)*uExtents[channel];
                if(distance<best){best=distance;gap=blockerGap;outward=normalize(edge-light.xy)*(lit?-1.0:1.0);}
                break;
            }
            previous=radius;if(radius>=float(uEdgeSearch))break;
        }
    }
    if(best>1e19||gap<=0.0)return RgbaEdgeInfo(lit?8.0:-8.0,lit?8.0:-8.0,vec2(0),lit?0.0:center.y,0.0,0.0);
    float margin=clamp(best/max(gap*uTanAngle,1e-7),0.0,8.0);
    float nearMargin=clamp(best/max(center.y*uTanAngle,1e-7),0.0,8.0);
    float ambiguous=0.0;for(int k=0;k<4;++k)if(crossed[k]&&crossed[k+4])ambiguous=1.0;
    float step=1.0/float(uMapResolution);
    float footprint=max(length(unwarpPoint((uv+vec2(step,0))*2.0-1.0)-light.xy),length(unwarpPoint((uv+vec2(0,step))*2.0-1.0)-light.xy))*uExtents[channel];
    float softness=gap*uTanAngle/max(footprint,1e-7);
    return RgbaEdgeInfo(lit?margin:-margin,lit?nearMargin:-nearMargin,outward,lit?0.0:center.y,ambiguous,softness);
}

float rgbaEdgeMargin(vec3 light,int channel,vec2 gradient){return rgbaEdgeInfo(light,channel,gradient).margin;}
float rgbaReconstructedDisk(vec2 points[4],RgbaEdgeInfo edges[4],vec3 normal,float raw,float contactRange){
    float margins[4];vec2 normalSum=vec2(0);float count=0.0,minGap=1e20,maxGap=0.0,contact=1.0,ambiguous=0.0,maxSoftness=0.0;
    for(int i=0;i<4;++i){
        margins[i]=edges[i].margin;ambiguous=max(ambiguous,edges[i].ambiguous);maxSoftness=max(maxSoftness,edges[i].softness);
        if(dot(edges[i].outward,edges[i].outward)>0.5){normalSum+=edges[i].outward;count+=1.0;}
        if(edges[i].nearGap>0.0){minGap=min(minGap,edges[i].nearGap);maxGap=max(maxGap,edges[i].nearGap);}
        if(contactRange>0.0&&edges[i].nearGap>0.0&&edges[i].nearGap<contactRange&&dot(edges[i].outward,edges[i].outward)>0.5){
            vec3 plane=vec3(edges[i].outward,edges[i].nearMargin-dot(edges[i].outward,points[i]));
            float protect=1.0-smoothstep(contactRange*0.75,contactRange,edges[i].nearGap);
            float visibility=diskHalfPlane(plane);
            if(visibility<=0.001)contact=min(contact,1.0-protect);
        }
    }
    float coherence=count>0.0?length(normalSum)/count:0.0;
    float confidence=smoothstep(0.5,0.95,coherence);
    confidence*=1.0-ambiguous;
    if(maxGap>0.0){
        confidence*=smoothstep(0.1,0.5,minGap/maxGap);
        confidence*=smoothstep(1.0,2.0,maxSoftness);
    }
    return min(mix(raw,triangulatedVisibility(points,margins,normal),confidence),contact);
}
