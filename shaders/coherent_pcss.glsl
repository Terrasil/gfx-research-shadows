// Research candidate: local plane proposals followed by source-correlated visibility.
// One nearest-depth map. No caster IDs or direct access to scene boxes.
uniform int uLocalPlanes,uContourSamples;

bool coherentMapPoint(vec2 p,out ivec2 texel,out vec2 center,out float depth){
    vec2 uv=warpPoint(p)*0.5+0.5;
    if(any(lessThan(uv,vec2(0)))||any(greaterThanEqual(uv,vec2(1))))return false;
    texel=ivec2(uv*float(uMapResolution));
    center=unwarpPoint((vec2(texel)+0.5)/float(uMapResolution)*2.0-1.0);
    depth=texelFetch(uMaps,ivec3(texel,0),0).r;
    return depth<1.0;
}

vec2 coherentGradient(ivec2 texel,vec2 center,float depth){
    vec2 gradient=vec2(0);
    for(int axis=0;axis<2;++axis){
        ivec2 delta=axis==0?ivec2(1,0):ivec2(0,1);
        float left=1.0,right=1.0,leftSlope=0.0,rightSlope=0.0;
        if(texel[axis]>0){
            left=texelFetch(uMaps,ivec3(texel-delta,0),0).r;
            vec2 c=unwarpPoint((vec2(texel-delta)+0.5)/float(uMapResolution)*2.0-1.0);
            leftSlope=(depth-left)/max(center[axis]-c[axis],1e-10);
        }
        if(texel[axis]+1<uMapResolution){
            right=texelFetch(uMaps,ivec3(texel+delta,0),0).r;
            vec2 c=unwarpPoint((vec2(texel+delta)+0.5)/float(uMapResolution)*2.0-1.0);
            rightSlope=(right-depth)/max(c[axis]-center[axis],1e-10);
        }
        if(left<1.0&&right<1.0)gradient[axis]=abs(leftSlope)<abs(rightSlope)?leftSlope:rightSlope;
        else if(left<1.0)gradient[axis]=leftSlope;
        else if(right<1.0)gradient[axis]=rightSlope;
    }
    return gradient;
}

// Observed extrema form an inner approximation of their convex hull, not an AABB.
const vec2 coherentDirections[8]=vec2[8](
    vec2(1,0),vec2(1,1),vec2(0,1),vec2(-1,1),
    vec2(-1,0),vec2(-1,-1),vec2(0,-1),vec2(1,-1));

// Do not merge evidence across a visible gap between coplanar objects.
// Occlusion by a closer surface is allowed, but empty/behind-plane samples are not.
bool coherentConnected(vec2 a,vec2 b,vec3 plane){
    for(int j=1;j<16;++j){
        ivec2 texel;vec2 center;float depth;
        if(!coherentMapPoint(mix(a,b,float(j)/16.0),texel,center,depth))return false;
        if(depth>dot(plane.xy,center)+plane.z+max(2.0*uDepthBias,2e-6))return false;
    }
    return true;
}

bool coherentInside(vec2 p,int k,vec2 witness[64]){
    float area=0.0;
    for(int e=0;e<8;++e){
        vec2 a=witness[k*8+e],b=witness[k*8+((e+1)&7)];
        vec2 edge=b-a,relative=p-a;
        if(edge.x*relative.y-edge.y*relative.x < -1e-10)return false;
        area+=a.x*b.y-a.y*b.x;
    }
    return area>1e-10;
}

vec4 coherentSupport(vec2 p,vec3 plane,int k,vec2 witness[64],out vec2 fraction){
    fraction=vec2(0);
    vec2 uv=warpPoint(p)*0.5+0.5;
    if(any(lessThan(uv,vec2(0)))||any(greaterThanEqual(uv,vec2(1))))return vec4(0);
    vec2 pixel=uv*float(uMapResolution)-0.5;
    ivec2 base=ivec2(floor(pixel));fraction=fract(pixel);
    float support[4];
    for(int corner=0;corner<4;++corner){
        ivec2 q=base+ivec2(corner&1,corner>>1);
        support[corner]=0.0;
        if(any(lessThan(q,ivec2(0)))||any(greaterThanEqual(q,ivec2(uMapResolution))))continue;
        float depth=texelFetch(uMaps,ivec3(q,0),0).r;
        if(depth>=1.0)continue;
        vec2 center=unwarpPoint((vec2(q)+0.5)/float(uMapResolution)*2.0-1.0);
        float expected=dot(plane.xy,center)+plane.z;
        // Depth agreement validates a proposed surface rather than a global depth bin.
        float tolerance=max(2.0*uDepthBias,2e-6);
        bool known=abs(depth-expected)<=tolerance;
        // Continue only within the observed polygon and under a nearer surface.
        // Empty/background texels remain unfilled. Convexity is an assumption.
        bool hidden=uContourSamples>0&&depth<expected-tolerance;
        support[corner]=known||(hidden&&coherentInside(center,k,witness))?1.0:0.0;
    }
    return vec4(support[0],support[1],support[2],support[3]);
}

// Integrate the union over one shared bilinear footprint. Independent bilinear
// filtering followed by max can leave a bright seam between complementary faces.
float coherentJointVisibility(int count,vec4 masks[8],vec2 fractions[8]){
    float cuts[10];cuts[0]=0.0;cuts[count+1]=1.0;
    for(int k=0;k<count;++k){
        float cut=fractions[k].x;int j=k+1;
        while(j>1&&cuts[j-1]>cut){cuts[j]=cuts[j-1];--j;}
        cuts[j]=cut;
    }
    float visibility=0.0;
    for(int i=0;i<=count;++i){
        float width=cuts[i+1]-cuts[i];
        if(width<=0.0)continue;
        float x=0.5*(cuts[i]+cuts[i+1]),lo=0.0,hi=1.0;
        for(int k=0;k<count;++k){
            bool right=x<fractions[k].x;
            float bottom=right?masks[k].y:masks[k].x;
            float top=right?masks[k].w:masks[k].z;
            if(top>0.5)lo=max(lo,fractions[k].y);
            if(bottom>0.5)hi=min(hi,fractions[k].y);
        }
        visibility+=width*max(hi-lo,0.0);
    }
    return visibility;
}

float coherentPcss(vec3 light,vec3 normal,vec2 receiverSlope,float angle,out vec4 diagnostic){
    float scale=uDepthRange*uTanAngle/uExtent;
    float searchRadius=min(max(light.z,0.0)*scale,uMaxRadiusWorld/uExtent);
    vec3 planes[8];vec2 witness[64],anchors[8];int planeCount=0;
    float nearGap=1e20,farGap=0.0;
    for(int i=0;i<uBlockerSamples;++i){
        vec2 offset=vec2(0);
        if(i>0){
            int inner=max(1,(uBlockerSamples-1)/2);
            bool closeRing=i<=inner;
            int index=closeRing?i-1:i-inner-1;
            int count=closeRing?inner:max(1,uBlockerSamples-1-inner);
            offset=vogel(index,count,angle)*searchRadius*(closeRing?0.2:1.0);
        }
        ivec2 texel;vec2 center;float depth;
        if(!coherentMapPoint(light.xy+offset,texel,center,depth))continue;
        float gap=light.z+dot(receiverSlope,center-light.xy)-depth;
        if(gap<=uDepthBias)continue;
        vec2 gradient=coherentGradient(texel,center,depth);
        vec3 proposal=vec3(gradient,depth-dot(gradient,center));
        float proposedDepth=dot(proposal.xy,light.xy)+proposal.z;
        int match=-1;
        for(int k=0;k<planeCount;++k){
            float otherDepth=dot(planes[k].xy,light.xy)+planes[k].z;
            if(abs(proposedDepth-otherDepth)<max(2.0*uDepthBias,2e-6)&&
               length(proposal.xy-planes[k].xy)<0.015){match=k;break;}
        }
        if(match<0&&planeCount<uLocalPlanes){
            match=planeCount++;
            planes[match]=proposal;anchors[match]=center;
            for(int e=0;e<8;++e)witness[match*8+e]=center;
        }
        if(match>=0&&uContourSamples>0&&coherentConnected(anchors[match],center,planes[match]))for(int e=0;e<8;++e){
            if(dot(center,coherentDirections[e])>dot(witness[match*8+e],coherentDirections[e]))
                witness[match*8+e]=center;
        }
    }
    float contourRadius=min(searchRadius*3.0,uMaxRadiusWorld/uExtent);
    // Wider observations update existing proposals without changing the local
    // blocker search or consuming the plane budget with distant geometry.
    for(int i=0;i<uContourSamples;++i){
        vec2 offset=vogel(i,uContourSamples,angle)*contourRadius;
        ivec2 texel;vec2 center;float depth;
        if(!coherentMapPoint(light.xy+offset,texel,center,depth))continue;
        vec2 gradient=coherentGradient(texel,center,depth);
        for(int k=0;k<planeCount;++k){
            if(abs(depth-dot(planes[k].xy,center)-planes[k].z)>max(2.0*uDepthBias,2e-6)||
               length(gradient-planes[k].xy)>=0.015)continue;
            if(!coherentConnected(anchors[k],center,planes[k]))continue;
            for(int e=0;e<8;++e){
                if(dot(center,coherentDirections[e])>dot(witness[k*8+e],coherentDirections[e]))
                    witness[k*8+e]=center;
            }
        }
    }
    diagnostic=vec4(0,0,0,1);
    if(planeCount==0)return 1.0;
    for(int k=0;k<planeCount;++k){
        float gap=light.z-dot(planes[k].xy,light.xy)-planes[k].z;
        if(gap>uDepthBias){nearGap=min(nearGap,gap);farGap=max(farGap,gap);}
    }
    float visible=0.0,total=0.0,mixed=0.0,nearTotal=0.0,farTotal=0.0;
    vec3 tangent=normalize(cross(vec3(0,0,1),uSun)),bitangent=cross(uSun,tangent);
    mat2 rotation=mat2(cos(angle),sin(angle),-sin(angle),cos(angle));
    float splitGap=sqrt(max(nearGap*farGap,1e-20));
    for(int i=0;i<uFilterSamples;++i){
        vec2 disk=rotation*sourceDisk(i);
        vec2 slope=disk*scale;
        vec2 source=disk*uTanAngle;
        vec3 direction=normalize(uSun+tangent*source.x+bitangent*source.y);
        float weight=pow(1.0+dot(source,source),-1.5)*max(dot(normal,direction),0.0);
        float nearCoverage=0.0,farCoverage=0.0;
        vec4 masks[8];vec2 fractions[8];
        for(int k=0;k<planeCount;++k){masks[k]=vec4(0);fractions[k]=vec2(0);}
        for(int k=0;k<planeCount;++k){
            vec3 plane=planes[k];
            float denominator=1.0+dot(plane.xy,slope);
            if(abs(denominator)<1e-8)continue;
            float gap=(light.z-dot(plane.xy,light.xy)-plane.z)/denominator;
            if(gap<=uDepthBias||gap*scale>uMaxRadiusWorld/uExtent)continue;
            masks[k]=coherentSupport(light.xy+gap*slope,plane,k,witness,fractions[k]);
            vec4 mask=masks[k];vec2 f=fractions[k];
            float coverage=mix(mix(mask.x,mask.y,f.x),mix(mask.z,mask.w,f.x),f.y);
            if(gap<=splitGap)nearCoverage=max(nearCoverage,coverage);
            else farCoverage=max(farCoverage,coverage);
        }
        visible+=coherentJointVisibility(planeCount,masks,fractions)*weight;total+=weight;
        mixed+=min(nearCoverage,farCoverage)*weight;
        nearTotal+=nearCoverage*weight;farTotal+=farCoverage*weight;
    }
    float radius0=nearGap<1e19?min(nearGap*scale*uExtent,uMaxRadiusWorld):0.0;
    float radius1=min(farGap*scale*uExtent,uMaxRadiusWorld);
    float visibility=total>1e-12?clamp(visible/total,0.0,1.0):1.0;
    float state=visibility>=0.99?1.0:visibility<=0.01?3.0:2.0;
    if(nearTotal>total*0.01&&farTotal>total*0.01&&radius1>radius0*1.5)state=4.0;
    diagnostic=vec4(radius0,radius1,total>1e-12?mixed/total:0.0,state);
    return visibility;
}
