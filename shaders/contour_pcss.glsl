// One nearest-depth map. Bounded local convex surface contours, analytic source area.
// No source-direction loop, scene boxes, depth peeling or temporal history.
#include "contour_area.glsl"
uniform int uContourSteps;

int contourHull(vec2 points[8],out vec2 hull[20]){
    for(int i=1;i<8;++i){
        vec2 point=points[i];int j=i;
        while(j>0&&(points[j-1].x>point.x||
              (points[j-1].x==point.x&&points[j-1].y>point.y))){points[j]=points[j-1];--j;}
        points[j]=point;
    }
    int count=0;
    for(int i=0;i<8;++i){
        while(count>=2&&contourCross(hull[count-1]-hull[count-2],points[i]-hull[count-1])<=1e-14)--count;
        hull[count++]=points[i];
    }
    int lower=count;
    for(int i=6;i>=0;--i){
        while(count>lower&&contourCross(hull[count-1]-hull[count-2],points[i]-hull[count-1])<=1e-14)--count;
        hull[count++]=points[i];
    }
    return max(count-1,0);
}

bool contourSupport(vec2 p,vec3 plane,int index,bool fill,vec2 witness[64]){
    ivec2 texel;vec2 center;float depth;
    if(!coherentMapPoint(p,texel,center,depth))return false;
    float expected=dot(plane.xy,center)+plane.z;
    float tolerance=max(2.0*uDepthBias,2e-6);
    if(abs(depth-expected)<=tolerance)return true;
    // Convex continuation is permitted only inside evidence from this surface.
    return fill&&depth<expected-tolerance&&coherentInside(center,index,witness);
}

int contourStraightCorners(vec2 hull[20],int count,vec3 plane,vec2 anchor,int index,
                           bool fill,vec2 witness[64],float reach,out vec2 result[20]){
    bool straight[8],extend[8];vec2 corners[8];
    for(int i=0;i<count;++i){
        vec2 a=hull[i],b=hull[(i+1)%count],middle=0.5*(a+b),edge=b-a;
        vec2 warped=warpPoint(middle);float step=2.0/float(uMapResolution);
        float pixel=max(length(unwarpPoint(warped+vec2(step,0))-middle),
                        length(unwarpPoint(warped+vec2(0,step))-middle));
        float epsilon=max(1.5*pixel,reach/exp2(float(uContourSteps+2)));
        vec2 outward=vec2(edge.y,-edge.x)/max(length(edge),1e-12);
        straight[i]=!contourSupport(middle+outward*epsilon,plane,index,fill,witness);
        extend[i]=false;
    }
    // An inscribed radial polygon cuts off corners. Extend only two observed
    // straight neighboring edges, and validate the inferred corner in the map.
    for(int i=0;i<count;++i){
        int previous=(i+count-1)%count,next=(i+1)%count;
        if(straight[i]||!straight[previous]||!straight[next])continue;
        vec2 a=hull[previous],d=hull[i]-a;
        vec2 b=hull[next],e=hull[(i+2)%count]-b;
        float determinant=contourCross(d,e);
        if(abs(determinant)<1e-12)continue;
        vec2 corner=a+d*(contourCross(b-a,e)/determinant);
        if(length(corner-anchor)>reach*1.6)continue;
        vec2 inward=anchor-corner;
        float epsilon=reach/exp2(float(uContourSteps+1));
        if(!contourSupport(corner+inward/max(length(inward),1e-12)*epsilon,plane,index,fill,witness))continue;
        extend[i]=true;corners[i]=corner;
    }
    int size=0;
    for(int i=0;i<count;++i){
        if(extend[(i+count-1)%count])continue;
        result[size++]=extend[i]?corners[i]:hull[i];
    }
    return size;
}

int contourSourcePolygon(vec2 p,float receiverDepth,vec3 plane,vec2 anchor,
                         int index,bool fill,vec2 witness[64],float reach,out vec2 source[20]){
    vec2 points[8];
    for(int direction=0;direction<8;++direction){
        vec2 axis=normalize(coherentDirections[direction]);
        float inside=0.0,outside=reach;bool found=false;
        // Coarse radial contour localization, followed by bounded bisection.
        for(int step=1;step<=8;++step){
            float distance=reach*float(step)/8.0;
            if(!contourSupport(anchor+axis*distance,plane,index,fill,witness)){
                outside=distance;found=true;break;
            }
            inside=distance;
        }
        if(found)for(int step=0;step<uContourSteps;++step){
            float middle=0.5*(inside+outside);
            if(contourSupport(anchor+axis*middle,plane,index,fill,witness))inside=middle;
            else outside=middle;
        }
        points[direction]=anchor+axis*(found?0.5*(inside+outside):reach);
    }
    vec2 hull[20];int count=contourHull(points,hull);
    vec2 outline[20];
    count=contourStraightCorners(hull,count,plane,anchor,index,fill,witness,reach,outline);
    if(count<3)return 0;
    vec2 outlinePoints[8];
    for(int i=0;i<8;++i)outlinePoints[i]=outline[i<count?i:0];
    count=contourHull(outlinePoints,hull);
    vec2 clipped[20];
    count=contourClip(hull,count,vec3(-plane.xy,receiverDepth-plane.z-uDepthBias),clipped);
    float scale=uDepthRange*uTanAngle/uExtent;
    for(int i=0;i<count;++i){
        float gap=max(receiverDepth-dot(plane.xy,clipped[i])-plane.z,uDepthBias);
        source[i]=(clipped[i]-p)/(gap*scale);
    }
    float orientation=0.0;
    for(int i=0;i<count;++i)orientation+=contourCross(source[i],source[(i+1)%count]);
    if(orientation<0.0)for(int i=0;i<count/2;++i){
        vec2 swap=source[i];source[i]=source[count-1-i];source[count-1-i]=swap;
    }
    return count;
}

float contourPcss(vec3 light,vec2 receiverSlope,float angle,bool fill,out vec4 diagnostic){
    float scale=uDepthRange*uTanAngle/uExtent;
    float searchRadius=min(max(light.z,0.0)*scale,uMaxRadiusWorld/uExtent);
    vec3 planes[8];vec2 anchors[8],witness[64];
    float gapSum[8],counts[8];int planeCount=0;
    diagnostic=vec4(0,0,0,1);
    for(int i=0;i<uBlockerSamples;++i){
        vec2 point=light.xy;
        if(i>0&&i<9){
            vec2 uv=warpPoint(light.xy)*0.5+0.5;
            point=unwarpPoint((uv+coherentDirections[i-1]/float(uMapResolution))*2.0-1.0);
        }else if(i>=9){
            int remaining=max(uBlockerSamples-9,1),inner=max(remaining/2,1);
            int j=i-9;bool close=j<inner;
            point+=vogel(close?j:j-inner,close?inner:max(remaining-inner,1),angle)*
                   searchRadius*(close?0.2:1.0);
        }
        ivec2 texel;vec2 center;float depth;
        if(!coherentMapPoint(point,texel,center,depth))continue;
        float gap=light.z+dot(receiverSlope,center-light.xy)-depth;
        if(gap<=uDepthBias)continue;
        int match=-1;
        for(int k=0;k<planeCount;++k){
            if(abs(dot(planes[k].xy,center)+planes[k].z-depth)<max(2.0*uDepthBias,2e-6)){match=k;break;}
        }
        if(match<0&&planeCount<uLocalPlanes){
            vec2 gradient=coherentGradient(texel,center,depth);
            vec3 plane=vec3(gradient,depth-dot(gradient,center));
            match=planeCount++;planes[match]=plane;anchors[match]=center;
            counts[match]=0.0;gapSum[match]=0.0;
            for(int e=0;e<8;++e)witness[match*8+e]=center;
        }
        if(match>=0){
            counts[match]+=1.0;gapSum[match]+=gap;
            if(length(center-light.xy)<length(anchors[match]-light.xy))anchors[match]=center;
            for(int e=0;e<8;++e)if(dot(center,coherentDirections[e])>dot(witness[match*8+e],coherentDirections[e]))
                witness[match*8+e]=center;
        }
    }
    if(planeCount==0)return 1.0;
    // These are support probes in map space, not samples of the light source.
    if(fill)for(int i=0;i<uContourSamples;++i){
        vec2 point=light.xy+vogel(i,uContourSamples,angle)*min(searchRadius*3.0,uMaxRadiusWorld/uExtent);
        ivec2 texel;vec2 center;float depth;
        if(!coherentMapPoint(point,texel,center,depth))continue;
        for(int k=0;k<planeCount;++k){
            if(abs(depth-dot(planes[k].xy,center)-planes[k].z)>max(2.0*uDepthBias,2e-6))continue;
            for(int e=0;e<8;++e)if(dot(center,coherentDirections[e])>dot(witness[k*8+e],coherentDirections[e]))
                witness[k*8+e]=center;
        }
    }
    // A square enclosing the entire possible source footprint certifies umbra
    // for the local convex model. This is not a binary center-depth clamp.
    for(int k=0;k<planeCount;++k){
        float gap=light.z-dot(planes[k].xy,light.xy)-planes[k].z;
        float denominator=1.0-scale*length(planes[k].xy);
        if(gap<=uDepthBias||denominator<=0.05)continue;
        float radius=gap*scale/denominator;bool covered=true;
        for(int corner=0;corner<4;++corner){
            vec2 offset=vec2((corner&1)==0?-1.0:1.0,(corner&2)==0?-1.0:1.0)*radius;
            if(!contourSupport(light.xy+offset,planes[k],k,fill,witness)){covered=false;break;}
        }
        if(covered){diagnostic=vec4(gap*scale*uExtent,gap*scale*uExtent,0,3);return 0.0;}
    }
    int nearIndex=0,farIndex=0;
    for(int k=1;k<planeCount;++k){
        if(gapSum[k]/counts[k]<gapSum[nearIndex]/counts[nearIndex])nearIndex=k;
        if(gapSum[k]/counts[k]>gapSum[farIndex]/counts[farIndex])farIndex=k;
    }
    vec2 polygons[80];int sizes[8];
    float reach=max(searchRadius*2.0,2.0/float(uMapResolution));
    float nearArea=0.0,farArea=0.0;
    for(int k=0;k<planeCount;++k){
        vec2 polygon[20];
        sizes[k]=contourSourcePolygon(light.xy,light.z,planes[k],anchors[k],k,fill,witness,reach,polygon);
        for(int i=0;i<sizes[k];++i)polygons[k*10+i]=polygon[i];
        float area=contourDiskArea(polygon,sizes[k]);
        if(k==nearIndex)nearArea=area;
        if(k==farIndex)farArea=area;
    }
    float coverage=contourManyUnion(polygons,sizes,planeCount)/contourPi;
    float visibility=clamp(1.0-coverage,0.0,1.0);
    float nearGap=max(light.z-dot(planes[nearIndex].xy,light.xy)-planes[nearIndex].z,0.0);
    float farGap=max(light.z-dot(planes[farIndex].xy,light.xy)-planes[farIndex].z,0.0);
    bool mixed=nearIndex!=farIndex&&nearArea>1e-4&&farArea>1e-4;
    int state=visibility>=0.999?1:visibility<=0.001?3:mixed?4:2;
    diagnostic=vec4(min(nearGap,farGap)*scale*uExtent,max(nearGap,farGap)*scale*uExtent,
                    min(nearArea,farArea)/contourPi,float(state));
    return visibility;
}
