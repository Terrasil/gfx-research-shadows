// Exact disk area for the reconstructed convex polygons, not for missing geometry.
const int contourPolygonCapacity=20;
const float contourPi=3.141592653589793;
float contourCross(vec2 a,vec2 b){return a.x*b.y-a.y*b.x;}

float contourDiskEdge(vec2 a,vec2 b){
    vec2 delta=b-a;
    float aa=dot(delta,delta);
    if(aa<1e-20)return 0.0;
    float ab=dot(a,delta),cc=dot(a,a)-1.0;
    float discriminant=ab*ab-aa*cc;
    float cuts[4];cuts[0]=0.0;cuts[3]=1.0;
    cuts[1]=0.0;cuts[2]=0.0;
    if(discriminant>0.0){
        float root=sqrt(discriminant);
        cuts[1]=clamp((-ab-root)/aa,0.0,1.0);
        cuts[2]=clamp((-ab+root)/aa,0.0,1.0);
    }
    float area=0.0;
    for(int i=0;i<3;++i){
        if(cuts[i+1]-cuts[i]<1e-8)continue;
        vec2 p=a+delta*cuts[i],q=a+delta*cuts[i+1];
        vec2 middle=a+delta*(0.5*(cuts[i]+cuts[i+1]));
        area+=dot(middle,middle)<=1.0?
              0.5*contourCross(p,q):0.5*atan(contourCross(p,q),dot(p,q));
    }
    return area;
}

float contourDiskArea(vec2 polygon[20],int count){
    if(count<3)return 0.0;
    float area=0.0;
    for(int i=0;i<count;++i)area+=contourDiskEdge(polygon[i],polygon[(i+1)%count]);
    return clamp(area,0.0,contourPi);
}

// Keep dot(line.xy,p)+line.z >= 0. Input and output are CCW convex polygons.
int contourClip(vec2 polygon[20],int count,vec3 line,out vec2 result[20]){
    if(count==0)return 0;
    int size=0;
    vec2 previous=polygon[count-1];float before=dot(line.xy,previous)+line.z;
    for(int i=0;i<count;++i){
        vec2 point=polygon[i];float after=dot(line.xy,point)+line.z;
        if((before>=0.0)!=(after>=0.0)){
            float t=before/(before-after);
            if(size<contourPolygonCapacity)result[size++]=mix(previous,point,t);
        }
        if(after>=0.0&&size<contourPolygonCapacity)result[size++]=point;
        previous=point;before=after;
    }
    return size;
}

int contourIntersection(vec2 a[20],int na,vec2 b[20],int nb,out vec2 result[20]){
    if(na<3||nb<3)return 0;
    for(int i=0;i<na;++i)result[i]=a[i];
    int count=na;
    for(int edge=0;edge<nb&&count>0;++edge){
        vec2 p=b[edge],direction=b[(edge+1)%nb]-p;
        vec2 inward=vec2(-direction.y,direction.x);
        vec2 clipped[20];
        count=contourClip(result,count,vec3(inward,-dot(inward,p)),clipped);
        for(int i=0;i<count;++i)result[i]=clipped[i];
    }
    return count;
}

float contourUnionArea(vec2 a[20],int na,vec2 b[20],int nb,out vec3 areas){
    areas=vec3(contourDiskArea(a,na),contourDiskArea(b,nb),0);
    if(na>=3&&nb>=3){
        vec2 overlap[20];int count=contourIntersection(a,na,b,nb,overlap);
        areas.z=min(min(areas.x,areas.y),contourDiskArea(overlap,count));
    }
    return clamp(areas.x+areas.y-areas.z,0.0,contourPi);
}

// Integrate only exposed pieces of the polygon boundaries. Convex clipping
// gives one covered interval per other contour, avoiding 2^N inclusion-exclusion.
float contourManyUnion(vec2 polygons[80],int sizes[8],int count){
    float area=0.0;
    for(int shape=0;shape<count;++shape)for(int edge=0;edge<sizes[shape];++edge){
        vec2 a=polygons[shape*10+edge],b=polygons[shape*10+(edge+1)%sizes[shape]];
        vec2 direction=b-a;
        if(dot(direction,direction)<1e-16)continue;
        vec2 intervals[8];int intervalCount=0;
        for(int other=0;other<count;++other){
            if(other==shape||sizes[other]<3)continue;
            float lo=0.0,hi=1.0;bool possible=true;
            for(int e=0;e<sizes[other];++e){
                vec2 p=polygons[other*10+e],d=polygons[other*10+(e+1)%sizes[other]]-p;
                float before=contourCross(d,a-p),after=contourCross(d,b-p);
                float tolerance=1e-7*max(length(d),1e-8);
                if(abs(before)<=tolerance&&abs(after)<=tolerance){
                    // Coincident outer edges belong to the first contour only.
                    if(dot(d,direction)>0.0&&shape<other){possible=false;break;}
                    continue;
                }
                if(before<0.0&&after<0.0){possible=false;break;}
                if(before<0.0)lo=max(lo,before/(before-after));
                if(after<0.0)hi=min(hi,before/(before-after));
                if(lo>=hi){possible=false;break;}
            }
            if(possible&&hi>lo){
                int j=intervalCount++;
                while(j>0&&intervals[j-1].x>lo){intervals[j]=intervals[j-1];--j;}
                intervals[j]=vec2(lo,hi);
            }
        }
        float cursor=0.0;
        for(int i=0;i<intervalCount;++i){
            if(intervals[i].x>cursor)area+=contourDiskEdge(a+direction*cursor,a+direction*intervals[i].x);
            cursor=max(cursor,intervals[i].y);
        }
        if(cursor<1.0)area+=contourDiskEdge(a+direction*cursor,b);
    }
    return clamp(area,0.0,contourPi);
}
