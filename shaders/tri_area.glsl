const float triPi=3.14159265358979323846;
float cross2(vec2 a,vec2 b){return a.x*b.y-a.y*b.x;}
vec3 fitTrianglePlane(vec2 a,vec2 b,vec2 c,float fa,float fb,float fc){
    vec2 u=a-c,v=b-c;float det=cross2(u,v);
    vec2 g=vec2((fa-fc)*v.y-(fb-fc)*u.y,u.x*(fb-fc)-v.x*(fa-fc))/det;
    return vec3(g,fc-dot(g,c));
}
float diskHalfPlane(vec3 plane){
    float len=length(plane.xy);
    if(len<1e-10)return plane.z>=0.0?1.0:0.0;
    float d=clamp(plane.z/len,-1.0,1.0);
    float h=sqrt(max(0.0,1.0-d*d));
    return clamp(0.5+(atan(d,h)+d*h)/triPi,0.0,1.0);
}
// Exact signed area of the intersection between a unit disk and an edge triangle.
float circleEdgeArea(vec2 a,vec2 b){
    vec2 d=b-a;float aa=dot(d,d);if(aa<1e-14)return 0.0;
    float bb=2.0*dot(a,d),cc=dot(a,a)-1.0,disc=bb*bb-4.0*aa*cc;
    float cuts[4];int count=1;cuts[0]=0.0;
    if(disc>0.0){
        float root=sqrt(disc),t0=(-bb-root)/(2.0*aa),t1=(-bb+root)/(2.0*aa);
        if(t0>0.0&&t0<1.0)cuts[count++]=t0;
        if(t1>0.0&&t1<1.0)cuts[count++]=t1;
    }
    cuts[count++]=1.0;float area=0.0;
    for(int i=0;i<count-1;++i){
        vec2 p=a+d*cuts[i],q=a+d*cuts[i+1],mid=(p+q)*0.5;
        // A tangent segment touches the disk at one point but has no interior.
        area+=dot(mid,mid)<1.0-1e-6?0.5*cross2(p,q):0.5*atan(cross2(p,q),dot(p,q));
    }
    return area;
}
// A circumscribed triangle encloses the full circular sector, including its arc.
float litSectorArea(vec2 a,vec2 b,vec3 plane,float sectorAngle){
    float outer=1.0/cos(sectorAngle*0.5);
    vec2 vertices[3];vertices[0]=vec2(0);vertices[1]=a*outer;vertices[2]=b*outer;
    vec2 polygon[5];int count=0;
    for(int i=0;i<3;++i){
        vec2 p=vertices[i],q=vertices[(i+1)%3];
        float fp=dot(plane,vec3(p,1)),fq=dot(plane,vec3(q,1));
        if((fp>=0.0)!=(fq>=0.0))polygon[count++]=mix(p,q,clamp(fp/(fp-fq),0.0,1.0));
        if(fq>=0.0)polygon[count++]=q;
    }
    if(count<3)return 0.0;
    float area=0.0;for(int i=0;i<count;++i)area+=circleEdgeArea(polygon[i],polygon[(i+1)%count]);
    return clamp(area,0.0,sectorAngle*0.5);
}
