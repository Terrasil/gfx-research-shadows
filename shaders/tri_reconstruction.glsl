#include "tri_area.glsl"
void trianglePoints(float radius,float rotation,out vec2 points[TRI_COUNT]){
    int ring=TRI_COUNT==3?3:TRI_COUNT-1;
    int start=TRI_COUNT==3?0:1;
    if(start==1)points[0]=vec2(0);
    for(int i=0;i<ring;++i){float a=rotation+2.0*triPi*float(i)/float(ring);points[i+start]=radius*vec2(cos(a),sin(a));}
}
float triangulatedVisibility(vec2 points[TRI_COUNT],float values[TRI_COUNT],vec3 normal){
#if TRI_COUNT == 3
    return diskHalfPlane(fitTrianglePlane(points[0],points[1],points[2],values[0],values[1],values[2]));
#else
    const int ring=TRI_COUNT-1;float delta=2.0*triPi/float(ring),lit=0.0,total=0.0;
    vec3 u=normalize(cross(vec3(0,0,1),uSun)),v=cross(uSun,u);
    for(int i=0;i<ring;++i){
        int a=i+1,b=(i+1)%ring+1;
        vec3 plane=fitTrianglePlane(vec2(0),points[a],points[b],values[0],values[a],values[b]);
        vec2 pa=normalize(points[a]),pb=normalize(points[b]);
        float area=litSectorArea(pa,pb,plane,delta);
        // Piecewise-constant irradiance quadrature at each circular sector centroid.
        vec2 sectorCenter=normalize(pa+pb)*(4.0*sin(delta*0.5)/(3.0*delta));
        vec3 d=uSun+uTanAngle*(u*sectorCenter.x+v*sectorCenter.y);
        float w=max(dot(normal,normalize(d)),0.0)*pow(1.0+uTanAngle*uTanAngle*dot(sectorCenter,sectorCenter),-1.5);
        lit+=area*w;total+=delta*0.5*w;
    }
    return total>1e-12?clamp(lit/total,0.0,1.0):1.0;
#endif
}
