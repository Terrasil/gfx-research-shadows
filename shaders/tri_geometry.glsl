// Raw visibility masks, combined using the source Jacobian and receiver cosine.
// Filtering the combined result commutes with filtering each mask when the
// normalized ray weights are constant over the filter footprint.
vec2 rayMaskSums(vec3 origin,vec2 points[TRI_COUNT],vec3 normal){
    vec3 u=normalize(cross(vec3(0,0,1),uSun)),v=cross(uSun,u);
    float lit=0.0,total=0.0;
    for(int i=0;i<TRI_COUNT;++i){
        vec2 p=points[i];vec3 d=normalize(uSun+uTanAngle*(u*p.x+v*p.y));
        float w=max(dot(normal,d),0.0)*pow(1.0+uTanAngle*uTanAngle*dot(p,p),-1.5);
        total+=w;if(w>0.0&&!blocked(origin,d))lit+=w;
    }
    return vec2(lit,total);
}
// Query the signed support-plane margin of the ray/AABB intersection cone.
// Its sign is an exact ray visibility test; its magnitude is a reconstruction aid.
// The fixed sun and supported angular radii keep direction component signs fixed.
void coneMargins(vec3 origin,vec2 points[TRI_COUNT],out float values[TRI_COUNT]){
    vec3 u=normalize(cross(vec3(0,0,1),uSun)),v=cross(uSun,u);
    for(int k=0;k<TRI_COUNT;++k)values[k]=1e4;
    for(int box=0;box<uBoxCount;++box){
        vec4 lo,hi;sceneBox(box,lo,hi);
        if(int(lo.w+0.5)==uDroppedGroup)continue;
        vec3 a=mix(hi.xyz,lo.xyz,greaterThan(uSun,vec3(0)))-origin;
        vec3 b=mix(lo.xyz,hi.xyz,greaterThan(uSun,vec3(0)))-origin;
        if(any(lessThanEqual(b/uSun,vec3(0))))continue;
        float field[TRI_COUNT];for(int k=0;k<TRI_COUNT;++k)field[k]=-1e4;
        for(int i=0;i<3;++i)for(int j=0;j<3;++j){
            float signPair=sign(uSun[i]*uSun[j]);
            float c=(a[i]*uSun[j]-b[j]*uSun[i])*signPair;
            vec2 g=uTanAngle*vec2(a[i]*u[j]-b[j]*u[i],a[i]*v[j]-b[j]*v[i])*signPair;
            float len=length(g);
            for(int k=0;k<TRI_COUNT;++k){
                float f=len>1e-10?(c+dot(g,points[k]))/len:(c>0.0?1e4:-1e4);
                field[k]=max(field[k],clamp(f,-1e4,1e4));
            }
        }
        for(int k=0;k<TRI_COUNT;++k)values[k]=min(values[k],field[k]);
    }
}
