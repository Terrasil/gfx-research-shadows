uniform int uSharedAdaptive;
uniform float uSharedRadius;
bool rgbaSharedActive(){
    return uPcfStrength>0.0&&uSharedRadius>0.0&&(uSharedAdaptive==0||uPcfMaxRadius>0.0);
}
float rgbaSharedRadiusWorld(vec4 radii,vec4 valid){
    if(!rgbaSharedActive()||dot(valid,vec4(1))<0.5)return 0.0;
    float limit=uSharedRadius*uPcfStrength;
    if(uSharedAdaptive==0)return limit;
    float radius=1e20;
    for(int i=0;i<4;++i)if(valid[i]>0.5&&radii[i]>1e-8)
        radius=min(radius,radii[i]*uExtents[i]);
    return radius<1e19?min(radius,limit):0.0;
}
// Project a disk perpendicular to the sun onto the local receiver plane.
// All channel cameras then observe the SAME displaced world point.
vec3 rgbaSharedOffset(vec2 disk,vec3 normal){
    vec3 sun=normalize(uSun);
    vec3 helper=abs(sun.z)<0.9?vec3(0,0,1):vec3(0,1,0);
    vec3 tangent=normalize(cross(helper,sun)),bitangent=cross(sun,tangent);
    vec3 delta=tangent*disk.x+bitangent*disk.y;
    float den=dot(normal,sun);
    if(abs(den)>=0.1)return delta-sun*(dot(normal,delta)/den);
    // Bound the footprint at grazing incidence, where the plane projection
    // is ill-conditioned. This fallback remains on the receiver plane.
    helper=abs(normal.z)<0.9?vec3(0,0,1):vec3(0,1,0);
    tangent=normalize(cross(helper,normal));bitangent=cross(normal,tangent);
    return tangent*disk.x+bitangent*disk.y;
}
float rgbaSharedFilter(vec3 world,vec3 normal,vec2 gradients[4],float radius,float fallback){
    if(radius<=1e-8)return fallback;
    float total=0.0,count=0.0;
    for(int j=0;j<uPcfSamples;++j){
        vec3 point=world+rgbaSharedOffset(rgbaPcfDisk(j,uPcfSamples)*radius,normal);
        vec4 values=vec4(1),margins=vec4(-1),valid=vec4(0);
        for(int i=0;i<4;++i){
            vec4 clip=uViews[i]*vec4(point,1);vec3 light=clip.xyz/clip.w;light.z=light.z*0.5+0.5;
            margins[i]=1.0-max(abs(light.x),abs(light.y));
            valid[i]=margins[i]>0.0&&light.z>=0.0&&light.z<=1.0?1.0:0.0;
            if(valid[i]>0.5)values[i]=visibility(light,i,gradients[i]);
        }
        // Uncovered samples do not add a spurious lit border.
        if(dot(valid,vec4(1))<0.5)continue;
        total+=rgbaCombinedVisibility(values,margins,valid,uCascades!=0,uCascadeBlend,uAngularMix,uCenterWeight,uMixMode);
        count+=1.0;
    }
    return count>0.0?total/count:fallback;
}
