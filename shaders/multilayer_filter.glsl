layout(binding=4) uniform sampler2DArray uHierarchy;
uniform int uHierarchyLevels;
float multilayerVisibility(vec2 p,float receiverDepth,vec2 gradient){
    float occlusion=0.0;
    float scale=uDepthRange*uTanAngle/uExtent;
    for(int level=0;level<uHierarchyLevels;++level){
        ivec2 size=textureSize(uHierarchy,level).xy;
        vec2 pixel=(p*0.5+0.5)*vec2(size)-0.5;
        ivec2 base=ivec2(floor(pixel));vec2 fraction=fract(pixel);
        float neighborhood[4];
        for(int child=0;child<4;++child){
            ivec2 q=base+ivec2(child&1,child>>1);
            float coverage=0.0;
            if(all(greaterThanEqual(q,ivec2(0)))&&all(lessThan(q,size))){
                vec2 center=(vec2(q)+0.5)/vec2(size)*2.0-1.0;
                for(int k=0;k<8;++k){
                    vec2 fragment=texelFetch(uHierarchy,ivec3(q,k),level).rg;
                    if(fragment.y<=0.0)break;
                    float gap=receiverDepth+dot(gradient,center-p)-fragment.x;
                    if(gap<=uDepthBias)continue;
                    float radius=min(gap*scale,uMaxRadiusWorld/uExtent);
                    // Mip width approximates a square filter diameter.
                    float lod=clamp(log2(max(radius*float(uMapResolution),1.0)),0.0,float(uHierarchyLevels-1));
                    float weight=max(0.0,1.0-abs(lod-float(level)));
                    coverage+=fragment.y*weight;
                }
            }
            neighborhood[child]=coverage;
        }
        occlusion+=mix(mix(neighborhood[0],neighborhood[1],fraction.x),
                     mix(neighborhood[2],neighborhood[3],fraction.x),fraction.y);
    }
    return clamp(1.0-occlusion,0.0,1.0);
}
