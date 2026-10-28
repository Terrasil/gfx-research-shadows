// Compare first, reconstruct visibility afterwards. Never interpolate unrelated depths.
float compareAt(ivec2 texel,int channel,float receiverDepth,vec2 origin,vec2 gradient){
    if(any(lessThan(texel,ivec2(0)))||any(greaterThanEqual(texel,ivec2(uMapResolution))))return 1.0;
    float d=rgbaDepthAt(texel,channel);
    vec2 p=unwarpPoint((vec2(texel)+0.5)/float(uMapResolution)*2.0-1.0);
    return d>=1.0||receiverDepth+dot(gradient,p-origin)-uDepthBias*uBiasScale[channel]<=d?1.0:0.0;
}
vec4 cubic(float f){float g=1.0-f;return vec4(g*g*g,3.0*f*f*f-6.0*f*f+4.0,-3.0*f*f*f+3.0*f*f+3.0*f+1.0,f*f*f)/6.0;}
float visibility(vec3 light,int channel,vec2 gradient){
    vec2 uv=warpPoint(light.xy)*0.5+0.5;
    if(any(lessThan(uv,vec2(0)))||any(greaterThanEqual(uv,vec2(1)))||light.z<0.0||light.z>1.0)return 1.0;
    if(uSmoothing==0)return compareAt(ivec2(uv*float(uMapResolution)),channel,light.z,light.xy,gradient);
    vec2 t=uv*float(uMapResolution)-0.5,f=fract(t);ivec2 base=ivec2(floor(t));
    if(uSmoothing==3){
        vec4 wx=cubic(f.x),wy=cubic(f.y);float total=0.0;
        for(int y=0;y<4;++y)for(int x=0;x<4;++x)
            total+=wx[x]*wy[y]*compareAt(base+ivec2(x-1,y-1),channel,light.z,light.xy,gradient);
        return total;
    }
    if(uSmoothing==2)f=f*f*(3.0-2.0*f);
    float total=0.0;
    for(int y=0;y<2;++y)for(int x=0;x<2;++x)
        total+=(x==0?1.0-f.x:f.x)*(y==0?1.0-f.y:f.y)*compareAt(base+ivec2(x,y),channel,light.z,light.xy,gradient);
    return total;
}
