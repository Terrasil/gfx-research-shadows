RgbaPcfMoment rgbaPcfMomentVisibility(vec3 light,int channel,vec2 gradient,float radius,float gap,float footprint){
    if(radius<=1e-8)return RgbaPcfMoment(rgbaChannelVisibilityAtRadius(light,channel,gradient,radius,gap),vec2(0),0.0,0.0,gap,0.0);
    float sum=0.0,count=0.0;vec2 mean=vec2(0),lit=vec2(0);vec3 second=vec3(0);
    for(int i=0;i<uPcfSamples;++i){
        vec2 point=rgbaPcfDisk(i,uPcfSamples),delta=point*radius;
        vec3 sampleLight=vec3(light.xy+delta,light.z+dot(gradient,delta));
        if(!rgbaPcfCovered(sampleLight))continue;
        float value=visibility(sampleLight,channel,gradient);
        sum+=value;count+=1.0;mean+=point;lit+=point*value;
        second+=vec3(point.x*point.x,point.x*point.y,point.y*point.y);
    }
    if(count<=0.0)return RgbaPcfMoment(visibility(light,channel,gradient),vec2(0),0.0,0.0,gap,0.0);
    float physical=gap*uTanAngle/uExtents[channel];
    RgbaPcfMoment q=rgbaMomentFromSums(sum/count,mean/count,lit/count,second/count,radius/max(physical,1e-8),gap,count>=float(uPcfSamples)?1.0:0.0);
    // The continuous disk model needs a resolved penumbra in this map.
    q.confidence*=smoothstep(1.0,3.0,physical/max(footprint,1e-8));
    return q;
}
