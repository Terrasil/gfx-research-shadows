// All inputs are per-view visibility, never raw cross-view depth.
uniform float uGeometricBlend;
float rgbaGeometricVisibility(vec4 values,vec4 weights){
    float logarithm=0.0;
    for(int i=0;i<4;++i)if(weights[i]>0.0){
        // Zero visibility is a real zero. Zero-weight channels cannot extinguish light.
        if(values[i]<=0.0)return 0.0;
        logarithm+=weights[i]*log(min(values[i],1.0));
    }
    return exp(logarithm);
}
float rgbaCombinedVisibilityOverride(vec4 values,vec4 margins,vec4 valid,bool cascades,float width,float angularMix,float centerWeight,int mode,float angularOverride){
    float validCount=dot(valid,vec4(1));if(validCount<0.5)return 1.0;
    vec4 weights=vec4(centerWeight,vec3((1.0-centerWeight)/3.0))*valid;
    float total=dot(weights,vec4(1));
    // E.g. central weight=1 outside R: fall back to the available coarser views.
    if(total<=1e-7)weights=valid/validCount;else weights/=total;
    float angular=dot(weights,values);
    if(mode==1){angular=1.0;for(int i=0;i<4;++i)if(weights[i]>0.0)angular=min(angular,values[i]);}
    if(mode==2){angular=1.0;for(int i=0;i<4;++i)if(weights[i]>0.0)angular*=values[i];}
    if(mode==3||(mode==4&&uGeometricBlend>0.0)){
        float geometric=min(angular,rgbaGeometricVisibility(values,weights));
        angular=mode==3||uGeometricBlend>=1.0?geometric:mix(angular,geometric,clamp(uGeometricBlend,0.0,1.0));
    }
    if(mode==0&&angularOverride>=0.0)angular=angularOverride;
    if(!cascades)return clamp(angular,0.0,1.0);
    float remaining=1.0,cascade=0.0;
    for(int i=0;i<4;++i){
        if(valid[i]<0.5)continue;
        bool coarser=false;for(int j=i+1;j<4;++j)coarser=coarser||valid[j]>0.5;
        float take=coarser&&width>0.0?smoothstep(0.0,width,margins[i]):1.0;
        cascade+=remaining*take*values[i];remaining*=1.0-take;
    }
    // Uncovered pixels are lit. Valid last coverage consumes all residual weight.
    cascade+=remaining;
    return clamp(mix(cascade,angular,angularMix),0.0,1.0);
}

float rgbaCombinedVisibility(vec4 values,vec4 margins,vec4 valid,bool cascades,float width,float angularMix,float centerWeight,int mode){
    return rgbaCombinedVisibilityOverride(values,margins,valid,cascades,width,angularMix,centerWeight,mode,-1.0);
}
