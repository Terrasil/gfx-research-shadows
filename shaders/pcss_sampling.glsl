// PCG RXS-M-XS integer hash. Jarzynski and Olano, JCGT 9(3), 2020.
uint pcssHash(uint inputValue){
    uint state=inputValue*747796405u+2891336453u;
    uint word=((state>>((state>>28u)+4u))^state)*277803737u;
    return (word>>22u)^word;
}
uint diskSeed(uvec2 pixel,uint seed,uint stream){
    return pcssHash(pixel.x^pcssHash(pixel.y)^pcssHash(seed)^((stream+1u)*0x85ebca6bu));
}
float diskRandom(uint value){return float(pcssHash(value)>>8u)*(1.0/16777216.0);}
vec2 vogel(int i,int count,float angle){
    float r=sqrt((float(i)+0.5)/float(count)),a=float(i)*2.39996322972865+angle;
    return vec2(cos(a),sin(a))*r;
}
vec2 diskSample(int i,int count,float angle,int pattern,uint key){
    if(pattern==0)return vogel(i,count,angle);
    uint value=key+uint(i)*0x9e3779b9u;
    vec2 xi=vec2(diskRandom(value),diskRandom(value+0x68bc21ebu));
    float area=xi.x,azimuth=xi.y;
    if(pattern==2){
        // Annuli contain an integer number of equal-area angular strata.
        // Every cell has disk area pi/count, including non-square sample counts.
        int rings=max(1,int(floor(sqrt(float(count)/3.141592653589793)+0.5)));
        int begin=0,end=count;
        for(int ring=1;ring<=9;++ring){
            if(ring>rings)break;
            int boundary=(2*count*ring*ring+rings*rings)/(2*rings*rings);
            if(i<boundary){end=boundary;break;}
            begin=boundary;
        }
        int cells=end-begin;
        area=(float(begin)+float(cells)*xi.x)/float(count);
        azimuth=(float(i-begin)+xi.y)/float(cells);
    }
    float r=sqrt(area),a=6.283185307179586*azimuth+angle;
    return vec2(cos(a),sin(a))*r;
}
