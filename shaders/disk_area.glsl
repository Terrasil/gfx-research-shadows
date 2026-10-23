// Signed area in [0,x] x [0,y] intersected with the unit disk.
float diskPrimitive(float x,float y){
    vec2 p=clamp(abs(vec2(x,y)),0.0,1.0);
    float area=p.x*p.y;
    if(dot(p,p)>1.0)area=0.5*(dot(p,sqrt(max(vec2(0),1.0-p*p)))+atan(p.x,sqrt(max(0.0,1.0-p.x*p.x)))+atan(p.y,sqrt(max(0.0,1.0-p.y*p.y)))-1.5707963267948966);
    return sign(x)*sign(y)*area;
}
float diskRectangle(vec2 lo,vec2 hi){
    if(any(greaterThanEqual(lo,hi))||any(greaterThanEqual(lo,vec2(1)))||any(lessThanEqual(hi,vec2(-1))))return 0.0;
    return max(0.0,diskPrimitive(hi.x,hi.y)-diskPrimitive(lo.x,hi.y)-diskPrimitive(hi.x,lo.y)+diskPrimitive(lo.x,lo.y));
}
