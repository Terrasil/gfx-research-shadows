uniform float uWarp;
uniform vec2 uFocus;
vec2 warpGain(){
    vec2 a=(-vec2(1)-uFocus)/(vec2(1)+uWarp*abs(-vec2(1)-uFocus));
    vec2 b=(vec2(1)-uFocus)/(vec2(1)+uWarp*abs(vec2(1)-uFocus));
    return 2.0/(b-a);
}
vec2 warpOffset(){
    vec2 a=(-vec2(1)-uFocus)/(vec2(1)+uWarp*abs(-vec2(1)-uFocus));
    return -vec2(1)-a*warpGain();
}
vec2 warpPoint(vec2 p){return (p-uFocus)*warpGain()/(vec2(1)+uWarp*abs(p-uFocus))+warpOffset();}
vec2 unwarpPoint(vec2 p){vec2 t=p-warpOffset();return t/(warpGain()-uWarp*abs(t))+uFocus;}
