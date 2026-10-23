uniform vec3 uSun;
uniform float uTanAngle;
float radicalInverse(uint n){float r=0,s=0.5;while(n!=0u){r+=float(n&1u)*s;n>>=1u;s*=0.5;}return r;}
vec2 sourceDisk(int i){
    float r=sqrt(radicalInverse(uint(i+1)));
    float a=6.28318530718*fract((float(i)+0.5)*0.61803398875);
    return vec2(cos(a),sin(a))*r;
}
vec3 sourceDirection(int i,out float weight){
    vec2 p=sourceDisk(i)*uTanAngle;
    vec3 u=normalize(cross(vec3(0,0,1),uSun)),v=cross(uSun,u);
    weight=pow(1.0+dot(p,p),-1.5);
    return normalize(uSun+u*p.x+v*p.y);
}
