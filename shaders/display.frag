#version 460 core
in vec2 vUv;
layout(location=0) out vec4 outColor;
layout(binding=0) uniform sampler2D uVisibility;
layout(binding=1) uniform sampler2D uNormal;
layout(binding=2) uniform sampler2D uAlbedo;
layout(binding=3) uniform sampler2D uReference;
layout(binding=4) uniform sampler2D uDiagnostic;
uniform int uView,uDiagnosticValid;
uniform vec3 uSun;
void main(){
    float visibility=texture(uVisibility,vUv).r;
    vec3 n=texture(uNormal,vUv).xyz;
    vec3 c;
    if(uView==1)c=vec3(visibility);
    else if(uView==2){
        float e=visibility-texture(uReference,vUv).r;
        c=mix(vec3(0.04),e>0?vec3(1,0.2,0.05):vec3(0.1,0.45,1),clamp(abs(e)*4.0,0.0,1.0));
    }else if(uView==3||uView==4){
        vec4 d=texture(uDiagnostic,vUv);c=vec3(0.12);
        if(uDiagnosticValid!=0&&d.a>0.5){
            if(uView==4)c=vec3(clamp(d.z,0.0,1.0));
            else c=mix(vec3(0.05,0.35,1.0),vec3(1.0,0.5,0.05),clamp(d.x/0.3,0.0,1.0));
        }
    }else{
        if(dot(n,n)<0.1)c=vec3(0.07,0.10,0.14);
        else{vec3 albedo=texture(uAlbedo,vUv).rgb;c=albedo*(0.16+max(dot(normalize(n),uSun),0.0)*visibility*0.84);}
        c=pow(max(c,vec3(0)),vec3(1.0/2.2));
    }
    outColor=vec4(c,1);
}
