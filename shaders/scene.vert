#version 460 core
#include "stress_instances.glsl"
layout(location=0) in vec3 aPosition;
layout(location=1) in vec3 aNormal;
uniform mat4 uModel,uVp;
uniform vec3 uColor;
out vec3 vNormal;
flat out vec3 vColor;
void main() {
    if(uStressMode!=0){
        vec3 lo,hi,color;int group;stressInstance(gl_InstanceID,lo,hi,color,group);
        vec3 size=hi-lo;
        vNormal=aNormal/size;vColor=color;
        gl_Position=uVp*vec4((lo+hi)*0.5+aPosition*size,1);return;
    }
    vColor=uColor;
    vNormal=mat3(transpose(inverse(uModel)))*aNormal;
    gl_Position=uVp*uModel*vec4(aPosition,1);
}
