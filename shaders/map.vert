#version 460 core
#include "stress_instances.glsl"
layout(location=0) in vec3 aPosition;
layout(location=1) in vec3 aNormal;
uniform mat4 uModel;
uniform int uDroppedGroup;
out vec3 vsPosition,vsNormal;
void main(){
    if(uStressMode!=0){
        vec3 lo,hi,color;int group;stressInstance(gl_InstanceID,lo,hi,color,group);
        vec3 size=hi-lo;
        vsPosition=group==uDroppedGroup?vec3(1e5):(lo+hi)*0.5+aPosition*size;
        vsNormal=normalize(aNormal/size);
        gl_Position=vec4(vsPosition,1);return;
    }
    vsPosition=(uModel*vec4(aPosition,1)).xyz;
    vsNormal=normalize(mat3(transpose(inverse(uModel)))*aNormal);
    gl_Position=vec4(vsPosition,1);
}
