#version 460 core
layout(location=0) in vec3 aPosition;
layout(location=1) in vec3 aNormal;
uniform mat4 uModel;
out vec3 vsPosition,vsNormal;
void main(){
    vsPosition=(uModel*vec4(aPosition,1)).xyz;
    vsNormal=normalize(mat3(transpose(inverse(uModel)))*aNormal);
    gl_Position=vec4(vsPosition,1);
}
