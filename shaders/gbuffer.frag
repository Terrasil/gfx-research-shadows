#version 460 core
in vec3 vNormal;
flat in vec3 vColor;
layout(location=0) out vec4 outAlbedo;
layout(location=1) out vec4 outNormal;
uniform int uReceiver;
void main(){outAlbedo=vec4(vColor,1);outNormal=vec4(normalize(vNormal),float(uReceiver));}
