#version 460 core
in vec3 vNormal;
layout(location=0) out vec4 outAlbedo;
layout(location=1) out vec4 outNormal;
uniform vec3 uColor;
uniform int uReceiver;
void main(){outAlbedo=vec4(uColor,1);outNormal=vec4(normalize(vNormal),float(uReceiver));}
