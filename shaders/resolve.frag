#version 460 core
in vec2 vUv;
layout(location=0) out float outVisibility;
layout(binding=0) uniform sampler2D uAccum;
void main(){vec2 s=texture(uAccum,vUv).rg;outVisibility=s.y>1e-12?s.x/s.y:1.0;}
