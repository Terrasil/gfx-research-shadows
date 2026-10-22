layout(binding=0) uniform sampler2D uDepth;
layout(binding=1) uniform sampler2D uNormal;
uniform mat4 uInverseVp;
uniform float uNormalBias;
vec3 worldPosition(vec2 uv,float depth){vec4 p=uInverseVp*vec4(uv*2.0-1.0,depth*2.0-1.0,1);return p.xyz/p.w;}
