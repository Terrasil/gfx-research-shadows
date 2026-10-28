// The supplied PNG is scalar: RGB channels are identical. Two distant reads
// rotate complete Vogel disks; this is not N independent blue-noise vectors.
layout(binding=5) uniform sampler2D uBlueNoise;
uniform float uBlueNoiseTile;
ivec2 blueNoiseTexel(vec2 worldXZ,uint seed){
    ivec2 size=textureSize(uBlueNoise,0);
    ivec2 p=min(ivec2(floor(fract(worldXZ/uBlueNoiseTile)*vec2(size))),size-1);
    uvec2 shift=uvec2(pcssHash(seed),pcssHash(seed^0x68bc21ebu))%uvec2(size);
    return (p+ivec2(shift))%size;
}
vec2 worldBlueNoiseAngles(vec2 worldXZ,uint seed){
    ivec2 size=textureSize(uBlueNoise,0),p=blueNoiseTexel(worldXZ,seed);
    vec2 code=vec2(texelFetch(uBlueNoise,p,0).r,
                   texelFetch(uBlueNoise,(p+size/2)%size,0).r);
    // Preserve the R8 bins: no interpolation, mipmaps or screen-space jitter.
    return (floor(code*255.0+0.5)+0.5)*(6.283185307179586/256.0);
}
