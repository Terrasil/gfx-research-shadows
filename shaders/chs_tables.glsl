// Immutable original AMD matrices, uploaded once from chs_weights.hpp.
// 645 vec4 weights + 5 vec4 sums = 10400 bytes. This is filter data, not another shadow map.
layout(std140,binding=6) uniform ChsKernelData {vec4 chsData[645];vec4 chsSums[5];};
int chsIndex(int fs){return (fs-7)/2;}
int chsOffset(int fs){return fs==7?0:fs==9?49:fs==11?130:fs==13?251:420;}
vec4 chsControls(int fs,int row,int col){
    if(row<0||col<0||row>=fs||col>=fs)return vec4(0);
    return chsData[chsOffset(fs)+row*fs+col];
}
vec4 chsControlSum(int fs){return chsSums[chsIndex(fs)];}
