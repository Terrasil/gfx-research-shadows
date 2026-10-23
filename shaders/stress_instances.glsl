// Procedural instancing: same bounds as stressBox() in scene.hpp. No per-object
// uniform loop, upload, SSBO or CPU vector for the large stress populations.
uniform int uStressMode,uStressGrid;
void stressInstance(int index,out vec3 lo,out vec3 hi,out vec3 color,out int group){
    int x=index%uStressGrid,z=index/uStressGrid;
    group=(x+z)%2;
    float cell=12.0/float(uStressGrid),width=cell*0.28;
    vec3 center=vec3((float(x)+0.5)*cell-6.0,0,(float(z)+0.5)*cell-6.0);
    lo=center+vec3(-width*0.5,0.1,-width*0.5);
    hi=center+vec3(width*0.5,group!=0?3.6:0.75,width*0.5);
    color=group!=0?vec3(0.82,0.55,0.19):vec3(0.18,0.46,0.75);
}
