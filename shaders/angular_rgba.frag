#version 460 core
#include "screen.glsl"
#include "warp.glsl"
in vec2 vUv;
layout(location=0) out float outVisibility;
layout(location=1) out vec4 outDiagnostic;
#include "rgba_depth_read.glsl"
uniform mat4 uViews[4];
uniform vec3 uAxisX[4],uAxisY[4],uAxisZ[4];
uniform int uMapResolution,uSmoothing;
uniform float uDepthBias,uCenterWeight,uCascadeBlend,uAngularMix;
uniform int uCascades,uMixMode,uPcfSharp,uJointBlend;
uniform vec4 uBiasScale,uExtents,uDepthRanges;
uniform int uReconstruct,uEdgeSearch;
uniform float uEdgeContactRange,uReconstructStrength,uTanAngle,uTriRadius,uTriRotation;
uniform vec3 uSun;
uniform int uMoments;
uniform float uMomentStrength;
#define TRI_COUNT 4
#include "tri_reconstruction.glsl"
#include "rgba_edge.glsl"
#include "rgba_mixing.glsl"
#include "rgba_visibility.glsl"
#include "rgba_pcf.glsl"
#include "rgba_shared.glsl"
#include "rgba_moments.glsl"
#include "rgba_moment_sampling.glsl"
void main(){
    outVisibility=1.0;outDiagnostic=vec4(1);
    float depth=texture(uDepth,vUv).r;if(depth>=1.0)return;
    vec3 n=normalize(texture(uNormal,vUv).xyz),world=worldPosition(vUv,depth)+n*uNormalBias;
    bool momentActive=uMoments!=0&&uMomentStrength>0.0&&(uPcfMode==1||uPcfMode==2||uPcfMode==4);
    RgbaPcfMoment observations[4];
    vec4 values=vec4(1),margins=vec4(-1),valid=vec4(0);
    vec3 lights[4];vec2 gradients[4];vec4 baseValues=vec4(1),radii=vec4(0);float sharpSupport=0.0,jointProtection=0.0,softFootprint=0.0,minFootprint=1e20,coverage=0.0;
    vec2 screenStep=1.0/vec2(textureSize(uDepth,0));
    vec3 dx=worldPosition(vUv+vec2(screenStep.x,0),depth)-worldPosition(vUv,depth);
    vec3 dy=worldPosition(vUv+vec2(0,screenStep.y),depth)-worldPosition(vUv,depth);
    for(int i=0;i<4;++i){
        vec4 clip=uViews[i]*vec4(world,1);vec3 light=clip.xyz/clip.w;light.z=light.z*0.5+0.5;
        margins[i]=1.0-max(abs(light.x),abs(light.y));
        valid[i]=margins[i]>0.0&&light.z>=0.0&&light.z<=1.0?1.0:0.0;
        // Receiver-plane depth gradient in this view's unwarped coordinates.
        vec3 ax=uAxisX[i],ay=uAxisY[i],az=uAxisZ[i];
        float den=dot(n,az);vec2 gradient=abs(den)>1e-7?-0.5*vec2(dot(n,ax),dot(n,ay))/den:vec2(0);
        lights[i]=light;gradients[i]=gradient;
        float momentGap=0.0,momentFootprint=0.0;
        float radius=momentActive||uPcfMode==4?rgbaPcfRadiusInfo(light,i,gradient,momentGap,momentFootprint):(uPcfMode==3&&uSharedAdaptive==0?0.0:rgbaPcfRadius(light,i,gradient));
        radii[i]=radius;
        if(uPcfMode==3){baseValues[i]=visibility(light,i,gradient);values[i]=baseValues[i];}
        else if(momentActive){
            observations[i]=rgbaPcfMomentVisibility(light,i,gradient,radius,momentGap,momentFootprint);
            values[i]=observations[i].visibility;
            vec3 edge=normalize(ax)*observations[i].direction.x+normalize(ay)*observations[i].direction.y;
            vec3 sun=normalize(uSun),sourceU=normalize(cross(vec3(0,0,1),sun)),sourceV=cross(sun,sourceU);
            vec2 direction=vec2(dot(edge,sourceU),dot(edge,sourceV));
            if(length(direction)>1e-7)observations[i].direction=normalize(direction);
            else observations[i].confidence=0.0;
        }else values[i]=rgbaChannelVisibilityAtRadius(light,i,gradient,radius,momentGap);
        float pixelFootprint=max(length((uViews[i]*vec4(dx,0)).xy),length((uViews[i]*vec4(dy,0)).xy));
        if(momentActive&&uPcfMode==1&&uPcfSharp!=0&&valid[i]>0.5&&momentGap>0.0){
            // Suppress only the moment correction near contact. Keep the fixed
            // PCF baseline unchanged, including its chosen filter radius.
            float base=visibility(light,i,gradient),physical=momentGap*uTanAngle/uExtents[i];
            sharpSupport=max(sharpSupport,(1.0-base)*(1.0-smoothstep(0.75,1.5,physical/max(pixelFootprint,1e-7))));
        }
        if(uJointBlend!=0&&valid[i]>0.5){
            coverage+=float(1<<i);
            // Physical penumbra support accumulated over available views. The
            // nearest shadow is protected separately, even inside a broad one.
            float physicalRadius=radius/((uPcfMode==2||uPcfMode==4)?max(uPcfStrength,1e-6):1.0);
            float footprint=physicalRadius/max(pixelFootprint,1e-7);
            softFootprint=max(softFootprint,footprint);
            if(physicalRadius>1e-8)minFootprint=min(minFootprint,footprint);
        }
        bool protect=uPcfMode==2||uPcfMode==4?uPcfStrength>0.0&&uPcfMaxRadius>0.0:uPcfMode==3&&rgbaSharedActive();
        if(protect&&uPcfSharp!=0&&valid[i]>0.5){
            baseValues[i]=visibility(light,i,gradient);
            float gap=rgbaPcfGap(light.xy,light,i,gradient);
            if(gap>0.0){
                float support=rgbaPcfSharpSupport(baseValues[i],gap,uExtents[i],pixelFootprint);
                // PCSS finishes each channel independently before any mixing.
                if(uPcfMode==4)values[i]=mix(values[i],baseValues[i],support);
                if(uPcfMode==3){
                    float physical=gap*uTanAngle/uExtents[i];
                    support=(1.0-baseValues[i])*(1.0-smoothstep(0.75,1.5,physical/max(pixelFootprint,1e-7)));
                }
                sharpSupport=max(sharpSupport,support);
                if(uJointBlend!=0){
                    float nearRadius=gap*uTanAngle/uExtents[i]/max(pixelFootprint,1e-7);
                    jointProtection=max(jointProtection,(1.0-baseValues[i])*(1.0-smoothstep(1.0,4.0,nearRadius)));
                }
            }
        }
    }
    if(uPcfMode==2&&uPcfSharp!=0&&uPcfStrength>0.0&&uPcfMaxRadius>0.0)values=mix(values,baseValues,sharpSupport);
    float angular=-1.0;
    if(uReconstruct!=0&&uReconstructStrength>0.0&&dot(valid,vec4(1))>3.5){
        vec2 points[4];RgbaEdgeInfo edges[4];trianglePoints(uTriRadius,uTriRotation,points);
        for(int i=0;i<4;++i)edges[i]=rgbaEdgeInfo(lights[i],i,gradients[i]);
        float raw=rgbaCombinedVisibility(values,margins,valid,false,0.0,0.0,uCenterWeight,0);
        angular=mix(raw,rgbaReconstructedDisk(points,edges,n,raw,uEdgeContactRange),uReconstructStrength);
    }
    outVisibility=rgbaCombinedVisibilityOverride(values,margins,valid,uCascades!=0,uCascadeBlend,uAngularMix,uCenterWeight,uMixMode,angular);
    if(momentActive&&dot(valid,vec4(1))>3.5&&abs(dot(n,normalize(uSun)))>0.2){
        vec2 points[4];trianglePoints(uTriRadius,uTriRotation,points);
        vec2 reconstructed=rgbaMomentReconstruct(points,observations);
        float angularWeight=uCascades!=0?uAngularMix:1.0;
        float confidence=reconstructed.y*uMomentStrength*(1.0-sharpSupport)*angularWeight;
        outVisibility=mix(outVisibility,reconstructed.x,confidence);
    }
    if(uPcfMode==3){
        float radius=rgbaSharedRadiusWorld(radii,valid);
        float center=rgbaCombinedVisibility(baseValues,margins,valid,uCascades!=0,uCascadeBlend,uAngularMix,uCenterWeight,uMixMode);
        outVisibility=mix(rgbaSharedFilter(world,n,gradients,radius,center),center,sharpSupport);
        float footprint=max(length(rgbaSharedOffset(vec2(radius,0),n)),length(rgbaSharedOffset(vec2(0,radius),n)));
        softFootprint=footprint/max(max(length(dx),length(dy)),1e-7);
    }
    // Conflicting widths signal different blocker layers. Prefer the nearest
    // observed support there, rather than borrowing the widest blur.
    if(uPcfMode!=3&&minFootprint<1e19){
        float disagreement=smoothstep(1.5,3.0,softFootprint/max(minFootprint,1e-7));
        softFootprint=mix(softFootprint,minFootprint,disagreement);
    }
    // The optional joint pass uses a guide built from all four views, rather
    // than interpreting one channel's visibility as a blocker distance.
    outDiagnostic=uJointBlend!=0?vec4(softFootprint,max(sharpSupport,jointProtection),coverage,1):values;
}
