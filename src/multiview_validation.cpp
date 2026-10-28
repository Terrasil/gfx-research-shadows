#include <shadows/renderer.hpp>
#include <glad/gl.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace shadows {
void Renderer::validateMultiViewStorage(const std::filesystem::path& path) {
    const auto savedSettings=settings_;
    const auto savedScene=scene_;
    std::ofstream out(path);
    out<<"case_id,scene,map_backend,cascades,warp,filter,mix,smoothing,storage,depth_max_error,visibility_rmse,visibility_max_error,guide_max_error,array_separate_max_error,map_bytes,layers,textures,gl_error\n"<<std::setprecision(10);
    int index=0;
    for(const char* sceneId:{"mixed-overlap","crossing-four","sloped"})
    for(bool analytic:{false,true})for(bool cascades:{false,true})for(int filter:{0,1,2,3,4}){
        Settings s;
        s.width=96;s.height=64;s.mapResolution=64;
        s.analyticMaps=analytic;s.warp=index%2?8.0f:0.0f;
        s.angle=3.0f;s.rgbaExtent=cascades?4.0f:14.0f;
        s.rgbaCascades=cascades;s.rgbaCascadeScale=2.0f;s.rgbaAngularMix=1.0f;
        s.rgbaPcfMode=filter;s.rgbaPcfSamples=8;s.rgbaPcssBlockerSamples=16;
        s.rgbaPcssMean=index%2;s.rgbaSmoothing=index%4;s.rgbaMixMode=index%5;
        s.rgbaMoments=(filter==1||filter==4)&&index%2==0;
        if(s.rgbaMoments)s.rgbaMixMode=0;
        s.resultBlend=(filter==2||filter==3)?0.5f:0.0f;
        s.rgbaReconstruct=filter==0&&index%2==0;s.rgbaEdgeSearch=4;
        const auto scene=makeScene(sceneId);
        std::vector<float> baselineDepth,baselineVisibility,baselineGuide,arrayVisibility;
        // Final 0 tests switching back without changing resolution, including
        // FBO attachments, channel write masks and previously compiled shaders.
        for(int storage:{0,1,2,0}){
            if(storage==1)s.multiViewStorage=1;
            else applyMultiViewVariant(s,(cascades?2:0)+(storage==2?1:0));
            prepare(scene,s);render(Method::AngularRgba);
            const auto depths=readMultiViewDepth(),guide=diagnostics();
            const auto image=read();
            if(baselineDepth.empty()){
                baselineDepth=depths;baselineVisibility=image.visibility;baselineGuide=guide;
            }
            double depthError=0,visibilityError=0,guideError=0,squared=0,arrayError=0;
            for(std::size_t p=0;p<depths.size();++p){
                if(!std::isfinite(depths[p]))throw std::runtime_error("Nonfinite multi-view depth");
                depthError=std::max(depthError,std::abs(double(depths[p])-baselineDepth[p]));
            }
            for(std::size_t p=0;p<image.visibility.size();++p){
                const double v=image.visibility[p],error=std::abs(v-baselineVisibility[p]);
                if(!std::isfinite(v)||v<0.0||v>1.00001)throw std::runtime_error("Invalid multi-view visibility");
                visibilityError=std::max(visibilityError,error);squared+=error*error;
                if(storage==2)arrayError=std::max(arrayError,std::abs(v-arrayVisibility[p]));
            }
            for(std::size_t p=0;p<guide.size();++p){
                if(!std::isfinite(guide[p]))throw std::runtime_error("Nonfinite multi-view guide");
                guideError=std::max(guideError,std::abs(double(guide[p])-baselineGuide[p]));
            }
            if(storage==1)arrayVisibility=image.visibility;
            const auto error=glGetError();
            const double rmse=std::sqrt(squared/double(image.visibility.size()));
            out<<index<<','<<sceneId<<','<<(analytic?"analytic":"raster")<<','<<int(cascades)<<','<<s.warp<<','<<filter<<','<<s.rgbaMixMode<<','<<s.rgbaSmoothing<<','<<multiViewStorageId(storage)
               <<','<<depthError<<','<<rmse<<','<<visibilityError<<','<<guideError<<','<<arrayError
               <<','<<activeMapBytes(Method::AngularRgba,s)<<','<<activeMapLayers(Method::AngularRgba,s)<<','<<mapTextureCount(Method::AngularRgba,s)<<','<<error<<'\n';
            if(error!=GL_NO_ERROR||depthError>2e-5||rmse>1e-4||visibilityError>0.002||arrayError>1e-5)
                throw std::runtime_error("Multiple light view storage mismatch in case "+std::to_string(index)+" / "+multiViewStorageId(storage));
        }
        ++index;
    }
    if(!out)throw std::runtime_error("Cannot write multi-view storage fixtures");
    prepare(savedScene,savedSettings);
}
} // namespace shadows
