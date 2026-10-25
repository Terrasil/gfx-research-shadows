#include <cmath>
#include <fstream>
#include <gfx/research/orbit_camera.hpp>
#include <gfx/research/primitives.hpp>
#include <gfx/research/screenshot.hpp>
#include <glad/gl.h>
#include <iomanip>
#include <shadows/chs_weights.hpp>
#include <shadows/renderer.hpp>
#include <sstream>
#include <stdexcept>

namespace shadows {
namespace {
constexpr std::array<const char*, methodCount> ids = {"hard", "pcf", "pcss", "arr", "geometric", "dpcf", "chs", "geometric-layered", "vsm", "csm", "esm", "arr-triangulation", "angular-rgba", "gsm"};
constexpr std::array<const char*, methodCount> labels = {"Hard shadow",
                                                         "Fixed PCF",
                                                         "PCSS mean blocker",
                                                         "Geometry reference (ARR exact boxes)",
                                                         "PCSS geometric mean",
                                                         "DPCF (Filament v1.23.1)",
                                                         "CHS (AMD ShadowFX fixed gather4)",
                                                         "Geometric depth bands (paired samples)",
                                                         "VSM (Gaussian moments)", "CSM (Fourier convolution)",
                                                         "ESM (exponential + fallback)", "Geometry reference (ARR triangulation)", "Multiple light view (center + 3 views)", "GSM (prefiltered geometric mean)"};
std::string loadSource(const std::filesystem::path& p, int depth = 0) {
    if (depth > 8)
        throw std::runtime_error("Shader include nesting exceeded");
    std::ifstream in(p);
    if (!in)
        throw std::runtime_error("Cannot read shader: " + p.string());
    std::string line, out;
    while (std::getline(in, line)) {
        if (line.starts_with("#include \"")) {
            const auto end = line.find('"', 10);
            out += loadSource(p.parent_path() / line.substr(10, end - 10), depth + 1);
        } else
            out += line + '\n';
    }
    return out;
}
void loadShader(gfx::research::Shader& target,
                std::initializer_list<std::pair<gfx::research::ShaderStage, const char*>> names,
                const std::string& defines = {}) {
    std::vector<gfx::research::ShaderSource> sources;
    for (auto [stage, name] : names) {
        auto source = loadSource(std::filesystem::path(GFX_SHADOWS_SOURCE_DIR) / "shaders" / name);
        if (!defines.empty())
            source.insert(source.find('\n') + 1, defines);
        sources.push_back({stage, std::move(source)});
    }
    if (!target.load(sources))
        throw std::runtime_error(target.last_error());
}
glm::mat4 boxModel(const Box& b) {
    return glm::translate(glm::mat4(1), (b.lo + b.hi) * 0.5f) * glm::scale(glm::mat4(1), b.hi - b.lo);
}
std::vector<float> readTexture(unsigned texture, int width, int height, GLenum format, int channels) {
    std::vector<float> v(std::size_t(width) * std::size_t(height) * std::size_t(channels));
    glGetTextureImage(texture, 0, format, GL_FLOAT, GLsizei(v.size() * sizeof(float)), v.data());
    return v;
}
} // namespace
void applySoftPackedPreset(Settings& s) {
    s.packedIntervals = true;
    s.smoothing = 3;
    s.samplePattern = 2;
    s.filterSamples = 64;
    s.blockerSamples = 64;
    s.rotateSamples = false;
    s.sharedSamplePattern = true;
    s.layerMinFilterTexels = 1.0f;
}
bool usesPcss(Method method) {
    return method==Method::Pcss||method==Method::Geometric||method==Method::GeometricLayered;
}
bool usesPackedDepth(Method method,const Settings& s) { return usesPcss(method)&&s.packedIntervals; }
bool usesResultBlend(Method method,const Settings& s) {
    const bool angular=method==Method::AngularRgba&&s.rgbaPcfMode!=0&&s.rgbaPcfStrength>0.0f&&
        (s.rgbaPcfMode==1?(s.rgbaPcfWorldUnits?s.rgbaPcfWorldRadius:s.rgbaPcfRadius)>0.0f:s.rgbaPcfMode==3?s.rgbaSharedRadius>0.0f&&(!s.rgbaSharedAdaptive||s.rgbaPcfMaxRadius>0.0f):s.rgbaPcfMaxRadius>0.0f);
    return (usesPcss(method)||angular)&&s.resultBlend>0.0f;
}
std::uint64_t resultBlendBytes(Method method,const Settings& s) {
    return usesResultBlend(method,s)?std::uint64_t(s.width)*s.height*4u:0u;
}
void applyCleanBlueNoisePreset(Settings& s) {
    s.samplePattern=3;s.smoothing=3;s.rotateSamples=false;s.sharedSamplePattern=false;
    s.packedIntervals=true;s.contactGuard=1.0f;s.contactMaxGap=1.0f;
    s.resultBlend=0.8f;s.resultBlendRadius=1;s.resultBlendSigma=0.2f;
    s.layerMinFilterTexels=0.0f;
}
void applySmoothAngularPreset(Settings& s) {
    s.rgbaMoments=false;
    s.rgbaPcfMode=2;s.rgbaPcfSamples=16;s.rgbaPcfStrength=0.5f;
    s.rgbaPcfSharp=true;s.rgbaPcfMaxRadius=16;s.rgbaReconstruct=false;
    s.rgbaSmoothing=1;s.rgbaMixMode=0;s.rgbaAngularMix=1;
    s.resultBlend=0.85f;s.resultBlendRadius=2;s.resultBlendSigma=0.35f;
}
void applySharedAngularPreset(Settings& s) {
    applySmoothAngularPreset(s);s.rgbaPcfMode=3;s.resultBlend=0.0f;
    s.rgbaSharedAdaptive=true;s.rgbaSharedRadius=0.25f;
}
void applyMomentAngularPreset(Settings& s) {
    applySmoothAngularPreset(s);s.rgbaMoments=true;s.rgbaMomentStrength=0.5f;
    s.rgbaPcfSamples=32;s.resultBlend=0.0f;
}
void applyFixedAngularPreset(Settings& s) {
    s.rgbaPcfMode=1;s.rgbaPcfSamples=32;s.rgbaPcfStrength=0.5f;s.rgbaPcfRadius=2.0f;
    s.rgbaPcfWorldUnits=false;s.rgbaPcfWorldRadius=0.25f;s.rgbaPcfMaxRadius=16;s.rgbaPcfSharp=true;
    s.rgbaMoments=false;s.rgbaMomentStrength=0.5f;s.rgbaReconstruct=false;s.resultBlend=0;
    s.rgbaCascades=true;s.rgbaExtent=14;s.rgbaCascadeScale=4;s.rgbaCascadeBlend=0;
    s.rgbaAngularMix=1;s.rgbaCenterWeight=0.25f;s.rgbaSmoothing=1;s.rgbaMixMode=0;s.rgbaGeometricBlend=0.25f;
}
void applyWorldAngularPreset(Settings& s) {
    applyFixedAngularPreset(s);s.rgbaPcfWorldUnits=true;
}
void applyPcssAngularPreset(Settings& s) {
    applyFixedAngularPreset(s);s.rgbaPcfMode=4;
    s.rgbaPcssBlockerSamples=32;s.rgbaPcssMean=0;s.rgbaPcssSearchRadius=3.0f;
}
int multiViewVariant(const Settings& s) {
    if(s.multiViewStorage==1)return -1;
    return (s.rgbaCascades?2:0)+(s.multiViewStorage==2?1:0);
}
const char* multiViewVariantId(int variant) {
    constexpr std::array<const char*,4> names={"rgba","separate","rgba-cascaded","separate-cascaded"};
    return names.at(std::size_t(variant));
}
const char* multiViewVariantId(const Settings& s) {
    const int variant=multiViewVariant(s);
    return variant>=0?multiViewVariantId(variant):s.rgbaCascades?"array-cascaded":"array";
}
const char* multiViewVariantLabel(int variant) {
    constexpr std::array<const char*,4> names={"Angular RGBA (one texture)","Angular separate (one texture per view)",
        "Angular RGBA cascaded (one texture)","Angular separate cascaded (one texture per view)"};
    return names.at(std::size_t(variant));
}
void applyMultiViewVariant(Settings& s,int variant) {
    if(variant<0||variant>=multiViewVariantCount)throw std::runtime_error("Invalid multiple light view variant");
    s.multiViewStorage=variant%2==0?0:2;
    s.rgbaCascades=variant>=2;
}
bool usesMomentReconstruction(Method method,const Settings& s){
    const bool radius=s.rgbaPcfMode==1?(s.rgbaPcfWorldUnits?s.rgbaPcfWorldRadius:s.rgbaPcfRadius)>0.0f:
        (s.rgbaPcfMode==2||s.rgbaPcfMode==4)&&s.rgbaPcfMaxRadius>0.0f;
    return method==Method::AngularRgba&&s.rgbaMoments&&s.rgbaMomentStrength>0.0f&&
        radius&&s.rgbaPcfStrength>0.0f&&
        s.rgbaOffset&&s.rgbaRadius>=0.1f&&s.rgbaMixMode==0&&(!s.rgbaCascades||s.rgbaAngularMix>0.0f);
}
bool usesPrefilter(Method method) {
    return method == Method::Vsm || method == Method::Csm || method == Method::Esm || method == Method::Gsm;
}
bool isAngularRayMethod(Method method) {
    return method==Method::Reference||method==Method::ArrTriangulation;
}
int angularSamples(Method method,const Settings& s){
    return method==Method::Reference?(s.arrSamples>0?s.arrSamples:s.referenceSamples):method==Method::ArrTriangulation?s.triSamples:0;
}
int angularSmoothing(Method method,const Settings& s){
    return isAngularRayMethod(method)&&s.arrSmoothingRadius>0?s.arrSmoothing:0;
}
int shadowPasses(Method method,const Settings& s){
    return 1+(angularSmoothing(method,s)!=0?1:0)+(usesResultBlend(method,s)?1:0);
}
std::uint64_t auxiliaryMapBytes(Method method, const Settings& settings) {
    const std::uint64_t pixels = std::uint64_t(settings.mapResolution) * settings.mapResolution;
    return pixels * (method == Method::Vsm ? 16u : (method == Method::Esm || method == Method::Gsm) ? 8u : method == Method::Csm ? 16u * settings.csmTerms : 0u);
}
int prefilterPasses(Method method) {
    return usesPrefilter(method) ? 3 : 0;
}
void Renderer::preparePrefilter(Method method) {
    const int slot = method==Method::Gsm?3:int(method)-int(Method::Vsm);
    const int layers = method == Method::Csm ? settings_.csmTerms / 2 : 1;
    const GLenum format = method == Method::Vsm ? GL_RG32F : (method == Method::Esm || method == Method::Gsm) ? GL_R32F : GL_RGBA32F;
    if (!basisShaders_[slot].valid()) {
        const std::string define = "#define BASIS_METHOD " + std::to_string(slot) + "\n";
        loadShader(basisShaders_[slot], {{gfx::research::ShaderStage::Compute, "basis_encode.comp"}}, define);
        loadShader(blurShaders_[slot], {{gfx::research::ShaderStage::Compute, "basis_blur.comp"}}, define);
    }
    if (basisMethod_ != method || basisResolution_ != settings_.mapResolution || basisLayers_ != layers) {
        glDeleteTextures(2, basisTextures_);
        glCreateTextures(GL_TEXTURE_2D_ARRAY, 2, basisTextures_);
        for (unsigned texture : basisTextures_) {
            glTextureStorage3D(texture, 1, format, settings_.mapResolution, settings_.mapResolution, layers);
            glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTextureParameteri(texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTextureParameteri(texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
        basisMethod_ = method;
        basisResolution_ = settings_.mapResolution;
        basisLayers_ = layers;
    }
}
void Renderer::prefilter(Method method) {
    const int slot = method==Method::Gsm?3:int(method)-int(Method::Vsm), size = settings_.mapResolution;
    const GLenum format = method == Method::Vsm ? GL_RG32F : (method == Method::Esm || method == Method::Gsm) ? GL_R32F : GL_RGBA32F;
    auto& encode = basisShaders_[slot];
    encode.bind();
    encode.set_int("uSize", size);
    encode.set_int("uLayers", basisLayers_);
    encode.set_float("uExponent", settings_.esmExponent);
    glBindTextureUnit(2, mapTexture_);
    glBindImageTexture(0, basisTextures_[0], 0, GL_TRUE, 0, GL_WRITE_ONLY, format);
    glDispatchCompute((size + 7) / 8, (size + 7) / 8, basisLayers_);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    auto& blur = blurShaders_[slot];
    blur.bind();
    blur.set_int("uSize", size);
    blur.set_int("uLayers", basisLayers_);
    blur.set_int("uRadius", settings_.prefilterRadius);
    for (int axis = 0; axis < 2; ++axis) {
        blur.set_int("uHorizontal", axis == 0 ? 1 : 0);
        glBindTextureUnit(3, basisTextures_[axis]);
        glBindImageTexture(0, basisTextures_[1 - axis], 0, GL_TRUE, 0, GL_WRITE_ONLY, format);
        glDispatchCompute((size + 7) / 8, (size + 7) / 8, basisLayers_);
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }
}
void Renderer::validatePrefilter(const std::filesystem::path& path) {
    const auto savedSettings = settings_;
    const unsigned savedMap = mapTexture_;
    std::array<float, 32 * 32> depths;
    for (int y = 0; y < 32; ++y)
        for (int x = 0; x < 32; ++x)
            depths[y * 32 + x] = float(((x * 13 + y * 7 + x * y * 3) % 89) + 5) / 100.0f;
    glCreateTextures(GL_TEXTURE_2D_ARRAY, 1, &mapTexture_);
    glTextureStorage3D(mapTexture_, 1, GL_DEPTH_COMPONENT32F, 32, 32, 1);
    glTextureSubImage3D(mapTexture_, 0, 0, 0, 0, 32, 32, 1, GL_DEPTH_COMPONENT, GL_FLOAT, depths.data());
    glTextureParameteri(mapTexture_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(mapTexture_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    settings_.mapResolution = 32;
    settings_.prefilterRadius = 3;
    settings_.csmTerms = 8;
    settings_.esmExponent = 80;
    std::ofstream out(path);
    out << "method,case_id,x,y,t,visibility";
    for (int c = 0; c < 16; ++c) out << ",b" << c;
    out << ",radius,sharpness\n" << std::setprecision(10);
    unsigned buffer = 0;
    std::array<float, 128> visibility;
    glCreateBuffers(1, &buffer);
    glNamedBufferStorage(buffer, sizeof(visibility), nullptr, GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);
    for(int slot=0;slot<4;++slot){
        const auto method=slot==3?Method::Gsm:Method(int(Method::Vsm)+slot);
        if(slot==3){
            for(int y=0;y<32;++y)for(int x=0;x<32;++x){
                float& d=depths[std::size_t(y*32+x)];
                if((x+y)%11==0)d=1.0f;else if((x+y)%13==0)d=0.0f;
            }
            glTextureSubImage3D(mapTexture_,0,0,0,0,32,32,1,GL_DEPTH_COMPONENT,GL_FLOAT,depths.data());
        }
        const std::vector<int> radii=slot==3?std::vector<int>{0,3,16}:std::vector<int>{3};
        const std::vector<float> powers=slot==3?std::vector<float>{1,64,128}:std::vector<float>{0};
        for(int radius:radii)for(float power:powers){
            settings_.prefilterRadius=radius;
            preparePrefilter(method);prefilter(method);
            gfx::research::Shader shader;
            loadShader(shader,{{gfx::research::ShaderStage::Compute,"basis_validation.comp"}},"#define BASIS_METHOD "+std::to_string(slot)+"\n");
            shader.bind();shader.set_float("uGsmSharpness",power);
            glBindTextureUnit(3,basisTextures_[0]);glDispatchCompute(128,1,1);
            glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT|GL_TEXTURE_UPDATE_BARRIER_BIT);
            glGetNamedBufferSubData(buffer,0,sizeof(visibility),visibility.data());
            const int channels=slot==0?2:slot==1?4:1;
            std::vector<float> data(std::size_t(32*32*basisLayers_*channels));
            glGetTextureImage(basisTextures_[0],0,channels==1?GL_RED:channels==2?GL_RG:GL_RGBA,GL_FLOAT,GLsizei(data.size()*sizeof(float)),data.data());
            for(int i=0;i<128;++i){
                const int x=(i*7)%32,y=(i*13+i/32)%32;
                out<<methodId(method)<<','<<i<<','<<x<<','<<y<<','<<float(i%17)/16.0f<<','<<visibility[std::size_t(i)];
                for(int c=0;c<16;++c){
                    float value=0;
                    if(c<channels*basisLayers_)value=data[std::size_t(((c/channels)*32*32+y*32+x)*channels+c%channels)];
                    out<<','<<value;
                }
                out<<','<<radius<<','<<power<<'\n';
            }
        }
    }
    out.flush();
    if (!out) throw std::runtime_error("Cannot write prefilter fixtures");
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
    glDeleteBuffers(1, &buffer);
    glDeleteTextures(1, &mapTexture_);
    mapTexture_ = savedMap;
    settings_ = savedSettings;
}
int activeMapLayers(Method m, const Settings& s) {
    return isAngularRayMethod(m) ? 0 : m==Method::AngularRgba&&s.multiViewStorage!=0 ? 4 : 1;
}
const char* multiViewStorageId(int storage) {
    constexpr std::array<const char*,3> names={"rgba","array","separate"};
    return names.at(std::size_t(storage));
}
const char* mapStorageId(Method m,const Settings& s) {
    if(m==Method::AngularRgba){
        constexpr std::array<const char*,3> names={"rgba32f-angular","d32f-array-angular","d32f-separate-angular"};
        return names.at(std::size_t(s.multiViewStorage));
    }
    return isAngularRayMethod(m)?"none":usesPackedDepth(m,s)?"r32ui-unorm16-minmax":"d32f";
}
int mapTextureCount(Method m,const Settings& s) {
    return isAngularRayMethod(m)?0:m==Method::AngularRgba&&s.multiViewStorage==2?4:1;
}
void Renderer::validateIntervals(const std::filesystem::path& path) {
    unsigned texture=0;
    glCreateTextures(GL_TEXTURE_2D_ARRAY,1,&texture);
    glTextureStorage3D(texture,1,GL_R32UI,16,16,1);
    const unsigned empty=0xffff0000u;
    glClearTexImage(texture,0,GL_RED_INTEGER,GL_UNSIGNED_INT,&empty);
    gfx::research::Shader shader;
    loadShader(shader,{{gfx::research::ShaderStage::Compute,"interval_validation.comp"}});
    shader.bind();
    glBindImageTexture(0,texture,0,GL_TRUE,0,GL_READ_WRITE,GL_R32UI);
    glDispatchCompute(64,1,1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT|GL_TEXTURE_UPDATE_BARRIER_BIT);
    std::array<unsigned,256> values;
    glGetTextureImage(texture,0,GL_RED_INTEGER,GL_UNSIGNED_INT,sizeof(values),values.data());
    std::ofstream out(path);out<<"pixel,packed\n";
    for(std::size_t i=0;i<values.size();++i)out<<i<<','<<values[i]<<'\n';
    glBindImageTexture(0,0,0,GL_TRUE,0,GL_READ_WRITE,GL_R32UI);
    glDeleteTextures(1,&texture);
    if(!out)throw std::runtime_error("Cannot save interval fixtures");
}
void Renderer::validateAngular(const std::filesystem::path& path) {
    unsigned buffer=0;std::array<glm::vec4,96*3> values;
    glCreateBuffers(1,&buffer);glNamedBufferStorage(buffer,sizeof(values),nullptr,GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,buffer);
    std::ofstream out(path);out<<"kind,case_id,queries,a,b,c,value,ox,oy,oz,px,py\n"<<std::setprecision(10);
    const auto scene=makeScene("mixed-overlap");
    for(int count:{3,4,7,13,25}){
        gfx::research::Shader shader;
        loadShader(shader,{{gfx::research::ShaderStage::Compute,"angular_validation.comp"}},"#define TRI_COUNT "+std::to_string(count)+"\n");
        shader.bind();shader.set_vec3("uSun",glm::vec3(sunDirection()));shader.set_float("uTanAngle",std::tan(glm::radians(1.5f)));
        shader.set_int("uBoxCount",int(scene.boxes.size()));shader.set_int("uDroppedGroup",-1);
        for(std::size_t i=0;i<scene.boxes.size();++i){
            shader.set_vec4(("uBoxLo["+std::to_string(i)+"]").c_str(),glm::vec4(scene.boxes[i].lo,float(scene.boxes[i].group)));
            shader.set_vec4(("uBoxHi["+std::to_string(i)+"]").c_str(),glm::vec4(scene.boxes[i].hi,0));
        }
        shader.set_int("uTestKind",0);glDispatchCompute(1,1,1);glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        glGetNamedBufferSubData(buffer,0,sizeof(values),values.data());
        for(int i=0;i<32;++i){const auto v=values[std::size_t(i)*3];out<<"plane,"<<i<<','<<count<<','<<v.x<<','<<v.y<<','<<v.z<<','<<v.w<<",0,0,0,0,0\n";}
        if(count==4){
            shader.set_int("uTestKind",1);glDispatchCompute(2,1,1);glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
            glGetNamedBufferSubData(buffer,0,sizeof(values),values.data());
            for(int i=0;i<96;++i){const auto v=values[std::size_t(i)*3],p=values[std::size_t(i)*3+1];out<<"margin,"<<i<<",4,0,0,0,"<<v.w<<','<<v.x<<','<<v.y<<','<<v.z<<','<<p.x<<','<<p.y<<'\n';}
        }
        shader.set_int("uTestKind",2);glDispatchCompute(1,1,1);glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        glGetNamedBufferSubData(buffer,0,sizeof(values),values.data());
        for(int i=0;i<32;++i){
            const auto n=values[std::size_t(i)*3],p=values[std::size_t(i)*3+1];
            out<<"masks,"<<i<<','<<count<<','<<n.x<<','<<n.y<<','<<n.z<<','<<n.w<<','<<p.x<<','<<p.y<<','<<p.z<<",0.8,0.37\n";
        }
    }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,0);glDeleteBuffers(1,&buffer);
    if(!out)throw std::runtime_error("Cannot save angular fixtures");
    constexpr int size=16;unsigned textures[3]={};glCreateTextures(GL_TEXTURE_2D,3,textures);
    std::array<glm::vec2,size*size> sums;std::array<float,size*size> depths;std::array<glm::vec4,size*size> normals;
    for(int y=0;y<size;++y)for(int x=0;x<size;++x){
        const int i=y*size+x;float v=float((x*13+y*7+x*y*3)%97)/96.0f,total=0.5f+float((x+y)%7)*0.1f;
        sums[std::size_t(i)]={v*total,total};depths[std::size_t(i)]=y==15?1.0f:x<8?0.5f:0.7f;normals[std::size_t(i)]={0,0,1,1};
    }
    glTextureStorage2D(textures[0],1,GL_RG32F,size,size);glTextureSubImage2D(textures[0],0,0,0,size,size,GL_RG,GL_FLOAT,sums.data());
    glTextureStorage2D(textures[1],1,GL_R32F,size,size);glTextureSubImage2D(textures[1],0,0,0,size,size,GL_RED,GL_FLOAT,depths.data());
    glTextureStorage2D(textures[2],1,GL_RGBA32F,size,size);glTextureSubImage2D(textures[2],0,0,0,size,size,GL_RGBA,GL_FLOAT,normals.data());
    for(unsigned t:textures){glTextureParameteri(t,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTextureParameteri(t,GL_TEXTURE_MAG_FILTER,GL_NEAREST);}
    using A=gfx::research::ColorAttachmentDesc;const A attachment[]={{GL_R32F,GL_NEAREST,GL_NEAREST}};
    gfx::research::Framebuffer target;if(!target.create(size,size,attachment,false))throw std::runtime_error("Angular filter fixture allocation failed");
    target.bind();glViewport(0,0,size,size);glDisable(GL_DEPTH_TEST);glDisable(GL_BLEND);
    resolveShader_.bind();resolveShader_.set_mat4("uInverseVp",glm::mat4(1));resolveShader_.set_float("uArrSmoothingRadius",0.75f);
    for(int i=0;i<3;++i)glBindTextureUnit(unsigned(i),textures[i]);
    std::ofstream filters(path.parent_path()/"angular_filter_gpu.csv");filters<<"mode,x,y,value\n"<<std::setprecision(10);
    for(int mode:{1,3,4,5}){
        resolveShader_.set_int("uArrSmoothing",mode);fullscreen_.draw();
        const auto data=readTexture(target.color(),size,size,GL_RED,1);
        for(int y=0;y<size;++y)for(int x=0;x<size;++x)filters<<mode<<','<<x<<','<<y<<','<<data[std::size_t(y*size+x)]<<'\n';
    }
    glDeleteTextures(3,textures);if(!filters)throw std::runtime_error("Cannot save angular filter fixtures");
}
std::uint64_t activeMapBytes(Method m, const Settings& s) {
    const auto components=m==Method::AngularRgba?4u:unsigned(activeMapLayers(m,s));
    return std::uint64_t(components) * s.mapResolution * s.mapResolution * 4u + auxiliaryMapBytes(m, s);
}
int mapGenerationPasses(Method m, const Settings& s) {
    return m==Method::AngularRgba?4:activeMapLayers(m, s);
}
const char* methodId(Method m) {
    return ids.at(std::size_t(m));
}
const char* methodLabel(Method m) {
    return labels.at(std::size_t(m));
}
Method parseMethod(const std::string& name) {
    if(name=="pcss-geometric")return Method::Geometric;
    if(name=="multiple-light-view")return Method::AngularRgba;
    if(name=="geometry-reference")return Method::Reference;
    for (int i = 0; i < methodCount; ++i)
        if (name == ids[std::size_t(i)])
            return Method(i);
    throw std::runtime_error("Unknown method: " + name);
}
const char* samplePatternId(int pattern) {
    constexpr std::array<const char*, 4> patterns = {"vogel", "random", "stratified", "blue-noise"};
    return patterns.at(std::size_t(pattern));
}
int parseSamplePattern(const std::string& name) {
    for (int i = 0; i < 4; ++i)
        if (name == samplePatternId(i))
            return i;
    throw std::runtime_error("Unknown sample pattern: " + name);
}
bool supportsSmoothing(Method method) {
    return method == Method::Pcf || method == Method::Pcss || method == Method::Geometric || method == Method::GeometricLayered;
}
int smoothingKernel(Method method, const Settings& settings) {
    if (!supportsSmoothing(method))
        return 0;
    const int kernel = settings.smoothing >= 6 ? 1 : settings.smoothing;
    return method == Method::Pcf && kernel == 2 ? 1 : kernel;
}
int effectiveSamplePattern(Method method, const Settings& settings) {
    if (method != Method::Pcss && method != Method::Geometric && method != Method::GeometricLayered)
        return 0;
    return settings.smoothing >= 6 ? settings.smoothing - 5 : settings.samplePattern;
}
const char* penumbraModeId(Method method, const Settings& settings) {
    if (method != Method::Pcss && method != Method::Geometric && method != Method::GeometricLayered)
        return "native";
    return settings.innerPenumbra ? "inner" : "full";
}
std::uint64_t samplingTextureBytes(Method method, const Settings& settings) {
    return effectiveSamplePattern(method, settings) == 3 ? 2048u * 2048u : 0u;
}
Renderer::Renderer() = default;
Renderer::~Renderer() {
    if(rgbaTexture_)glDeleteTextures(1,&rgbaTexture_);
    glDeleteTextures(4,rgbaSeparate_.data());
    if(rgbaFbo_)glDeleteFramebuffers(1,&rgbaFbo_);
    if(angularDirections_)glDeleteBuffers(1,&angularDirections_);
    if (chsWeights_)
        glDeleteBuffers(1, &chsWeights_);
    if (mapTexture_)
        glDeleteTextures(1, &mapTexture_);
    glDeleteTextures(2, basisTextures_);
    if (mapFbo_)
        glDeleteFramebuffers(1, &mapFbo_);
}
void Renderer::initialize() {
    using S = gfx::research::ShaderStage;
    loadShader(gbufferShader_, {{S::Vertex, "scene.vert"}, {S::Fragment, "gbuffer.frag"}});
    loadShader(mapRaster_, {{S::Vertex, "map.vert"},
                            {S::TessellationControl, "map.tesc"},
                            {S::TessellationEvaluation, "map.tese"},
                            {S::Fragment, "map.frag"}});
    loadShader(mapAnalytic_, {{S::Vertex, "fullscreen.vert"}, {S::Fragment, "map_analytic.frag"}});
    loadShader(resolveShader_, {{S::Vertex, "fullscreen.vert"}, {S::Fragment, "angular_resolve.frag"}});
    loadShader(displayShader_, {{S::Vertex, "fullscreen.vert"}, {S::Fragment, "display.frag"}});
    cube_ = gfx::research::make_cube(1);
    plane_ = gfx::research::make_plane(30, 1);
    glCreateFramebuffers(1, &mapFbo_);
}
void Renderer::prepareResultBlend() {
    auto& shader=lastMethod_==Method::AngularRgba?rgbaBlendShader_:blendShader_;
    if(!shader.valid())
        loadShader(shader,{{gfx::research::ShaderStage::Vertex,"fullscreen.vert"},{gfx::research::ShaderStage::Fragment,lastMethod_==Method::AngularRgba?"rgba_blend.frag":"result_blend.frag"}});
    if(blendWidth_!=settings_.width||blendHeight_!=settings_.height){
        const gfx::research::ColorAttachmentDesc attachment[]={{GL_R32F,GL_NEAREST,GL_NEAREST}};
        if(!blended_.create(settings_.width,settings_.height,attachment,false))
            throw std::runtime_error("Result blend allocation failed");
        blendWidth_=settings_.width;blendHeight_=settings_.height;
    }
}
void Renderer::blendResult() {
    blended_.bind();glViewport(0,0,settings_.width,settings_.height);
    glDisable(GL_DEPTH_TEST);glDisable(GL_BLEND);
    auto& shader=lastMethod_==Method::AngularRgba?rgbaBlendShader_:blendShader_;
    shader.bind();shader.set_mat4("uInverseVp",inverseVp_);
    shader.set_float("uBlendStrength",settings_.resultBlend);
    shader.set_float("uBlendSigma",settings_.resultBlendSigma);
    shader.set_int("uBlendRadius",settings_.resultBlendRadius);
    shader.set_int("uContactEnabled",settings_.contactGuard>0?1:0);
    shader.set_int("uInnerPenumbra",settings_.innerPenumbra?1:0);
    glBindTextureUnit(0,result_.color());glBindTextureUnit(1,gbuffer_.depth());
    glBindTextureUnit(2,gbuffer_.color(1));glBindTextureUnit(3,result_.color(1));
    fullscreen_.draw();resultBlended_=true;
}
void Renderer::allocateChsWeights() {
    if (!chsWeights_) {
        glCreateBuffers(1, &chsWeights_);
        glNamedBufferStorage(chsWeights_, sizeof(chsKernelData), chsKernelData.data(), 0);
    }
    glBindBufferBase(GL_UNIFORM_BUFFER, 6, chsWeights_);
}
void Renderer::prepareBlueNoise() {
    if (blueNoise_.id()) return;
    gfx::research::Texture2D source;
    const auto path=std::filesystem::path(GFX_SHADOWS_SOURCE_DIR)/"assets"/"blue_noise.png";
    if (!source.load(path,false,false)||source.width()!=2048||source.height()!=2048)
        throw std::runtime_error("Expected the supplied 2048x2048 blue-noise PNG: "+path.string());
    // Reuse the base image loader; retain only the scalar R channel on the GPU.
    std::vector<unsigned char> pixels(2048u*2048u);
    glGetTextureImage(source.id(),0,GL_RED,GL_UNSIGNED_BYTE,GLsizei(pixels.size()),pixels.data());
    blueNoise_.create(2048,2048,GL_R8);
    blueNoise_.upload(0,GL_RED,GL_UNSIGNED_BYTE,pixels.data());
    blueNoise_.set_filters(GL_NEAREST,GL_NEAREST);
    blueNoise_.set_wrap(GL_REPEAT,GL_REPEAT);
}
gfx::research::Shader& Renderer::shadowProgram(Method method) {
    if(method==Method::AngularRgba)return rgbaShaders_[std::size_t(settings_.multiViewStorage)];
    const int kernel = smoothingKernel(method, settings_);
    const int slot = method == Method::Pcf ? 0 : method == Method::Pcss ? 1 : method == Method::Geometric ? 2 : 3;
    const bool packed = usesPackedDepth(method, settings_);
    const bool contact=usesPcss(method)&&settings_.contactGuard>0;
    auto& original = packed ? intervalShaders_.at(std::size_t(method==Method::Pcss?0:method==Method::Geometric?1:2)).at(std::size_t(kernel))
                    : method == Method::Chs       ? chsShaders_.at(std::size_t((settings_.chsKernel - 7) / 2))
                    : supportsSmoothing(method) ? smoothingShaders_.at(std::size_t(slot)).at(std::size_t(kernel))
                                                : shadowShaders_.at(std::size_t(method));
    auto& program=contact?contactShaders_.at(packed?1:0).at(std::size_t(method==Method::Pcss?0:method==Method::Geometric?1:2)).at(std::size_t(kernel)):original;
    if (!program.valid()) {
        std::string defines = "#define SHADOW_METHOD " + std::to_string(int(method)) + "\n";
        defines += "#define CHS_FILTER_SIZE " + std::to_string(settings_.chsKernel) + "\n";
        defines += "#define SHADOW_SMOOTHING " + std::to_string(kernel) + "\n";
        if (packed) defines += "#define PACKED_INTERVALS 1\n";
        if (contact) defines += "#define CONTACT_GUARD_ENABLED 1\n";
        using S = gfx::research::ShaderStage;
        loadShader(program, {{S::Vertex, "fullscreen.vert"}, {S::Fragment, "shadow.frag"}}, defines);
    }
    return program;
}
gfx::research::Shader& Renderer::angularProgram(Method method,bool accumulate){
    auto& shader=method==Method::Reference?angularShaders_.at(std::size_t(accumulate)):
        triangulationShaders_.at(std::size_t(settings_.triSamples-3)).at(std::size_t(accumulate));
    if(!shader.valid()){
        std::string defines="#define ARR_ACCUMULATE "+std::to_string(int(accumulate))+"\n";
        defines+="#define ARR_TRIANGULATION "+std::to_string(int(method==Method::ArrTriangulation))+"\n";
        defines+="#define TRI_COUNT "+std::to_string(settings_.triSamples)+"\n";
        using S=gfx::research::ShaderStage;
        loadShader(shader,{{S::Vertex,"fullscreen.vert"},{S::Fragment,"angular.frag"}},defines);
    }
    return shader;
}
void Renderer::prepareAngularDirections(){
    const int count=std::max(settings_.referenceSamples,angularSamples(Method::Reference,settings_));
    if(!angularDirections_){
        glCreateBuffers(1,&angularDirections_);
        glNamedBufferStorage(angularDirections_,16384*sizeof(glm::vec4),nullptr,GL_DYNAMIC_STORAGE_BIT);
    }
    if(angularDirectionAngle_!=settings_.angle||uploadedAngularDirections_<count){
        std::vector<glm::vec4> directions(std::size_t(count),glm::vec4(0));
        for(int i=0;i<count;++i){const auto a=sourceSample(unsigned(i),settings_.angle);directions[std::size_t(i)]=glm::vec4(glm::vec3(a.direction),float(a.weight));}
        glNamedBufferSubData(angularDirections_,0,GLsizeiptr(directions.size()*sizeof(glm::vec4)),directions.data());
        angularDirectionAngle_=settings_.angle;uploadedAngularDirections_=count;
    }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,7,angularDirections_);
}
void Renderer::allocate() {
    const auto& s = settings_;
    if (s.width != allocatedWidth_ || s.height != allocatedHeight_) {
        using A = gfx::research::ColorAttachmentDesc;
        const A g[] = {{GL_RGBA8, GL_NEAREST, GL_NEAREST}, {GL_RGBA32F, GL_NEAREST, GL_NEAREST}};
        const A r[] = {{GL_R32F, GL_NEAREST, GL_NEAREST}, {GL_RGBA16F, GL_NEAREST, GL_NEAREST}};
        const A a[] = {{GL_RG32F, GL_NEAREST, GL_NEAREST}};
        const A p[] = {{GL_RGBA8, GL_NEAREST, GL_NEAREST}};
        if (!gbuffer_.create(s.width, s.height, g, true) || !result_.create(s.width, s.height, r, false) ||
            !accum_.create(s.width, s.height, a, false) || !preview_.create(s.width, s.height, p, false))
            throw std::runtime_error("Framebuffer allocation failed");
        allocatedWidth_ = s.width;
        allocatedHeight_ = s.height;
    }
    allocateMap(mapPacked_);
}
void Renderer::allocateMap(bool packed) {
    const auto& s = settings_;
    if (s.mapResolution != allocatedMap_ || packed != mapPacked_) {
        glNamedFramebufferTexture(mapFbo_, GL_DEPTH_ATTACHMENT, 0, 0);
        if (mapTexture_)
            glDeleteTextures(1, &mapTexture_);
        glCreateTextures(GL_TEXTURE_2D_ARRAY, 1, &mapTexture_);
        glTextureStorage3D(mapTexture_, 1, packed ? GL_R32UI : GL_DEPTH_COMPONENT32F, s.mapResolution, s.mapResolution, 1);
        glTextureParameteri(mapTexture_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(mapTexture_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTextureParameteri(mapTexture_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTextureParameteri(mapTexture_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        const float white[] = {1, 1, 1, 1};
        if (packed) {
            const unsigned empty[] = {0xffff0000u,0,0,0};
            glTextureParameterIuiv(mapTexture_, GL_TEXTURE_BORDER_COLOR, empty);
        } else glTextureParameterfv(mapTexture_, GL_TEXTURE_BORDER_COLOR, white);
        glNamedFramebufferDrawBuffer(mapFbo_, GL_NONE);
        glNamedFramebufferReadBuffer(mapFbo_, GL_NONE);
        allocatedMap_ = s.mapResolution;
        mapPacked_ = packed;
    }
}
void Renderer::validateFilters(const std::filesystem::path& path) {
    // Independent Python oracle uses the original AMD PCF backend, not this gather4 implementation.
    constexpr int size = 32, count = 320;
    std::array<float, size * size> depth{};
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x)
            depth[std::size_t(y * size + x)] = float(((x * 13 + y * 7 + x * y * 3) % 89) + 5) / 100.0f;
    unsigned texture = 0, buffer = 0;
    glCreateTextures(GL_TEXTURE_2D_ARRAY, 1, &texture);
    glTextureStorage3D(texture, 1, GL_DEPTH_COMPONENT32F, size, size, 1);
    glTextureSubImage3D(texture, 0, 0, 0, 0, size, size, 1, GL_DEPTH_COMPONENT, GL_FLOAT, depth.data());
    glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTextureParameteri(texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    const float border[] = {1, 1, 1, 1};
    glTextureParameterfv(texture, GL_TEXTURE_BORDER_COLOR, border);
    glBindTextureUnit(2, texture);
    glCreateBuffers(1, &buffer);
    std::array<float, count * 8> values{};
    glNamedBufferStorage(buffer, sizeof(values), nullptr, GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);
    allocateChsWeights();
    for (int mode = 0; mode < 4; ++mode) {
        gfx::research::Shader shader;
        loadShader(shader, {{gfx::research::ShaderStage::Compute, "filter_validation.comp"}},
                   "#define SHADOW_SMOOTHING " + std::to_string(mode + 2) + "\n");
        shader.bind();
        shader.set_float("uWarp", 8.0f);
        shader.set_vec2("uFocus", {0, 0});
        shader.set_int("uMapResolution", size);
        shader.set_float("uDepthBias", 0.00015f);
        shader.set_int("uResultOffset", mode * 80);
        glDispatchCompute(80, 1, 1);
    }
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glGetNamedBufferSubData(buffer, 0, sizeof(values), values.data());
    std::ofstream out(path);
    out << "case_id,dpcf,chs,smooth_visibility,smooth_gap,reconstruction,smooth_mean_gap\n" << std::setprecision(10);
    for (int i = 0; i < count; ++i) {
        out << i;
        for (int c = 0; c < 4; ++c) {
            float v = values[std::size_t(i * 4 + c)];
            if (!std::isfinite(v))
                throw std::runtime_error("Non-finite filter validation");
            out << ',' << v;
        }
        const float mean = values[std::size_t((i + count) * 4)];
        if (!std::isfinite(mean))
            throw std::runtime_error("Non-finite mean blocker validation");
        out << ',' << (i / 80 + 2) << ',' << mean << '\n';
    }
    if (!out)
        throw std::runtime_error("Cannot save filter validation");
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
    glDeleteBuffers(1, &buffer);
    glDeleteTextures(1, &texture);
}
void Renderer::validateSampling(const std::filesystem::path& path) {
    constexpr int count = 1568;
    gfx::research::Shader shader;
    loadShader(shader, {{gfx::research::ShaderStage::Compute, "sampling_validation.comp"}});
    std::array<float, count * 4> values{};
    unsigned buffer = 0;
    glCreateBuffers(1, &buffer);
    glNamedBufferStorage(buffer, sizeof(values), nullptr, GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);
    shader.bind();
    glDispatchCompute(count, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glGetNamedBufferSubData(buffer, 0, sizeof(values), values.data());
    std::ofstream out(path);
    out << "case_id,blocker_x,blocker_y,filter_x,filter_y\n" << std::setprecision(10);
    for (int i = 0; i < count; ++i) {
        out << i;
        for (int c = 0; c < 4; ++c) {
            float v = values[std::size_t(i * 4 + c)];
            if (!std::isfinite(v))
                throw std::runtime_error("Non-finite sampling validation");
            out << ',' << v;
        }
        out << '\n';
    }
    if (!out)
        throw std::runtime_error("Cannot save sampling validation");
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
    glDeleteBuffers(1, &buffer);
}
void Renderer::validateBlueNoise(const std::filesystem::path& path) {
    constexpr int count=128;
    gfx::research::Shader shader;
    loadShader(shader,{{gfx::research::ShaderStage::Compute,"blue_noise_validation.comp"}});
    std::array<unsigned char,64> pixels{};
    for(int y=0;y<8;++y)for(int x=0;x<8;++x)pixels[std::size_t(y*8+x)]=static_cast<unsigned char>((x*17+y*29+13)%256);
    gfx::research::Texture2D fixture;
    fixture.create(8,8,GL_R8);fixture.upload(0,GL_RED,GL_UNSIGNED_BYTE,pixels.data());fixture.bind(5);
    std::array<float,count*8> values{};
    unsigned buffer=0;glCreateBuffers(1,&buffer);
    glNamedBufferStorage(buffer,sizeof(values),nullptr,GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,buffer);shader.bind();
    const float tiles[]={0.5f,1.0f,4.0f,16.0f};
    for(int group=0;group<4;++group){
        shader.set_int("uCaseBase",group*32);shader.set_float("uBlueNoiseTile",tiles[group]);
        glDispatchCompute(32,1,1);
    }
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glGetNamedBufferSubData(buffer,0,sizeof(values),values.data());
    std::ofstream out(path);
    out<<"case_id,blocker_angle,filter_angle,periodic_blocker_angle,periodic_filter_angle,blocker_x,blocker_y,filter_x,filter_y\n"<<std::setprecision(10);
    for(int i=0;i<count;++i){
        out<<i;
        for(int component=0;component<8;++component){
            float value=values[std::size_t(i*8+component)];
            if(!std::isfinite(value))throw std::runtime_error("Non-finite blue-noise validation");
            out<<','<<value;
        }
        out<<'\n';
    }
    if(!out)throw std::runtime_error("Cannot save blue-noise validation");
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,0);glDeleteBuffers(1,&buffer);
}
void Renderer::validateRgbaBlend(const std::filesystem::path& path) {
    constexpr int size=16;
    unsigned textures[4]={};glCreateTextures(GL_TEXTURE_2D,4,textures);
    for(int i=0;i<4;++i){
        glTextureStorage2D(textures[i],1,i<2?GL_R32F:GL_RGBA32F,size,size);
        glTextureParameteri(textures[i],GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        glTextureParameteri(textures[i],GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        glBindTextureUnit(unsigned(i),textures[i]);
    }
    const gfx::research::ColorAttachmentDesc attachment[]={{GL_R32F,GL_NEAREST,GL_NEAREST}};
    gfx::research::Framebuffer target;
    if(!target.create(size,size,attachment,false))throw std::runtime_error("Angular blend fixture allocation failed");
    gfx::research::Shader shader;loadShader(shader,{{gfx::research::ShaderStage::Vertex,"fullscreen.vert"},{gfx::research::ShaderStage::Fragment,"rgba_blend.frag"}});
    shader.bind();shader.set_mat4("uInverseVp",glm::mat4(1));shader.set_float("uBlendSigma",0.35f);
    target.bind();glViewport(0,0,size,size);glDisable(GL_DEPTH_TEST);glDisable(GL_BLEND);
    std::ofstream out(path);out<<"scene,mode,x,y,input,ideal,gpu\n"<<std::setprecision(10);
    for(int scene=0;scene<8;++scene){
        std::array<float,size*size> values{},ideal{},depths{};
        std::array<glm::vec4,size*size> normals{},guides{};
        for(int y=0;y<size;++y)for(int x=0;x<size;++x){
            const auto i=std::size_t(y*size+x);
            const float ramp=0.2f+0.6f*float(x)/float(size-1);
            ideal[i]=scene==0?0.375f:scene<3?ramp:scene==6?(x==6?0.0f:1.0f):x<8?0.0f:1.0f;
            values[i]=ideal[i];depths[i]=0.5f;normals[i]={0,0,1,1};guides[i]={6,0,15,1};
            if(scene==2)values[i]+=float((x*13+y*7+x*y*3)%11-5)*0.015f;
            if(scene==3&&x>=8)depths[i]=0.7f;
            if(scene==4&&x>=8)normals[i]={0,1,0,1};
            if(scene==5&&x>=8)guides[i].z=8;
            if(scene==6&&x==6)guides[i].y=1;
            if(scene==7)guides[i].x=0.25f;
        }
        glTextureSubImage2D(textures[0],0,0,0,size,size,GL_RED,GL_FLOAT,values.data());
        glTextureSubImage2D(textures[1],0,0,0,size,size,GL_RED,GL_FLOAT,depths.data());
        glTextureSubImage2D(textures[2],0,0,0,size,size,GL_RGBA,GL_FLOAT,normals.data());
        glTextureSubImage2D(textures[3],0,0,0,size,size,GL_RGBA,GL_FLOAT,guides.data());
        for(int mode=0;mode<3;++mode){
            shader.set_float("uBlendStrength",mode==0?0.0f:0.85f);shader.set_int("uBlendRadius",mode==1?1:2);
            fullscreen_.draw();const auto gpu=readTexture(target.color(),size,size,GL_RED,1);
            for(int y=0;y<size;++y)for(int x=0;x<size;++x){
                const auto i=std::size_t(y*size+x);
                if(!std::isfinite(gpu[i]))throw std::runtime_error("Non-finite angular blend fixture");
                out<<scene<<','<<mode<<','<<x<<','<<y<<','<<values[i]<<','<<ideal[i]<<','<<gpu[i]<<'\n';
            }
        }
    }
    glDeleteTextures(4,textures);
    if(!out)throw std::runtime_error("Cannot save angular blend validation");
}
void Renderer::validateContactBlend(const std::filesystem::path& path) {
    constexpr int size=16,count=64;
    unsigned maps[2]={};glCreateTextures(GL_TEXTURE_2D_ARRAY,2,maps);
    for(unsigned map:maps){glTextureParameteri(map,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTextureParameteri(map,GL_TEXTURE_MAG_FILTER,GL_NEAREST);}
    std::array<float,size*size> front{};std::array<unsigned,size*size> packed{};
    for(int y=0;y<size;++y)for(int x=0;x<size;++x){
        float a=x<4?0.495f:0.4f,b=x<8?0.495f:0.41f;
        if(y==0)a=b=0.51f;
        const auto i=std::size_t(y*size+x);front[i]=y==15?1.0f:a;
        packed[i]=y==15?0xffff0000u:(unsigned(std::lround(a*65535.0f))<<16u)|unsigned(std::lround(b*65535.0f));
    }
    glTextureStorage3D(maps[0],1,GL_R32F,size,size,1);
    glTextureSubImage3D(maps[0],0,0,0,0,size,size,1,GL_RED,GL_FLOAT,front.data());
    glTextureStorage3D(maps[1],1,GL_R32UI,size,size,1);
    glTextureSubImage3D(maps[1],0,0,0,0,size,size,1,GL_RED_INTEGER,GL_UNSIGNED_INT,packed.data());
    unsigned buffer=0;glCreateBuffers(1,&buffer);glNamedBufferStorage(buffer,count*sizeof(glm::vec4),nullptr,GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,buffer);
    std::ofstream out(path);out<<"kind,mode,x,y,value,radius,coverage,gap\n"<<std::setprecision(10);
    for(int mode=0;mode<2;++mode){
        gfx::research::Shader shader;
        loadShader(shader,{{gfx::research::ShaderStage::Compute,"contact_validation.comp"}},mode?"#define PACKED_INTERVALS 1\n":"");
        shader.bind();shader.set_int("uMapResolution",size);shader.set_float("uDepthBias",0.00015f);
        shader.set_float("uDepthRange",80);shader.set_float("uExtent",0.2f);shader.set_float("uTanAngle",0.026f);
        shader.set_float("uMaxRadiusWorld",3);shader.set_float("uContactGuard",1);shader.set_float("uContactMaxGap",1);
        shader.set_float("uWarp",0);shader.set_vec2("uFocus",{0,0});glBindTextureUnit(2,maps[mode]);
        glDispatchCompute(count,1,1);glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        std::array<glm::vec4,count> data{};glGetNamedBufferSubData(buffer,0,sizeof(data),data.data());
        for(int i=0;i<count;++i){const auto v=data[std::size_t(i)];out<<"contact,"<<mode<<','<<i<<",0,"<<v.x<<','<<v.y<<','<<v.z<<','<<v.w<<'\n';}
    }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,0);glDeleteBuffers(1,&buffer);glDeleteTextures(2,maps);
    unsigned textures[4]={};glCreateTextures(GL_TEXTURE_2D,4,textures);
    std::array<float,size*size> visibility{},depths{};
    std::array<glm::vec4,size*size> normals{},guides{};
    for(int y=0;y<size;++y)for(int x=0;x<size;++x){
        auto i=std::size_t(y*size+x);
        visibility[i]=x==0?1.0f:x==3?0.0f:float((x*13+y*7+x*y*3)%97)/96.0f;
        depths[i]=y==15?1.0f:x<8?0.5f:0.7f;
        normals[i]=x==12?glm::vec4(0,1,0,1):glm::vec4(0,0,1,1);
        guides[i]={x<8?0.01f:0.1f,0,x==3?1.0f:0.0f,1};
    }
    for(int i=0;i<2;++i)glTextureStorage2D(textures[i],1,GL_R32F,size,size);
    for(int i=2;i<4;++i)glTextureStorage2D(textures[i],1,GL_RGBA32F,size,size);
    glTextureSubImage2D(textures[0],0,0,0,size,size,GL_RED,GL_FLOAT,visibility.data());
    glTextureSubImage2D(textures[1],0,0,0,size,size,GL_RED,GL_FLOAT,depths.data());
    glTextureSubImage2D(textures[2],0,0,0,size,size,GL_RGBA,GL_FLOAT,normals.data());
    glTextureSubImage2D(textures[3],0,0,0,size,size,GL_RGBA,GL_FLOAT,guides.data());
    const gfx::research::ColorAttachmentDesc attachment[]={{GL_R32F,GL_NEAREST,GL_NEAREST}};
    gfx::research::Framebuffer target;
    if(!target.create(size,size,attachment,false))throw std::runtime_error("Blend fixture allocation failed");
    gfx::research::Shader shader;loadShader(shader,{{gfx::research::ShaderStage::Vertex,"fullscreen.vert"},{gfx::research::ShaderStage::Fragment,"result_blend.frag"}});
    target.bind();glViewport(0,0,size,size);glDisable(GL_DEPTH_TEST);glDisable(GL_BLEND);
    shader.bind();shader.set_mat4("uInverseVp",glm::mat4(1));
    for(int i=0;i<4;++i){glTextureParameteri(textures[i],GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTextureParameteri(textures[i],GL_TEXTURE_MAG_FILTER,GL_NEAREST);glBindTextureUnit(unsigned(i),textures[i]);}
    for(int mode=0;mode<4;++mode){
        shader.set_float("uBlendStrength",mode==0?0.0f:mode==2?1.0f:0.8f);
        shader.set_float("uBlendSigma",mode==3?0.1f:0.2f);shader.set_int("uBlendRadius",mode<2?1:2);
        shader.set_int("uContactEnabled",mode>=2?1:0);shader.set_int("uInnerPenumbra",mode==3?1:0);
        fullscreen_.draw();const auto data=readTexture(target.color(),size,size,GL_RED,1);
        for(int y=0;y<size;++y)for(int x=0;x<size;++x)out<<"blend,"<<mode<<','<<x<<','<<y<<','<<data[std::size_t(y*size+x)]<<",0,0,0\n";
    }
    glDeleteTextures(4,textures);
    if(!out||glGetError()!=GL_NO_ERROR)throw std::runtime_error("Contact/blend fixture failure");
}
void Renderer::validateStress(const std::filesystem::path& path) {
    gfx::research::Shader shader;
    loadShader(shader, {{gfx::research::ShaderStage::Compute, "stress_validation.comp"}});
    unsigned buffer = 0;
    glCreateBuffers(1, &buffer);
    glNamedBufferStorage(buffer, 8 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);
    shader.bind();
    shader.set_int("uStressMode", 1);
    shader.set_vec3("uDirection", {0, -1, 0});
    std::ofstream out(path);
    out << "objects,instance,mode,expected_blocked,gpu_blocked,bounds_max_abs_error\n" << std::setprecision(10);
    for (int count : {33, 65, 1024, 1'000'000}) {
        const int index = count - 1;
        const auto target = stressBox(index, count);
        shader.set_int("uBoxCount", count);shader.set_int("uStressGrid", stressGrid(count));
        shader.set_int("uTestIndex", index);
        for (int mode = 0; mode < 3; ++mode) {
            const glm::vec3 origin = mode == 2 ? glm::vec3{20, 8, 20} :
                glm::vec3{(target.lo.x + target.hi.x) * 0.5f, 8, (target.lo.z + target.hi.z) * 0.5f};
            const int dropped = mode == 1 ? target.group : -1;
            shader.set_vec3("uOrigin", origin);shader.set_int("uDroppedGroup", dropped);
            glDispatchCompute(1, 1, 1);
            glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
            std::array<float, 8> result{};
            glGetNamedBufferSubData(buffer, 0, sizeof(result), result.data());
            bool expected = false;
            for (int i = 0; i < count; ++i) {
                const auto box = stressBox(i, count);
                if (box.group == dropped) continue;
                double hit = 0;
                if (rayBox(glm::dvec3(origin), {0, -1, 0}, box, hit)) { expected = true; break; }
            }
            double error = std::abs(double(result[7]) - target.group);
            for (int i = 0; i < 3; ++i) {
                error = std::max(error, std::abs(double(result[std::size_t(1+i)]) - target.lo[i]));
                error = std::max(error, std::abs(double(result[std::size_t(4+i)]) - target.hi[i]));
            }
            out << count << ',' << index << ',' << mode << ',' << int(expected) << ',' << result[0] << ',' << error << '\n';
            if (!std::isfinite(error) || error > 2e-6 || result[0] != float(expected))
                throw std::runtime_error("Procedural ARR disagrees with CPU bounds/ray oracle beyond 32 objects");
        }
    }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);glDeleteBuffers(1, &buffer);
    if (!out) throw std::runtime_error("Cannot save procedural ARR verification");
}
double Renderer::prepare(const Scene& scene, const Settings& settings, bool timed) {
    if (scene.boxes.size() > maxCasters)
        throw std::runtime_error("Too many casters");
    scene_ = scene;
    settings_ = settings;
    if (scene_.isStress() && scene_.casterCount() > maxCasters && settings_.analyticMaps)
        throw std::runtime_error("Large stress scenes require raster maps; the analytic AABB backend supports 32 boxes");
    allocate();
    gfx::research::OrbitCamera camera;
    camera.set_target(settings_.customCamera ? settings_.cameraTarget : scene_.target);
    camera.set_clip(0.1f, 100);
    camera.set_distance(settings_.customCamera ? settings_.cameraDistance : settings_.camera == 0 ? 16.0f : 14.0f);
    camera.set_yaw(settings_.customCamera ? settings_.cameraYaw : -2.1f);
    camera.set_pitch(settings_.customCamera ? settings_.cameraPitch : settings_.camera == 0 ? 0.65f : 1.25f);
    camera.set_fov(52);
    const auto vp = camera.view_projection(float(settings_.width) / float(settings_.height));
    inverseVp_ = glm::inverse(vp);
    const glm::vec3 center{0, 1.5f, 0}, sun = glm::vec3(sunDirection());
    lightVp_ = glm::ortho(-14.0f, 14.0f, -14.0f, 14.0f, 0.1f, 80.1f) *
               glm::lookAt(center + sun * 30.0f, center, glm::vec3(0, 0, 1));
    inverseLightVp_ = glm::inverse(lightVp_);
    unsigned queries[2] = {};
    if (timed) {
        glCreateQueries(GL_TIMESTAMP, 2, queries);
        glQueryCounter(queries[0], GL_TIMESTAMP);
    }
    gbuffer_.bind();
    glViewport(0, 0, settings_.width, settings_.height);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    const float zero[] = {0, 0, 0, 0}, one = 1;
    glClearNamedFramebufferfv(gbuffer_.id(), GL_COLOR, 0, zero);
    glClearNamedFramebufferfv(gbuffer_.id(), GL_COLOR, 1, zero);
    glClearNamedFramebufferfv(gbuffer_.id(), GL_DEPTH, 0, &one);
    gbufferShader_.bind();
    gbufferShader_.set_mat4("uVp", vp);
    gbufferShader_.set_mat4("uModel", glm::rotate(glm::mat4(1), scene_.receiverTilt, glm::vec3(0, 0, 1)));
    gbufferShader_.set_vec3("uColor", {0.62f, 0.65f, 0.59f});
    gbufferShader_.set_int("uReceiver", 1);
    gbufferShader_.set_int("uStressMode", 0);
    plane_.draw();
    gbufferShader_.set_int("uReceiver", 0);
    if (scene_.isStress()) {
        gbufferShader_.set_int("uStressMode", 1);
        gbufferShader_.set_int("uStressGrid", stressGrid(scene_.stressObjects));
        if (scene_.stressObjects > 0) cube_.draw_instanced(scene_.stressObjects);
    } else for (const auto& b : scene_.boxes) {
        gbufferShader_.set_mat4("uModel", boxModel(b));
        gbufferShader_.set_vec3("uColor", b.color);
        cube_.draw();
    }
    glDisable(GL_DEPTH_TEST);
    if (timed) {
        glQueryCounter(queries[1], GL_TIMESTAMP);
        GLuint64 times[2] = {};
        for (int i = 0; i < 2; ++i) glGetQueryObjectui64v(queries[i], GL_QUERY_RESULT, &times[i]);
        glDeleteQueries(2, queries);
        return double(times[1] - times[0]) * 1e-6;
    }
    return 0;
}
void Renderer::bindScene(gfx::research::Shader& shader, int dropped) {
    shader.set_int("uBoxCount", scene_.casterCount());
    shader.set_int("uDroppedGroup", dropped);
    shader.set_int("uStressMode", scene_.isStress() ? 1 : 0);
    shader.set_int("uStressGrid", scene_.isStress() ? stressGrid(scene_.stressObjects) : 1);
    if (scene_.isStress()) return;
    for (std::size_t i = 0; i < scene_.boxes.size(); ++i) {
        const auto& b = scene_.boxes[i];
        shader.set_vec4(("uBoxLo[" + std::to_string(i) + "]").c_str(), glm::vec4(b.lo, float(b.group)));
        shader.set_vec4(("uBoxHi[" + std::to_string(i) + "]").c_str(), glm::vec4(b.hi, 0));
    }
}
void Renderer::bindScreen(gfx::research::Shader& shader) {
    glBindTextureUnit(0, gbuffer_.depth());
    glBindTextureUnit(1, gbuffer_.color(1));
    shader.set_mat4("uInverseVp", inverseVp_);
    shader.set_float("uNormalBias", settings_.normalBias);
}
void Renderer::configureShadow(gfx::research::Shader& shader, Method) {
    shader.set_mat4("uLightVp", lightVp_);
    shader.set_mat4("uInverseLightVp", inverseLightVp_);
    shader.set_float("uWarp", settings_.warp);
    shader.set_vec2("uFocus", {0, 0});
    shader.set_int("uMapResolution", settings_.mapResolution);
}
void Renderer::validateRgba(const std::filesystem::path& path){
    Settings settings;settings.width=64;settings.height=48;settings.mapResolution=32;
    settings.warp=0;settings.analyticMaps=true;settings.angle=5;settings.rgbaRotation=37;
    const auto scene=makeScene("crossing-four");
    std::ofstream out(path);out<<"offset,cascades,channel,x,y,gpu,cpu\n"<<std::setprecision(10);
    for(int layout=0;layout<4;++layout){
        const int offset=layout%2;
        settings.rgbaCascades=layout>=2;settings.rgbaExtent=layout>=2?3.0f:14.0f;
        settings.rgbaOffset=offset!=0;prepare(scene,settings);render(Method::AngularRgba);
        const auto depths=readMultiViewDepth();
        for(int channel=0;channel<4;++channel){
            const glm::dmat4 vp=rgbaVp_[std::size_t(channel)],inverse=glm::inverse(vp);
            for(int y=0;y<32;++y)for(int x=0;x<32;++x){
                const double px=(x+0.5)/16.0-1,py=(y+0.5)/16.0-1;
                auto a=inverse*glm::dvec4(px,py,-1,1),b=inverse*glm::dvec4(px,py,1,1);
                const auto origin=glm::dvec3(a)/a.w,direction=glm::normalize(glm::dvec3(b)/b.w-origin);
                double nearest=1e29;
                for(const auto& box:scene.boxes){double t;if(rayBox(origin,direction,box,t))nearest=std::min(nearest,t);}
                double expected=1;
                if(nearest<1e28){auto clip=vp*glm::dvec4(origin+direction*nearest,1);expected=clip.z/clip.w*0.5+0.5;}
                const float actual=depths[std::size_t((y*32+x)*4+channel)];
                if(!std::isfinite(actual)||std::abs(actual-expected)>2e-5)throw std::runtime_error("RGBA map differs from CPU ray/box depth");
                out<<offset<<','<<(settings.rgbaCascades?1:0)<<','<<channel<<','<<x<<','<<y<<','<<actual<<','<<expected<<'\n';
                if(!offset&&!settings.rgbaCascades&&actual!=depths[std::size_t((y*32+x)*4)])throw std::runtime_error("RGBA zero-offset channels differ");
            }
        }
        for(int smoothing=0;layout<2&&smoothing<4;++smoothing){
            settings.rgbaSmoothing=smoothing;prepare(scene,settings);render(Method::AngularRgba);
            const auto image=read();const auto channels=diagnostics();
            for(std::size_t i=0;i<image.visibility.size();++i){
                double expected=0;for(int k=0;k<4;++k)expected+=channels[i*4+std::size_t(k)]*0.25;
                if(!std::isfinite(image.visibility[i])||std::abs(image.visibility[i]-expected)>0.0006)
                    throw std::runtime_error("RGBA visibility does not match channel mixture");
            }
        }
    }
    unsigned buffer=0;glCreateBuffers(1,&buffer);glNamedBufferStorage(buffer,1312*sizeof(float),nullptr,GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,6,buffer);
    gfx::research::Shader shader;loadShader(shader,{{gfx::research::ShaderStage::Compute,"rgba_mix_validation.comp"}});
    shader.bind();
    std::ofstream mixes(path.parent_path()/"rgba_mix_gpu.csv");mixes<<"case_id,geometric_blend,gpu\n"<<std::setprecision(10);
    int blendSlot=0;
    for(float blend:{0.0f,0.5f,1.0f}){
        shader.set_float("uGeometricBlend",blend);glDispatchCompute(21,1,1);glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        std::array<float,1312> values{};glGetNamedBufferSubData(buffer,0,sizeof(values),values.data());
        for(int i=0;i<1312;++i)mixes<<blendSlot*1312+i<<','<<blend<<','<<values[std::size_t(i)]<<'\n';
        ++blendSlot;
    }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,6,0);glDeleteBuffers(1,&buffer);
    if(!mixes)throw std::runtime_error("Cannot save RGBA mixture validation");
    unsigned edgeMap=0,edgeBuffer=0;
    glCreateTextures(GL_TEXTURE_2D,1,&edgeMap);glTextureStorage2D(edgeMap,1,GL_RGBA32F,128,128);
    glTextureParameteri(edgeMap,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTextureParameteri(edgeMap,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glCreateBuffers(1,&edgeBuffer);glNamedBufferStorage(edgeBuffer,128*sizeof(glm::vec4),nullptr,GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,6,edgeBuffer);
    gfx::research::Shader edge;loadShader(edge,{{gfx::research::ShaderStage::Compute,"rgba_edge_validation.comp"}});
    edge.bind();edge.set_int("uMapResolution",128);edge.set_int("uEdgeSearch",64);
    edge.set_vec4("uBiasScale",glm::vec4(1));edge.set_vec4("uExtents",glm::vec4(0.2f));edge.set_vec4("uDepthRanges",glm::vec4(80));
    edge.set_float("uDepthBias",0);edge.set_float("uTanAngle",std::tan(glm::radians(1.5f)));
    edge.set_float("uTriRadius",1);edge.set_float("uTriRotation",0.37f);edge.set_vec3("uSun",{0,1,0});
    edge.set_float("uWarp",0);edge.set_vec2("uFocus",{0,0});
    std::ofstream edges(path.parent_path()/"rgba_edge_gpu.csv");edges<<"fixture,x,raw,reconstructed\n"<<std::setprecision(10);
    for(int fixture=0;fixture<3;++fixture){
        std::vector<glm::vec4> texels(128*128,glm::vec4(1));
        for(int y=0;y<128;++y)for(int x=0;x<128;++x)for(int channel=0;channel<4;++channel){
            const float px=((float(x)+0.5f)/64.0f-1.0f)*0.2f;
            const float source=channel==0?0.0f:std::cos(0.37f+2.0f*float(pi)*float(channel-1)/3.0f);
            const float farX=px+source*4.0f*std::tan(glm::radians(1.5f));
            const float nearX=px+source*0.1f*std::tan(glm::radians(1.5f));
            float d=1;
            if((fixture==0&&farX<0)||(fixture==2&&farX>-0.15f&&farX<0))d=0.45f;
            if(fixture!=0&&nearX>-0.03f&&nearX<0.03f)d=std::min(d,0.5f-0.1f/80.0f);
            texels[std::size_t(y*128+x)][channel]=d;
        }
        glTextureSubImage2D(edgeMap,0,0,0,128,128,GL_RGBA,GL_FLOAT,texels.data());glBindTextureUnit(2,edgeMap);
        edge.bind();glDispatchCompute(2,1,1);glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        std::array<glm::vec4,128> edgeValues{};glGetNamedBufferSubData(edgeBuffer,0,sizeof(edgeValues),edgeValues.data());
        for(const auto& v:edgeValues){
            if(!std::isfinite(v.z)||v.z<0||v.z>1)throw std::runtime_error("Invalid RGBA edge reconstruction");
            edges<<fixture<<','<<v.x<<','<<v.y<<','<<v.z<<'\n';
        }
    }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,6,0);glDeleteBuffers(1,&edgeBuffer);glDeleteTextures(1,&edgeMap);
    if(!edges)throw std::runtime_error("Cannot write edge reconstruction validation");
    if(!out)throw std::runtime_error("Cannot save RGBA validation");
}
void Renderer::validateRgbaMoments(const std::filesystem::path& path){
    unsigned buffer=0;glCreateBuffers(1,&buffer);
    glNamedBufferStorage(buffer,320*sizeof(glm::vec4),nullptr,GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,6,buffer);
    gfx::research::Shader shader;loadShader(shader,{{gfx::research::ShaderStage::Compute,"rgba_moments_validation.comp"}});
    shader.bind();std::ofstream out(path);out<<"fixture,samples,index,angle,margin,raw,model,confidence,reference\n"<<std::setprecision(10);
    for(int fixture=0;fixture<4;++fixture)for(int samples:{16,32,64}){
        shader.set_int("uFixture",fixture);shader.set_int("uSamples",samples);
        glDispatchCompute(5,1,1);glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        std::array<glm::vec4,260> values{};glGetNamedBufferSubData(buffer,0,sizeof(values),values.data());
        for(int i=0;i<260;++i){const auto v=values[std::size_t(i)];
            out<<fixture<<','<<samples<<','<<i<<','<<i/65<<','<<float(i%65-32)*0.0375f<<','<<v.x<<','<<v.y<<','<<v.z<<','<<v.w<<'\n';
        }
    }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,6,0);glDeleteBuffers(1,&buffer);
    if(!out)throw std::runtime_error("Cannot save PCF moment validation");
}
void Renderer::validateRgbaShared(const std::filesystem::path& path){
    unsigned texture=0,buffer=0;glCreateTextures(GL_TEXTURE_2D,1,&texture);
    glTextureStorage2D(texture,1,GL_RGBA32F,16,16);
    glTextureParameteri(texture,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTextureParameteri(texture,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    std::array<glm::vec4,256> depths{};
    for(int y=0;y<16;++y)for(int x=0;x<16;++x)for(int c=0;c<4;++c)
        depths[std::size_t(y*16+x)][c]=(x+y+c)%11==0?1.0f:float((x*13+y*7+x*y*3+c*11)%37+25)/100.0f;
    glTextureSubImage2D(texture,0,0,0,16,16,GL_RGBA,GL_FLOAT,depths.data());
    glCreateBuffers(1,&buffer);glNamedBufferStorage(buffer,256*sizeof(glm::vec4),nullptr,GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,6,buffer);glBindTextureUnit(2,texture);
    gfx::research::Shader shader;loadShader(shader,{{gfx::research::ShaderStage::Compute,"rgba_shared_validation.comp"}});
    shader.bind();shader.set_int("uMapResolution",16);shader.set_int("uPcfMode",3);
    shader.set_float("uDepthBias",0.00015f);shader.set_vec4("uBiasScale",glm::vec4(1));
    shader.set_float("uWarp",0);shader.set_vec2("uFocus",{0,0});
    shader.set_float("uPcfMaxRadius",4);shader.set_float("uCenterWeight",0.35f);
    shader.set_float("uCascadeBlend",0.2f);shader.set_float("uAngularMix",0.6f);
    shader.set_vec3("uSun",glm::normalize(glm::vec3(0.2f,0.3f,1)));
    const glm::vec4 extents(1,1.3f,1.8f,2.5f);shader.set_vec4("uExtents",extents);shader.set_vec4("uDepthRanges",glm::vec4(8));
    for(int i=0;i<4;++i){
        const float e=extents[i];
        auto view=glm::ortho(-e,e,-e,e,-4.0f,4.0f)*glm::rotate(glm::mat4(1),float(i)*0.21f,glm::vec3(0,0,1))*glm::rotate(glm::mat4(1),float(i)*0.1f,glm::vec3(0,1,0));
        view[3][0]=float(i)*0.15f;
        const auto inv=glm::inverse(view);const auto suffix="["+std::to_string(i)+"]";
        shader.set_mat4(("uViews"+suffix).c_str(),view);
        shader.set_vec3(("uAxisX"+suffix).c_str(),glm::vec3(inv[0]));
        shader.set_vec3(("uAxisY"+suffix).c_str(),glm::vec3(inv[1]));
        shader.set_vec3(("uAxisZ"+suffix).c_str(),glm::vec3(inv[2]));
    }
    std::ofstream out(path);out<<"config,mix,cascades,smoothing,adaptive,samples,strength,limit,x,y,visibility,radius,plane_error,base\n"<<std::setprecision(10);
    for(int config=0;config<26;++config){
        const int k=config<24?config:0,adaptive=k%2,smoothing=(k/2)%2,cascades=(k/4)%2,mix=k/8,samples=adaptive?16:8;
        const float strength=config==25?0.0f:0.5f,limit=config==24?0.0f:0.173f;
        shader.set_int("uSharedAdaptive",adaptive);shader.set_int("uSmoothing",smoothing);shader.set_int("uCascades",cascades);shader.set_int("uMixMode",mix);
        shader.set_int("uPcfSamples",samples);shader.set_float("uPcfStrength",strength);shader.set_float("uSharedRadius",limit);
        glDispatchCompute(2,2,1);glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        std::array<glm::vec4,256> values{};glGetNamedBufferSubData(buffer,0,sizeof(values),values.data());
        for(int y=0;y<16;++y)for(int x=0;x<16;++x){const auto v=values[std::size_t(y*16+x)];
            out<<config<<','<<mix<<','<<cascades<<','<<smoothing<<','<<adaptive<<','<<samples<<','<<strength<<','<<limit<<','<<x<<','<<y<<','<<v.x<<','<<v.y<<','<<v.z<<','<<v.w<<'\n';
        }
    }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,6,0);glDeleteBuffers(1,&buffer);glDeleteTextures(1,&texture);
    if(!out)throw std::runtime_error("Cannot save shared RGBA PCF validation");
}
void Renderer::validateRgbaDepthScale(const std::filesystem::path& path){
    unsigned texture=0,buffer=0;glCreateTextures(GL_TEXTURE_2D,1,&texture);
    glTextureStorage2D(texture,1,GL_RGBA32F,128,128);
    glTextureParameteri(texture,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTextureParameteri(texture,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glCreateBuffers(1,&buffer);glNamedBufferStorage(buffer,64*sizeof(glm::vec4),nullptr,GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,6,buffer);glBindTextureUnit(2,texture);
    gfx::research::Shader shader;loadShader(shader,{{gfx::research::ShaderStage::Compute,"rgba_depth_scale_validation.comp"}});
    shader.bind();shader.set_int("uMapResolution",128);shader.set_int("uSmoothing",1);shader.set_int("uPcfMode",4);
    shader.set_float("uDepthBias",0.00015f);shader.set_float("uTanAngle",0.026f);shader.set_float("uWarp",0);shader.set_vec2("uFocus",{0,0});
    shader.set_int("uPcssBlockerSamples",32);shader.set_int("uPcfSamples",16);
    shader.set_float("uPcfStrength",0.5f);shader.set_float("uPcfMaxRadius",16);shader.set_float("uPcssSearchRadius",3);
    std::ofstream out(path);out<<"fixture,extent,mean,bounds,encoding,index,visibility,mean_gap,filter_radius_m,search_radius_m\n"<<std::setprecision(10);
    for(int fixture=0;fixture<3;++fixture)for(float extent:{1.0f,4.0f})for(int mean:{0,1})for(int bounds:{0,1})for(int encoding=0;encoding<4;++encoding){
        const float range=std::ldexp(16.0f,encoding*2),distance=range*0.5f;
        std::vector<glm::vec4> depths(128*128,glm::vec4(1));
        for(int y=0;y<128;++y)for(int x=0;x<128;++x){
            const float px=((float(x)+0.5f)/64.0f-1.0f)*extent;
            float height=0;bool caster=false;
            if(fixture==0){caster=px<0;height=4;}
            if(fixture==1){caster=px< -0.06f||std::abs(px)<0.03f;height=px< -0.06f?4.0f:0.2f;}
            if(fixture==2){caster=true;height=(x/8+y/8)%2?4.0f:0.2f;}
            if(caster)depths[std::size_t(y*128+x)]=glm::vec4((distance-height)/range);
        }
        glTextureSubImage2D(texture,0,0,0,128,128,GL_RGBA,GL_FLOAT,depths.data());
        shader.set_int("uPcssMean",mean);shader.set_int("uPcssUseBounds",bounds);
        shader.set_float("uReceiverDepth",distance/range);shader.set_vec4("uDepthRanges",glm::vec4(range));
        shader.set_vec4("uBiasScale",glm::vec4(80.0f/range));shader.set_vec4("uExtents",glm::vec4(extent));
        shader.set_vec4("uPcssCasterNear",glm::vec4((distance-4.0f)/range));
        glDispatchCompute(1,1,1);glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        std::array<glm::vec4,64> values{};glGetNamedBufferSubData(buffer,0,sizeof(values),values.data());
        for(int i=0;i<64;++i){const auto v=values[std::size_t(i)];out<<fixture<<','<<extent<<','<<mean<<','<<bounds<<','<<encoding<<','<<i<<','<<v.x<<','<<v.y<<','<<v.z<<','<<v.w<<'\n';}
    }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,6,0);glDeleteBuffers(1,&buffer);glDeleteTextures(1,&texture);
    if(!out)throw std::runtime_error("Cannot save PCSS camera-distance validation");
}
void Renderer::validateRgbaPcss(const std::filesystem::path& path){
    unsigned texture=0,buffer=0;glCreateTextures(GL_TEXTURE_2D,1,&texture);
    glTextureStorage2D(texture,1,GL_RGBA32F,16,16);
    glTextureParameteri(texture,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTextureParameteri(texture,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glCreateBuffers(1,&buffer);glNamedBufferStorage(buffer,512*sizeof(glm::vec4),nullptr,GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,6,buffer);glBindTextureUnit(2,texture);
    gfx::research::Shader shader;loadShader(shader,{{gfx::research::ShaderStage::Compute,"rgba_pcss_validation.comp"}});
    shader.bind();shader.set_int("uMapResolution",16);shader.set_int("uPcfMode",4);
    shader.set_float("uDepthBias",0.00015f);shader.set_float("uTanAngle",0.026f);
    shader.set_vec4("uBiasScale",glm::vec4(1));shader.set_vec4("uDepthRanges",glm::vec4(80));shader.set_vec2("uFocus",{0,0});
    std::ofstream out(path);out<<"config,fixture,mean,smoothing,blocker_samples,filter_samples,strength,cap_texels,search_limit_m,layout,warp,x,y,visibility,mean_gap,radius,blockers,search_radius,center_gap,sharp_support\n"<<std::setprecision(10);
    for(int config=0;config<72;++config){
        int fixture=config<32?0:config<64?1+(config-32)/8:3;
        int k=config<32?config:config<64?(config-32)%8:0;
        int mean=k%2,smoothing=(k/2)%2,layout=(k/4)%2,blockers=(k/8)%2?32:8,samples=blockers==32?16:8;
        float warp=float(k/16)*4.0f,strength=0.5f,cap=4.0f,limit=blockers==32?1.5f:0.173f;
        if(config>=64){
            smoothing=1;layout=1;blockers=32;samples=16;limit=3.0f;
            if(config==64)strength=0;
            if(config==65)cap=0;
            if(config==66||config==67){limit=0;mean=config-66;}
            if(config==68||config==69){blockers=1;mean=config-68;}
            if(config==70||config==71){strength=1;cap=0.025f;limit=20;mean=config-70;}
        }
        std::array<glm::vec4,256> depths{};
        for(int y=0;y<16;++y)for(int x=0;x<16;++x)for(int c=0;c<4;++c){
            float d=(x+y+c)%11==0?1.0f:float((x*13+y*7+x*y*3+c*11)%37+25)/100.0f;
            if(fixture==1)d=1;
            if(fixture==2)d=0.31f;
            if(fixture==3)d=x<7?0.45f:x==8?0.4985f:1.0f;
            if(fixture==4){if(c==0)d=1;if(c==2)d=0.4f;if(c==3)d=x==8?0.4985f:1.0f;}
            depths[std::size_t(y*16+x)][c]=d;
        }
        glTextureSubImage2D(texture,0,0,0,16,16,GL_RGBA,GL_FLOAT,depths.data());
        shader.set_int("uPcssMean",mean);shader.set_int("uSmoothing",smoothing);shader.set_int("uPcssBlockerSamples",blockers);
        shader.set_int("uPcfSamples",samples);shader.set_float("uPcfStrength",strength);shader.set_float("uPcfMaxRadius",cap);
        shader.set_float("uPcssSearchRadius",limit);shader.set_float("uWarp",warp);
        shader.set_vec4("uExtents",layout?glm::vec4(1,4,16,64):glm::vec4(1));
        glDispatchCompute(2,2,1);glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        std::array<glm::vec4,512> values{};glGetNamedBufferSubData(buffer,0,sizeof(values),values.data());
        for(int y=0;y<16;++y)for(int x=0;x<16;++x){
            const auto v=values[std::size_t((y*16+x)*2)],s=values[std::size_t((y*16+x)*2+1)];
            out<<config<<','<<fixture<<','<<mean<<','<<smoothing<<','<<blockers<<','<<samples<<','<<strength<<','<<cap<<','<<limit<<','<<layout<<','<<warp<<','<<x<<','<<y<<','<<v.x<<','<<v.y<<','<<v.z<<','<<v.w<<','<<s.x<<','<<s.y<<','<<s.z<<'\n';
        }
    }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,6,0);glDeleteBuffers(1,&buffer);glDeleteTextures(1,&texture);
    if(!out)throw std::runtime_error("Cannot save per-channel PCSS validation");
}
void Renderer::validateRgbaPcf(const std::filesystem::path& path){
    unsigned texture=0,buffer=0;
    glCreateTextures(GL_TEXTURE_2D,1,&texture);glTextureStorage2D(texture,1,GL_RGBA32F,16,16);
    glTextureParameteri(texture,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTextureParameteri(texture,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    std::array<glm::vec4,256> depths{};
    for(int y=0;y<16;++y)for(int x=0;x<16;++x)for(int c=0;c<4;++c)
        depths[std::size_t(y*16+x)][c]=(x+y+c)%11==0?1.0f:float((x*13+y*7+x*y*3+c*11)%37+25)/100.0f;
    glTextureSubImage2D(texture,0,0,0,16,16,GL_RGBA,GL_FLOAT,depths.data());
    glCreateBuffers(1,&buffer);glNamedBufferStorage(buffer,256*sizeof(glm::vec4),nullptr,GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,6,buffer);glBindTextureUnit(2,texture);
    gfx::research::Shader shader;loadShader(shader,{{gfx::research::ShaderStage::Compute,"rgba_pcf_validation.comp"}});
    shader.bind();shader.set_int("uMapResolution",16);shader.set_float("uDepthBias",0.00015f);
    shader.set_float("uTanAngle",0.026f);shader.set_vec4("uBiasScale",glm::vec4(1));shader.set_vec4("uDepthRanges",glm::vec4(80));shader.set_vec4("uExtents",glm::vec4(1));
    shader.set_float("uWarp",0);shader.set_vec2("uFocus",{0,0});shader.set_float("uPcfRadius",2);shader.set_float("uPcfMaxRadius",4);
    std::ofstream out(path);out<<"config,mode,smoothing,samples,strength,x,y,visibility,radius,sharp_support,world_units,world_radius_m,extent,moment_gap,blocker_gap\n"<<std::setprecision(10);
    int config=0;
    for(int mode:{1,2})for(int smoothing:{0,1})for(int samples:{8,16})for(float strength:{0.0f,0.5f,1.0f}){
        shader.set_int("uPcfMode",mode);shader.set_int("uSmoothing",smoothing);shader.set_int("uPcfSamples",samples);shader.set_float("uPcfStrength",strength);
        glDispatchCompute(2,2,1);glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        std::array<glm::vec4,256> values{};glGetNamedBufferSubData(buffer,0,sizeof(values),values.data());
        for(int y=0;y<16;++y)for(int x=0;x<16;++x){const auto value=values[std::size_t(y*16+x)];out<<config<<','<<mode<<','<<smoothing<<','<<samples<<','<<strength<<','<<x<<','<<y<<','<<value.x<<','<<value.y<<','<<value.z<<",0,0,1,0,"<<value.w<<'\n';}
        ++config;
    }
    for(int worldUnits:{0,1})for(float extent:{1.0f,4.0f,16.0f,64.0f})for(int momentGap:{0,1})for(float strength:{0.0f,0.5f,1.0f}){
        shader.set_int("uPcfMode",1);shader.set_int("uSmoothing",1);shader.set_int("uPcfSamples",16);
        shader.set_int("uPcfWorldUnits",worldUnits);shader.set_float("uPcfWorldRadius",0.173f);shader.set_float("uPcfStrength",strength);
        shader.set_vec4("uExtents",glm::vec4(extent));shader.set_int("uPcfMomentGap",momentGap);
        glDispatchCompute(2,2,1);glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        std::array<glm::vec4,256> values{};glGetNamedBufferSubData(buffer,0,sizeof(values),values.data());
        for(int y=0;y<16;++y)for(int x=0;x<16;++x){const auto v=values[std::size_t(y*16+x)];
            out<<config<<",1,1,16,"<<strength<<','<<x<<','<<y<<','<<v.x<<','<<v.y<<','<<v.z<<','<<worldUnits<<",0.173,"<<extent<<','<<momentGap<<','<<v.w<<'\n';
        }
        ++config;
    }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,6,0);glDeleteBuffers(1,&buffer);glDeleteTextures(1,&texture);
    if(!out)throw std::runtime_error("Cannot save RGBA PCF validation");
}
void Renderer::prepareRgba(){
    using S=gfx::research::ShaderStage;
    if(!rgbaRaster_.valid()){
        loadShader(rgbaRaster_,{{S::Vertex,"map.vert"},{S::TessellationControl,"map.tesc"},{S::TessellationEvaluation,"map.tese"},{S::Fragment,"map.frag"}},"#define RGBA_DEPTH 1\n");
        loadShader(rgbaAnalytic_,{{S::Vertex,"fullscreen.vert"},{S::Fragment,"map_analytic.frag"}},"#define RGBA_DEPTH 1\n");
    }
    auto& shader=rgbaShaders_.at(std::size_t(settings_.multiViewStorage));
    if(!shader.valid())
        loadShader(shader,{{S::Vertex,"fullscreen.vert"},{S::Fragment,"angular_rgba.frag"}},
                   "#define MULTIVIEW_STORAGE "+std::to_string(settings_.multiViewStorage)+"\n");
    if(rgbaResolution_==settings_.mapResolution&&rgbaStorage_==settings_.multiViewStorage)return;
    glDeleteTextures(1,&rgbaTexture_);rgbaTexture_=0;
    glDeleteTextures(4,rgbaSeparate_.data());rgbaSeparate_.fill(0);
    if(!rgbaFbo_)glCreateFramebuffers(1,&rgbaFbo_);
    glNamedFramebufferTexture(rgbaFbo_,GL_COLOR_ATTACHMENT0,0,0);
    glNamedFramebufferTexture(rgbaFbo_,GL_DEPTH_ATTACHMENT,0,0);
    const int size=settings_.mapResolution;
    const auto setSampling=[](unsigned texture){
        glTextureParameteri(texture,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        glTextureParameteri(texture,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        glTextureParameteri(texture,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        glTextureParameteri(texture,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    };
    if(settings_.multiViewStorage==0){
        glCreateTextures(GL_TEXTURE_2D,1,&rgbaTexture_);
        glTextureStorage2D(rgbaTexture_,1,GL_RGBA32F,size,size);setSampling(rgbaTexture_);
        glNamedFramebufferTexture(rgbaFbo_,GL_COLOR_ATTACHMENT0,rgbaTexture_,0);
        const GLenum draw=GL_COLOR_ATTACHMENT0;glNamedFramebufferDrawBuffers(rgbaFbo_,1,&draw);
        glNamedFramebufferReadBuffer(rgbaFbo_,draw);
    }else{
        glNamedFramebufferDrawBuffer(rgbaFbo_,GL_NONE);glNamedFramebufferReadBuffer(rgbaFbo_,GL_NONE);
        if(settings_.multiViewStorage==1){
            glCreateTextures(GL_TEXTURE_2D_ARRAY,1,&rgbaTexture_);
            glTextureStorage3D(rgbaTexture_,1,GL_DEPTH_COMPONENT32F,size,size,4);setSampling(rgbaTexture_);
            glTextureParameteri(rgbaTexture_,GL_TEXTURE_COMPARE_MODE,GL_NONE);
            glNamedFramebufferTextureLayer(rgbaFbo_,GL_DEPTH_ATTACHMENT,rgbaTexture_,0,0);
        }else{
            glCreateTextures(GL_TEXTURE_2D,4,rgbaSeparate_.data());
            for(unsigned texture:rgbaSeparate_){
                glTextureStorage2D(texture,1,GL_DEPTH_COMPONENT32F,size,size);setSampling(texture);
                glTextureParameteri(texture,GL_TEXTURE_COMPARE_MODE,GL_NONE);
            }
            glNamedFramebufferTexture(rgbaFbo_,GL_DEPTH_ATTACHMENT,rgbaSeparate_[0],0);
        }
    }
    if(glCheckNamedFramebufferStatus(rgbaFbo_,GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)
        throw std::runtime_error("Multiple light view framebuffer incomplete");
    rgbaResolution_=size;rgbaStorage_=settings_.multiViewStorage;
}
std::vector<float> Renderer::readMultiViewDepth() const {
    const int size=rgbaResolution_;
    if(rgbaStorage_==0)return readTexture(rgbaTexture_,size,size,GL_RGBA,4);
    const auto pixels=std::size_t(size)*size;
    std::vector<float> values(pixels*4);
    if(rgbaStorage_==1){
        const auto layers=readTexture(rgbaTexture_,size,size*4,GL_DEPTH_COMPONENT,1);
        for(std::size_t p=0;p<pixels;++p)for(int c=0;c<4;++c)values[p*4+std::size_t(c)]=layers[std::size_t(c)*pixels+p];
    }else for(int c=0;c<4;++c){
        const auto layer=readTexture(rgbaSeparate_[std::size_t(c)],size,size,GL_DEPTH_COMPONENT,1);
        for(std::size_t p=0;p<pixels;++p)values[p*4+std::size_t(c)]=layer[p];
    }
    return values;
}
void Renderer::rgbaShadow(){
    result_.bind();const GLenum attachments[]={GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1};
    glNamedFramebufferDrawBuffers(result_.id(),2,attachments);
    auto& shader=rgbaShaders_.at(std::size_t(settings_.multiViewStorage));
    shader.bind();bindScreen(shader);configureShadow(shader,Method::AngularRgba);
    if(settings_.multiViewStorage==2){
        for(int i=0;i<4;++i)glBindTextureUnit(unsigned(2+i),rgbaSeparate_[std::size_t(i)]);
    }else glBindTextureUnit(2,rgbaTexture_);
    for(int i=0;i<4;++i){
        const auto suffix="["+std::to_string(i)+"]";
        const auto inverse=glm::inverse(rgbaVp_[std::size_t(i)]);
        shader.set_mat4(("uViews"+suffix).c_str(),rgbaVp_[std::size_t(i)]);
        shader.set_vec3(("uAxisX"+suffix).c_str(),glm::vec3(inverse[0]));
        shader.set_vec3(("uAxisY"+suffix).c_str(),glm::vec3(inverse[1]));
        shader.set_vec3(("uAxisZ"+suffix).c_str(),glm::vec3(inverse[2]));
    }
    shader.set_vec4("uBiasScale",rgbaBiasScale_);
    shader.set_vec4("uExtents",rgbaExtents_);
    shader.set_vec4("uDepthRanges",rgbaDepthRanges_);
    shader.set_int("uPcfSharp",settings_.rgbaPcfSharp?1:0);
    shader.set_int("uJointBlend",usesResultBlend(Method::AngularRgba,settings_)?1:0);
    shader.set_int("uPcfMode",settings_.rgbaPcfMode);
    shader.set_int("uPcfSamples",settings_.rgbaPcfSamples);
    shader.set_float("uPcfStrength",settings_.rgbaPcfStrength);
    shader.set_float("uPcfRadius",settings_.rgbaPcfRadius);
    shader.set_int("uPcfWorldUnits",settings_.rgbaPcfWorldUnits?1:0);
    shader.set_float("uPcfWorldRadius",settings_.rgbaPcfWorldRadius);
    shader.set_int("uPcssBlockerSamples",settings_.rgbaPcssBlockerSamples);
    shader.set_int("uPcssMean",settings_.rgbaPcssMean);
    shader.set_float("uPcssSearchRadius",settings_.rgbaPcssSearchRadius);
    shader.set_int("uPcssUseBounds",settings_.rgbaPcssCasterBounds?1:0);
    shader.set_vec4("uPcssCasterNear",rgbaCasterNear_);
    shader.set_int("uPcfMomentGap",usesMomentReconstruction(Method::AngularRgba,settings_)&&settings_.rgbaPcfMode==1?1:0);
    shader.set_float("uGeometricBlend",settings_.rgbaGeometricBlend);
    shader.set_float("uPcfMaxRadius",settings_.rgbaPcfMaxRadius);
    shader.set_float("uSharedRadius",settings_.rgbaSharedRadius);
    shader.set_int("uSharedAdaptive",settings_.rgbaSharedAdaptive?1:0);
    shader.set_int("uMoments",usesMomentReconstruction(Method::AngularRgba,settings_)?1:0);
    shader.set_float("uMomentStrength",settings_.rgbaMomentStrength);
    shader.set_int("uReconstruct",settings_.rgbaPcfMode==0&&settings_.rgbaReconstruct&&settings_.rgbaOffset&&settings_.rgbaRadius>=0.1f&&settings_.rgbaMixMode==0?1:0);
    shader.set_float("uEdgeContactRange",settings_.rgbaEdgeContactRange);
    shader.set_int("uEdgeSearch",settings_.rgbaEdgeSearch);
    shader.set_float("uReconstructStrength",settings_.rgbaReconstructStrength);
    shader.set_vec3("uSun",glm::vec3(sunDirection()));
    shader.set_float("uTanAngle",std::tan(glm::radians(settings_.angle)));
    shader.set_float("uTriRadius",settings_.rgbaRadius);
    shader.set_float("uTriRotation",glm::radians(settings_.rgbaRotation));
    shader.set_int("uCascades",settings_.rgbaCascades?1:0);
    shader.set_int("uMixMode",settings_.rgbaMixMode);
    shader.set_float("uCascadeBlend",settings_.rgbaCascadeBlend);
    shader.set_float("uAngularMix",settings_.rgbaAngularMix);
    shader.set_float("uDepthBias",settings_.depthBias);
    shader.set_float("uCenterWeight",settings_.rgbaCenterWeight);
    shader.set_int("uSmoothing",settings_.rgbaSmoothing);
    fullscreen_.draw();
}
void Renderer::maps(Method method, int dropped) {
    if(method==Method::AngularRgba){
        const auto saved=lightVp_,savedInverse=inverseLightVp_;
        const glm::vec3 center=!settings_.rgbaStableFocus&&settings_.rgbaCascades?(settings_.customCamera?settings_.cameraTarget:scene_.target):glm::vec3(0,1.5f,0),sun=glm::vec3(sunDirection());
        const auto tangent=glm::normalize(glm::cross(glm::vec3(0,0,1),sun));
        const auto bitangent=glm::cross(sun,tangent);
        if(settings_.multiViewStorage==0){const float clear[]={1,1,1,1};glClearTexImage(rgbaTexture_,0,GL_RGBA,GL_FLOAT,clear);}
        for(int i=0;i<4;++i){
            const float phi=glm::radians(settings_.rgbaRotation+120.0f*float(i-1));
            const float radius=i==0||!settings_.rgbaOffset?0.0f:std::tan(glm::radians(settings_.angle*settings_.rgbaRadius));
            const auto direction=glm::normalize(sun+radius*(std::cos(phi)*tangent+std::sin(phi)*bitangent));
            const float extent=settings_.rgbaExtent*(settings_.rgbaCascades?std::pow(settings_.rgbaCascadeScale,float(i)):1.0f);
            const float distance=extent<=14.0f?30.0f:extent*3.0f,range=extent<=14.0f?80.0f:extent*6.0f;
            rgbaBiasScale_[i]=80.0f/range;rgbaExtents_[i]=extent;rgbaDepthRanges_[i]=range;
            lightVp_=glm::ortho(-extent,extent,-extent,extent,0.1f,range+0.1f)*glm::lookAt(center+direction*distance,center,glm::vec3(0,0,1));
            inverseLightVp_=glm::inverse(lightVp_);rgbaVp_[std::size_t(i)]=lightVp_;
            rgbaCasterNear_[i]=settings_.rgbaPcssCasterBounds&&settings_.rgbaPcfMode==4?casterNearDepth(scene_,lightVp_,dropped):0.0f;
            rgbaChannel_=i;maps(Method::Hard,dropped);
        }
        rgbaChannel_=-1;lightVp_=saved;inverseLightVp_=savedInverse;
        glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);glDisable(GL_BLEND);glBlendEquation(GL_FUNC_ADD);
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT|GL_FRAMEBUFFER_BARRIER_BIT);
        return;
    }
    if (isAngularRayMethod(method))
        return;
    const bool colorView=rgbaChannel_>=0&&settings_.multiViewStorage==0;
    const unsigned framebuffer=rgbaChannel_>=0?rgbaFbo_:mapFbo_;
    glBindFramebuffer(GL_FRAMEBUFFER,framebuffer);
    glViewport(0, 0, settings_.mapResolution, settings_.mapResolution);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    const bool packed = usesPackedDepth(method, settings_);
    if(packed) glDisable(GL_DEPTH_TEST); else glEnable(GL_DEPTH_TEST);
    glDepthFunc(settings_.analyticMaps ? GL_ALWAYS : GL_LESS);
    auto& shader = packed ? (settings_.analyticMaps ? intervalAnalytic_ : intervalRaster_)
                          : (settings_.analyticMaps ? mapAnalytic_ : mapRaster_);
    auto& activeShader=colorView?(settings_.analyticMaps?rgbaAnalytic_:rgbaRaster_):shader;
    activeShader.bind();
    configureShadow(activeShader, method);
    activeShader.set_float("uTessellation", settings_.warp > 0 ? float(scene_.isStress() ? settings_.stressTessellation : 32) : 1.0f);
    activeShader.set_int("uAdaptiveTessellation", scene_.isStress() && settings_.stressAdaptiveTessellation ? 1 : 0);
    activeShader.set_int("uStressMode", 0);
    activeShader.set_int("uDroppedGroup", dropped);
    if(colorView){
        glDisable(GL_DEPTH_TEST);glEnable(GL_BLEND);glBlendEquation(GL_MIN);glBlendFunc(GL_ONE,GL_ONE);
        glColorMask(rgbaChannel_==0,rgbaChannel_==1,rgbaChannel_==2,rgbaChannel_==3);
    }else if(rgbaChannel_>=0){
        glDepthMask(GL_TRUE);
        if(settings_.multiViewStorage==1)
            glNamedFramebufferTextureLayer(rgbaFbo_,GL_DEPTH_ATTACHMENT,rgbaTexture_,0,rgbaChannel_);
        else glNamedFramebufferTexture(rgbaFbo_,GL_DEPTH_ATTACHMENT,rgbaSeparate_[std::size_t(rgbaChannel_)],0);
    } else if(packed) {
        glNamedFramebufferTexture(mapFbo_, GL_DEPTH_ATTACHMENT, 0, 0);
        glNamedFramebufferParameteri(mapFbo_, GL_FRAMEBUFFER_DEFAULT_WIDTH, settings_.mapResolution);
        glNamedFramebufferParameteri(mapFbo_, GL_FRAMEBUFFER_DEFAULT_HEIGHT, settings_.mapResolution);
        const unsigned empty=0xffff0000u;
        glClearTexImage(mapTexture_,0,GL_RED_INTEGER,GL_UNSIGNED_INT,&empty);
        glBindImageTexture(0,mapTexture_,0,GL_TRUE,0,GL_READ_WRITE,GL_R32UI);
    } else glNamedFramebufferTextureLayer(mapFbo_, GL_DEPTH_ATTACHMENT, mapTexture_, 0, 0);
    if (glCheckNamedFramebufferStatus(framebuffer, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw std::runtime_error("Shadow framebuffer incomplete");
    const float clearDepth = 1;
    if(!packed&&!colorView)glClearNamedFramebufferfv(framebuffer,GL_DEPTH,0,&clearDepth);
    if (settings_.analyticMaps) {
        bindScene(activeShader, dropped);
        activeShader.set_int("uIncludeReceiver", usesPrefilter(method) ? 1 : 0);
        activeShader.set_vec3("uReceiverNormal", {-std::sin(scene_.receiverTilt), std::cos(scene_.receiverTilt), 0});
        fullscreen_.draw();
    } else {
        glPatchParameteri(GL_PATCH_VERTICES, 3);
        if (usesPrefilter(method)) {
            activeShader.set_mat4("uModel", glm::rotate(glm::mat4(1), scene_.receiverTilt, glm::vec3(0, 0, 1)));
            glBindVertexArray(plane_.vao());
            glDrawElements(GL_PATCHES, GLsizei(plane_.index_count()), GL_UNSIGNED_INT, nullptr);
        }
        if (scene_.isStress()) {
            activeShader.set_int("uStressMode", 1);
            activeShader.set_int("uStressGrid", stressGrid(scene_.stressObjects));
            glBindVertexArray(cube_.vao());
            if (scene_.stressObjects > 0)
                glDrawElementsInstanced(GL_PATCHES, GLsizei(cube_.index_count()), GL_UNSIGNED_INT, nullptr, scene_.stressObjects);
        } else for (const auto& b : scene_.boxes) {
            if (b.group == dropped)
                continue;
            activeShader.set_mat4("uModel", boxModel(b));
            glBindVertexArray(cube_.vao());
            glDrawElements(GL_PATCHES, GLsizei(cube_.index_count()), GL_UNSIGNED_INT, nullptr);
        }
    }
    glDisable(GL_DEPTH_TEST);
    if(packed) {
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
        glBindImageTexture(0,0,0,GL_TRUE,0,GL_READ_WRITE,GL_R32UI);
    }
}
void Renderer::shadow(Method method, int dropped, bool oracle) {
    glViewport(0, 0, settings_.width, settings_.height);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    if(method==Method::AngularRgba){rgbaShadow();return;}
    if (isAngularRayMethod(method)) {
        const int smoothing=oracle?0:angularSmoothing(method,settings_);
        const bool accumulate=smoothing!=0;
        if(accumulate)accum_.bind();else result_.bind();
        if(!accumulate){const GLenum attachment=GL_COLOR_ATTACHMENT0;glNamedFramebufferDrawBuffers(result_.id(),1,&attachment);}
        auto& program=angularProgram(method,accumulate);
        program.bind();bindScreen(program);bindScene(program,dropped);
        program.set_vec3("uSun",glm::vec3(sunDirection()));
        program.set_float("uTanAngle",std::tan(glm::radians(settings_.angle)));
        program.set_int("uRaySamples",oracle?settings_.referenceSamples:angularSamples(method,settings_));
        program.set_int("uSignedMargin",settings_.triSignedMargin?1:0);
        program.set_int("uTriResultMasks",settings_.triResultMasks?1:0);
        program.set_float("uTriRadius",settings_.triRadius);
        program.set_float("uTriRotation",glm::radians(settings_.triRotation));
        if(method==Method::Reference)glBindBufferBase(GL_SHADER_STORAGE_BUFFER,7,angularDirections_);
        fullscreen_.draw();
        if(accumulate){
            result_.bind();const GLenum attachment=GL_COLOR_ATTACHMENT0;
            glNamedFramebufferDrawBuffers(result_.id(),1,&attachment);
            resolveShader_.bind();
            resolveShader_.set_mat4("uInverseVp",inverseVp_);
            resolveShader_.set_int("uArrSmoothing",smoothing);
            resolveShader_.set_float("uArrSmoothingRadius",settings_.arrSmoothingRadius);
            glBindTextureUnit(0,accum_.color());glBindTextureUnit(1,gbuffer_.depth());glBindTextureUnit(2,gbuffer_.color(1));
            fullscreen_.draw();
        }
    } else {
        result_.bind();
        const GLenum attachments[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
        glNamedFramebufferDrawBuffers(result_.id(), 2, attachments);
        auto& shadowShader = shadowProgram(method);
        if (method == Method::Chs)
            allocateChsWeights();
        shadowShader.bind();
        bindScreen(shadowShader);
        configureShadow(shadowShader, method);
        glBindTextureUnit(2, mapTexture_);
        shadowShader.set_int("uFilterSamples", settings_.filterSamples);
        shadowShader.set_int("uBlockerSamples", settings_.blockerSamples);
        shadowShader.set_int("uRotateSamples", settings_.rotateSamples ? 1 : 0);
        shadowShader.set_int("uSamplePattern", effectiveSamplePattern(method, settings_));
        if(effectiveSamplePattern(method,settings_)==3){
            blueNoise_.bind(5);
            shadowShader.set_float("uBlueNoiseTile",settings_.blueNoiseTile);
        }
        shadowShader.set_int("uSampleSeed", settings_.sampleSeed);
        shadowShader.set_int("uSharedSamples", settings_.sharedSamplePattern ? 1 : 0);
        shadowShader.set_int("uInnerPenumbra", settings_.innerPenumbra ? 1 : 0);
        shadowShader.set_float("uContactGuard",settings_.contactGuard);
        shadowShader.set_float("uContactMaxGap",settings_.contactMaxGap);
        shadowShader.set_float("uLayerSeparation", settings_.layerSeparation);
        shadowShader.set_float("uLayerMinFilterTexels", settings_.layerMinFilterTexels);
        if (usesPrefilter(method)) {
            glBindTextureUnit(3, basisTextures_[0]);
            shadowShader.set_float("uVsmMinVariance", settings_.vsmMinVariance);
            shadowShader.set_float("uBleedReduction", settings_.bleedReduction);
            shadowShader.set_float("uEsmExponent", settings_.esmExponent);
            shadowShader.set_float("uCsmBias", settings_.csmBias);
            shadowShader.set_int("uCsmTerms", settings_.csmTerms);
            shadowShader.set_float("uGsmSharpness",settings_.gsmSharpness);
            shadowShader.set_int("uEsmFallback", settings_.esmFallback ? 1 : 0);
        }
        shadowShader.set_int("uDpcfTaps", settings_.dpcfTaps);
        shadowShader.set_vec3("uSun", glm::vec3(sunDirection()));
        shadowShader.set_float("uTanAngle", std::tan(glm::radians(settings_.angle)));
        shadowShader.set_float("uDepthRange", 80);
        shadowShader.set_float("uExtent", 14);
        shadowShader.set_float("uDepthBias", settings_.depthBias);
        shadowShader.set_float("uMaxRadiusWorld", settings_.maxRadiusWorld);
        shadowShader.set_float("uFixedPcfRadius", settings_.fixedPcfRadius);
        shadowShader.set_float("uDpcfRadius", settings_.dpcfRadius);
        shadowShader.set_float("uDpcfRatioScale", settings_.dpcfRatioScale);
        shadowShader.set_float("uChsSunSize", settings_.chsSunSize);
        fullscreen_.draw();
    }
}
Timing Renderer::render(Method method, int dropped, bool timed, bool oracle) {
    lastMethod_ = method;
    resultBlended_=false;
    if(usesResultBlend(method,settings_))prepareResultBlend();
    if(effectiveSamplePattern(method,settings_)==3)prepareBlueNoise();
    const bool packed = usesPackedDepth(method, settings_);
    if(method==Method::AngularRgba)prepareRgba();
    else if(!isAngularRayMethod(method)) allocateMap(packed);
    if(packed && !intervalRaster_.valid()) {
        using S = gfx::research::ShaderStage;
        const std::string defines="#define PACKED_INTERVALS 1\n";
        loadShader(intervalRaster_, {{S::Vertex,"map.vert"},{S::TessellationControl,"map.tesc"},{S::TessellationEvaluation,"map.tese"},{S::Fragment,"map.frag"}},defines);
        loadShader(intervalAnalytic_,{{S::Vertex,"fullscreen.vert"},{S::Fragment,"map_analytic.frag"}},defines);
    }
    if(isAngularRayMethod(method)){
        (void)angularProgram(method,!oracle&&angularSmoothing(method,settings_)!=0);
        if(method==Method::Reference)prepareAngularDirections();
    } else
        (void)shadowProgram(method); // compilation stays outside GPU timing scopes
    if (usesPrefilter(method))
        preparePrefilter(method);
    Timing result;
    unsigned q[3] = {};
    if (timed) {
        glCreateQueries(GL_TIMESTAMP, 3, q);
        glQueryCounter(q[0], GL_TIMESTAMP);
    }
    maps(method, dropped);
    if (usesPrefilter(method))
        prefilter(method);
    if (timed)
        glQueryCounter(q[1], GL_TIMESTAMP);
    shadow(method, dropped, oracle);
    if(usesResultBlend(method,settings_)&&!oracle)blendResult();
    if (timed) {
        glQueryCounter(q[2], GL_TIMESTAMP);
        GLuint64 t[3] = {};
        for (int i = 0; i < 3; ++i)
            glGetQueryObjectui64v(q[i], GL_QUERY_RESULT, &t[i]);
        result.mapMs = double(t[1] - t[0]) * 1e-6;
        result.shadowMs = double(t[2] - t[1]) * 1e-6;
        glDeleteQueries(3, q);
    }
    return result;
}
Image Renderer::read() const {
    Image image;
    image.width = settings_.width;
    image.height = settings_.height;
    image.visibility = readTexture(resultTexture(), image.width, image.height, GL_RED, 1);
    image.depth = readTexture(gbuffer_.depth(), image.width, image.height, GL_DEPTH_COMPONENT, 1);
    const auto normal = normals();
    image.receiver.resize(image.visibility.size());
    for (std::size_t i = 0; i < image.receiver.size(); ++i)
        image.receiver[i] = normal[i * 4 + 3];
    return image;
}
std::vector<float> Renderer::positions() const {
    auto depth = readTexture(gbuffer_.depth(), settings_.width, settings_.height, GL_DEPTH_COMPONENT, 1);
    std::vector<float> out(depth.size() * 3);
    for (int y = 0; y < settings_.height; ++y)
        for (int x = 0; x < settings_.width; ++x) {
            auto i = std::size_t(y) * std::size_t(settings_.width) + std::size_t(x);
            glm::vec4 p =
                inverseVp_ * glm::vec4((float(x) + 0.5f) / float(settings_.width) * 2 - 1,
                                       (float(y) + 0.5f) / float(settings_.height) * 2 - 1, depth[i] * 2 - 1, 1);
            for (int k = 0; k < 3; ++k)
                out[i * 3 + std::size_t(k)] = p[k] / p.w;
        }
    return out;
}
std::vector<float> Renderer::normals() const {
    return readTexture(gbuffer_.color(1), settings_.width, settings_.height, GL_RGBA, 4);
}
std::vector<float> Renderer::diagnostics() const {
    return readTexture(result_.color(1), settings_.width, settings_.height, GL_RGBA, 4);
}
void Renderer::display(int width, int height, int view, unsigned reference) {
    glViewport(0, 0, width, height);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_FRAMEBUFFER_SRGB);
    displayShader_.bind();
    displayShader_.set_int("uView", view);
    displayShader_.set_int(
        "uDiagnosticValid",
        lastMethod_ == Method::Hard || lastMethod_ == Method::Pcf || isAngularRayMethod(lastMethod_) ? 0 : 1);
    displayShader_.set_vec3("uSun", glm::vec3(sunDirection()));
    glBindTextureUnit(0, resultTexture());
    glBindTextureUnit(1, gbuffer_.color(1));
    glBindTextureUnit(2, gbuffer_.color());
    glBindTextureUnit(3, reference ? reference : result_.color());
    glBindTextureUnit(4, result_.color(1));
    fullscreen_.draw();
}
void Renderer::capture(const std::filesystem::path& path, int view, unsigned reference) {
    preview_.bind();
    display(settings_.width, settings_.height, view, reference);
    if (!gfx::research::save_framebuffer_png(path, settings_.width, settings_.height))
        throw std::runtime_error("Cannot write PNG: " + path.string());
}
unsigned Renderer::copyResult() const {
    unsigned texture = 0;
    glCreateTextures(GL_TEXTURE_2D, 1, &texture);
    glTextureStorage2D(texture, 1, GL_R32F, settings_.width, settings_.height);
    glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glCopyImageSubData(resultTexture(), GL_TEXTURE_2D, 0, 0, 0, 0, texture, GL_TEXTURE_2D, 0, 0, 0, 0, settings_.width,
                       settings_.height, 1);
    return texture;
}
unsigned Renderer::resultTexture() const {
    return resultBlended_?blended_.color():result_.color();
}
} // namespace shadows
