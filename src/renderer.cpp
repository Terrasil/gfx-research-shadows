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
constexpr std::array<const char*, methodCount> ids = {"hard", "pcf", "pcss", "arr", "geometric", "dpcf", "chs", "geometric-layered", "vsm", "csm", "esm", "arr-triangulation"};
constexpr std::array<const char*, methodCount> labels = {"Hard shadow",
                                                         "Fixed PCF",
                                                         "PCSS mean blocker",
                                                         "Angular ray reference (exact boxes)",
                                                         "PCSS geometric mean",
                                                         "DPCF (Filament v1.23.1)",
                                                         "CHS (AMD ShadowFX fixed gather4)",
                                                         "Geometric depth bands (paired samples)",
                                                         "VSM (Gaussian moments)", "CSM (Fourier convolution)",
                                                         "ESM (exponential + fallback)", "ARR triangulation"};
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
bool usesPrefilter(Method method) {
    return method == Method::Vsm || method == Method::Csm || method == Method::Esm;
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
    return 1+(angularSmoothing(method,s)!=0?1:0);
}
std::uint64_t auxiliaryMapBytes(Method method, const Settings& settings) {
    const std::uint64_t pixels = std::uint64_t(settings.mapResolution) * settings.mapResolution;
    return pixels * (method == Method::Vsm ? 16u : method == Method::Esm ? 8u : method == Method::Csm ? 16u * settings.csmTerms : 0u);
}
int prefilterPasses(Method method) {
    return usesPrefilter(method) ? 3 : 0;
}
void Renderer::preparePrefilter(Method method) {
    const int slot = int(method) - int(Method::Vsm);
    const int layers = method == Method::Csm ? settings_.csmTerms / 2 : 1;
    const GLenum format = method == Method::Vsm ? GL_RG32F : method == Method::Esm ? GL_R32F : GL_RGBA32F;
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
    const int slot = int(method) - int(Method::Vsm), size = settings_.mapResolution;
    const GLenum format = method == Method::Vsm ? GL_RG32F : method == Method::Esm ? GL_R32F : GL_RGBA32F;
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
    out << '\n' << std::setprecision(10);
    unsigned buffer = 0;
    std::array<float, 128> visibility;
    glCreateBuffers(1, &buffer);
    glNamedBufferStorage(buffer, sizeof(visibility), nullptr, GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);
    for (int slot = 0; slot < 3; ++slot) {
        const auto method = Method(int(Method::Vsm) + slot);
        preparePrefilter(method);
        prefilter(method);
        gfx::research::Shader shader;
        loadShader(shader, {{gfx::research::ShaderStage::Compute, "basis_validation.comp"}}, "#define BASIS_METHOD " + std::to_string(slot) + "\n");
        shader.bind();
        glBindTextureUnit(3, basisTextures_[0]);
        glDispatchCompute(128, 1, 1);
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT);
        glGetNamedBufferSubData(buffer, 0, sizeof(visibility), visibility.data());
        const int channels = slot == 0 ? 2 : slot == 1 ? 4 : 1;
        std::vector<float> data(std::size_t(32 * 32 * basisLayers_ * channels));
        glGetTextureImage(basisTextures_[0], 0, channels == 1 ? GL_RED : channels == 2 ? GL_RG : GL_RGBA, GL_FLOAT, GLsizei(data.size() * sizeof(float)), data.data());
        for (int i = 0; i < 128; ++i) {
            const int x = (i * 7) % 32, y = (i * 13 + i / 32) % 32;
            out << methodId(method) << ',' << i << ',' << x << ',' << y << ',' << float(i % 17) / 16.0f << ',' << visibility[i];
            for (int c = 0; c < 16; ++c) {
                float value = 0;
                if (c < channels * basisLayers_)
                    value = data[std::size_t(((c / channels) * 32 * 32 + y * 32 + x) * channels + c % channels)];
                out << ',' << value;
            }
            out << '\n';
        }
    }
    if (!out) throw std::runtime_error("Cannot write prefilter fixtures");
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
    glDeleteBuffers(1, &buffer);
    glDeleteTextures(1, &mapTexture_);
    mapTexture_ = savedMap;
    settings_ = savedSettings;
}
int activeMapLayers(Method m, const Settings&) {
    return isAngularRayMethod(m) ? 0 : 1;
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
    return std::uint64_t(activeMapLayers(m, s)) * s.mapResolution * s.mapResolution * 4u + auxiliaryMapBytes(m, s);
}
int mapGenerationPasses(Method m, const Settings& s) {
    return activeMapLayers(m, s);
}
const char* methodId(Method m) {
    return ids.at(std::size_t(m));
}
const char* methodLabel(Method m) {
    return labels.at(std::size_t(m));
}
Method parseMethod(const std::string& name) {
    for (int i = 0; i < methodCount; ++i)
        if (name == ids[std::size_t(i)])
            return Method(i);
    throw std::runtime_error("Unknown method: " + name);
}
const char* samplePatternId(int pattern) {
    constexpr std::array<const char*, 3> patterns = {"vogel", "random", "stratified"};
    return patterns.at(std::size_t(pattern));
}
int parseSamplePattern(const std::string& name) {
    for (int i = 0; i < 3; ++i)
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
Renderer::Renderer() = default;
Renderer::~Renderer() {
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
void Renderer::allocateChsWeights() {
    if (!chsWeights_) {
        glCreateBuffers(1, &chsWeights_);
        glNamedBufferStorage(chsWeights_, sizeof(chsKernelData), chsKernelData.data(), 0);
    }
    glBindBufferBase(GL_UNIFORM_BUFFER, 6, chsWeights_);
}
gfx::research::Shader& Renderer::shadowProgram(Method method) {
    const int kernel = smoothingKernel(method, settings_);
    const int slot = method == Method::Pcf ? 0 : method == Method::Pcss ? 1 : method == Method::Geometric ? 2 : 3;
    const bool packed = method == Method::GeometricLayered && settings_.packedIntervals;
    auto& program = packed ? intervalShaders_.at(std::size_t(kernel))
                    : method == Method::Chs       ? chsShaders_.at(std::size_t((settings_.chsKernel - 7) / 2))
                    : supportsSmoothing(method) ? smoothingShaders_.at(std::size_t(slot)).at(std::size_t(kernel))
                                                : shadowShaders_.at(std::size_t(method));
    if (!program.valid()) {
        std::string defines = "#define SHADOW_METHOD " + std::to_string(int(method)) + "\n";
        defines += "#define CHS_FILTER_SIZE " + std::to_string(settings_.chsKernel) + "\n";
        defines += "#define SHADOW_SMOOTHING " + std::to_string(kernel) + "\n";
        if (packed) defines += "#define PACKED_INTERVALS 1\n";
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
void Renderer::maps(Method method, int dropped) {
    if (isAngularRayMethod(method))
        return;
    glBindFramebuffer(GL_FRAMEBUFFER, mapFbo_);
    glViewport(0, 0, settings_.mapResolution, settings_.mapResolution);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    const bool packed = method == Method::GeometricLayered && settings_.packedIntervals;
    if(packed) glDisable(GL_DEPTH_TEST); else glEnable(GL_DEPTH_TEST);
    glDepthFunc(settings_.analyticMaps ? GL_ALWAYS : GL_LESS);
    auto& shader = packed ? (settings_.analyticMaps ? intervalAnalytic_ : intervalRaster_)
                          : (settings_.analyticMaps ? mapAnalytic_ : mapRaster_);
    shader.bind();
    configureShadow(shader, method);
    shader.set_float("uTessellation", settings_.warp > 0 ? float(scene_.isStress() ? settings_.stressTessellation : 32) : 1.0f);
    shader.set_int("uAdaptiveTessellation", scene_.isStress() && settings_.stressAdaptiveTessellation ? 1 : 0);
    shader.set_int("uStressMode", 0);
    shader.set_int("uDroppedGroup", dropped);
    if(packed) {
        glNamedFramebufferTexture(mapFbo_, GL_DEPTH_ATTACHMENT, 0, 0);
        glNamedFramebufferParameteri(mapFbo_, GL_FRAMEBUFFER_DEFAULT_WIDTH, settings_.mapResolution);
        glNamedFramebufferParameteri(mapFbo_, GL_FRAMEBUFFER_DEFAULT_HEIGHT, settings_.mapResolution);
        const unsigned empty=0xffff0000u;
        glClearTexImage(mapTexture_,0,GL_RED_INTEGER,GL_UNSIGNED_INT,&empty);
        glBindImageTexture(0,mapTexture_,0,GL_TRUE,0,GL_READ_WRITE,GL_R32UI);
    } else glNamedFramebufferTextureLayer(mapFbo_, GL_DEPTH_ATTACHMENT, mapTexture_, 0, 0);
    if (glCheckNamedFramebufferStatus(mapFbo_, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw std::runtime_error("Shadow framebuffer incomplete");
    const float clearDepth = 1;
    if(!packed) glClearNamedFramebufferfv(mapFbo_, GL_DEPTH, 0, &clearDepth);
    if (settings_.analyticMaps) {
        bindScene(shader, dropped);
        shader.set_int("uIncludeReceiver", usesPrefilter(method) ? 1 : 0);
        shader.set_vec3("uReceiverNormal", {-std::sin(scene_.receiverTilt), std::cos(scene_.receiverTilt), 0});
        fullscreen_.draw();
    } else {
        glPatchParameteri(GL_PATCH_VERTICES, 3);
        if (usesPrefilter(method)) {
            shader.set_mat4("uModel", glm::rotate(glm::mat4(1), scene_.receiverTilt, glm::vec3(0, 0, 1)));
            glBindVertexArray(plane_.vao());
            glDrawElements(GL_PATCHES, GLsizei(plane_.index_count()), GL_UNSIGNED_INT, nullptr);
        }
        if (scene_.isStress()) {
            shader.set_int("uStressMode", 1);
            shader.set_int("uStressGrid", stressGrid(scene_.stressObjects));
            glBindVertexArray(cube_.vao());
            if (scene_.stressObjects > 0)
                glDrawElementsInstanced(GL_PATCHES, GLsizei(cube_.index_count()), GL_UNSIGNED_INT, nullptr, scene_.stressObjects);
        } else for (const auto& b : scene_.boxes) {
            if (b.group == dropped)
                continue;
            shader.set_mat4("uModel", boxModel(b));
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
        shadowShader.set_int("uSampleSeed", settings_.sampleSeed);
        shadowShader.set_int("uSharedSamples", settings_.sharedSamplePattern ? 1 : 0);
        shadowShader.set_int("uInnerPenumbra", settings_.innerPenumbra ? 1 : 0);
        shadowShader.set_float("uLayerSeparation", settings_.layerSeparation);
        shadowShader.set_float("uLayerMinFilterTexels", settings_.layerMinFilterTexels);
        if (usesPrefilter(method)) {
            glBindTextureUnit(3, basisTextures_[0]);
            shadowShader.set_float("uVsmMinVariance", settings_.vsmMinVariance);
            shadowShader.set_float("uBleedReduction", settings_.bleedReduction);
            shadowShader.set_float("uEsmExponent", settings_.esmExponent);
            shadowShader.set_float("uCsmBias", settings_.csmBias);
            shadowShader.set_int("uCsmTerms", settings_.csmTerms);
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
    const bool packed = method == Method::GeometricLayered && settings_.packedIntervals;
    if(!isAngularRayMethod(method)) allocateMap(packed);
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
    image.visibility = readTexture(result_.color(), image.width, image.height, GL_RED, 1);
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
    glBindTextureUnit(0, result_.color());
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
    glCopyImageSubData(result_.color(), GL_TEXTURE_2D, 0, 0, 0, 0, texture, GL_TEXTURE_2D, 0, 0, 0, 0, settings_.width,
                       settings_.height, 1);
    return texture;
}
unsigned Renderer::resultTexture() const {
    return result_.color();
}
} // namespace shadows
