#pragma once
#include "scene.hpp"
#include <cstdint>
#include <filesystem>
#include <gfx/research/framebuffer.hpp>
#include <gfx/research/fullscreen_triangle.hpp>
#include <gfx/research/mesh.hpp>
#include <gfx/research/shader.hpp>

namespace shadows {
enum class Method : int { Hard, Pcf, Pcss, Reference, Geometric, Dpcf, Chs };
constexpr int methodCount = 7;
const char* methodId(Method);
const char* methodLabel(Method);
Method parseMethod(const std::string&);
const char* samplePatternId(int);
int parseSamplePattern(const std::string&);
struct Settings {
    int width = 1280, height = 720, mapResolution = 1024;
    int filterSamples = 48, blockerSamples = 48, referenceSamples = 512;
    int dpcfTaps = 12, chsKernel = 11;
    int smoothing = 1;
    int samplePattern = 0, sampleSeed = 0;
    float angle = 1.5f, warp = 8.0f, depthBias = 0.00015f, normalBias = 0.002f;
    float maxRadiusWorld = 3.0f, fixedPcfRadius = 2.0f;
    float dpcfRadius = 6.0f, dpcfRatioScale = 1.0f, chsSunSize = 4.0f;
    bool analyticMaps = false, rotateSamples = false;
    int camera = 0;
    bool customCamera = false;
    glm::vec3 cameraTarget{-1.4f, 1.0f, -0.8f};
    float cameraYaw = -2.1f, cameraPitch = 0.65f, cameraDistance = 16.0f;
};
bool supportsSmoothing(Method);
int smoothingKernel(Method, const Settings&);
int effectiveSamplePattern(Method, const Settings&);
int activeMapLayers(Method, const Settings&);
std::uint64_t activeMapBytes(Method, const Settings&);
int mapGenerationPasses(Method, const Settings&);
struct Timing {
    double mapMs = 0, shadowMs = 0;
};
struct Image {
    int width = 0, height = 0;
    std::vector<float> visibility, receiver, depth;
};
class Renderer {
  public:
    Renderer();
    ~Renderer();
    void initialize();
    void prepare(const Scene&, const Settings&);
    Timing render(Method, int droppedGroup = -1, bool timed = false);
    Image read() const;
    std::vector<float> positions() const;
    std::vector<float> normals() const;
    std::vector<float> diagnostics() const;
    void display(int width, int height, int view = 0, unsigned reference = 0);
    void capture(const std::filesystem::path&, int view = 0, unsigned reference = 0);
    unsigned copyResult() const;
    unsigned resultTexture() const;
    const Settings& settings() const {
        return settings_;
    }
    const Scene& scene() const {
        return scene_;
    }
    glm::mat4 inverseVp() const {
        return inverseVp_;
    }
    void validateFilters(const std::filesystem::path&);
    void validateSampling(const std::filesystem::path&);

  private:
    void maps(Method, int);
    void shadow(Method, int);
    void bindScene(gfx::research::Shader&, int);
    void bindScreen(gfx::research::Shader&);
    void configureShadow(gfx::research::Shader&, Method);
    void allocate();
    gfx::research::Shader& shadowProgram(Method);
    void allocateChsWeights();
    Scene scene_;
    Settings settings_;
    glm::mat4 inverseVp_{1}, lightVp_{1}, inverseLightVp_{1};
    gfx::research::Mesh cube_, plane_;
    gfx::research::FullscreenTriangle fullscreen_;
    gfx::research::Shader gbufferShader_, mapRaster_, mapAnalytic_, referenceShader_, resolveShader_, displayShader_;
    std::array<gfx::research::Shader, methodCount> shadowShaders_;
    std::array<gfx::research::Shader, 5> chsShaders_;
    std::array<std::array<gfx::research::Shader, 6>, 3> smoothingShaders_;
    unsigned chsWeights_ = 0;
    gfx::research::Framebuffer gbuffer_, result_, accum_, preview_;
    unsigned mapTexture_ = 0, mapFbo_ = 0;
    Method lastMethod_ = Method::Hard;
    int allocatedWidth_ = 0, allocatedHeight_ = 0, allocatedMap_ = 0;
};
} // namespace shadows
