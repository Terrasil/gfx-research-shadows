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
constexpr std::array<const char*, methodCount> ids = {"hard", "pcf", "pcss", "arr", "geometric", "dpcf", "chs"};
constexpr std::array<const char*, methodCount> labels = {"Hard shadow",
                                                         "Fixed PCF",
                                                         "PCSS mean blocker",
                                                         "Angular ray reference (exact boxes)",
                                                         "PCSS geometric mean",
                                                         "DPCF (Filament v1.23.1)",
                                                         "CHS (AMD ShadowFX fixed gather4)"};
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
int activeMapLayers(Method m, const Settings&) {
    return m == Method::Reference ? 0 : 1;
}
std::uint64_t activeMapBytes(Method m, const Settings& s) {
    return std::uint64_t(activeMapLayers(m, s)) * s.mapResolution * s.mapResolution * 4u;
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
    return method == Method::Pcf || method == Method::Pcss || method == Method::Geometric;
}
int smoothingKernel(Method method, const Settings& settings) {
    if (!supportsSmoothing(method))
        return 0;
    const int kernel = settings.smoothing >= 6 ? 1 : settings.smoothing;
    return method == Method::Pcf && kernel == 2 ? 1 : kernel;
}
int effectiveSamplePattern(Method method, const Settings& settings) {
    if (method != Method::Pcss && method != Method::Geometric)
        return 0;
    return settings.smoothing >= 6 ? settings.smoothing - 5 : settings.samplePattern;
}
Renderer::Renderer() = default;
Renderer::~Renderer() {
    if (chsWeights_)
        glDeleteBuffers(1, &chsWeights_);
    if (mapTexture_)
        glDeleteTextures(1, &mapTexture_);
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
    loadShader(referenceShader_, {{S::Vertex, "fullscreen.vert"}, {S::Fragment, "reference.frag"}});
    loadShader(resolveShader_, {{S::Vertex, "fullscreen.vert"}, {S::Fragment, "resolve.frag"}});
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
    const int slot = method == Method::Pcf ? 0 : method == Method::Pcss ? 1 : 2;
    auto& program = method == Method::Chs       ? chsShaders_.at(std::size_t((settings_.chsKernel - 7) / 2))
                    : supportsSmoothing(method) ? smoothingShaders_.at(std::size_t(slot)).at(std::size_t(kernel))
                                                : shadowShaders_.at(std::size_t(method));
    if (!program.valid()) {
        std::string defines = "#define SHADOW_METHOD " + std::to_string(int(method)) + "\n";
        defines += "#define CHS_FILTER_SIZE " + std::to_string(settings_.chsKernel) + "\n";
        defines += "#define SHADOW_SMOOTHING " + std::to_string(kernel) + "\n";
        using S = gfx::research::ShaderStage;
        loadShader(program, {{S::Vertex, "fullscreen.vert"}, {S::Fragment, "shadow.frag"}}, defines);
    }
    return program;
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
    if (s.mapResolution != allocatedMap_) {
        if (mapTexture_)
            glDeleteTextures(1, &mapTexture_);
        glCreateTextures(GL_TEXTURE_2D_ARRAY, 1, &mapTexture_);
        glTextureStorage3D(mapTexture_, 1, GL_DEPTH_COMPONENT32F, s.mapResolution, s.mapResolution, 1);
        glTextureParameteri(mapTexture_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(mapTexture_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTextureParameteri(mapTexture_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTextureParameteri(mapTexture_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        const float white[] = {1, 1, 1, 1};
        glTextureParameterfv(mapTexture_, GL_TEXTURE_BORDER_COLOR, white);
        glNamedFramebufferDrawBuffer(mapFbo_, GL_NONE);
        glNamedFramebufferReadBuffer(mapFbo_, GL_NONE);
        allocatedMap_ = s.mapResolution;
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
void Renderer::prepare(const Scene& scene, const Settings& settings) {
    if (scene.boxes.size() > maxCasters)
        throw std::runtime_error("Too many casters");
    scene_ = scene;
    settings_ = settings;
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
    plane_.draw();
    gbufferShader_.set_int("uReceiver", 0);
    for (const auto& b : scene_.boxes) {
        gbufferShader_.set_mat4("uModel", boxModel(b));
        gbufferShader_.set_vec3("uColor", b.color);
        cube_.draw();
    }
    glDisable(GL_DEPTH_TEST);
}
void Renderer::bindScene(gfx::research::Shader& shader, int dropped) {
    shader.set_int("uBoxCount", int(scene_.boxes.size()));
    shader.set_int("uDroppedGroup", dropped);
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
    if (method == Method::Reference)
        return;
    glBindFramebuffer(GL_FRAMEBUFFER, mapFbo_);
    glViewport(0, 0, settings_.mapResolution, settings_.mapResolution);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(settings_.analyticMaps ? GL_ALWAYS : GL_LESS);
    auto& shader = settings_.analyticMaps ? mapAnalytic_ : mapRaster_;
    shader.bind();
    configureShadow(shader, method);
    shader.set_float("uTessellation", settings_.warp > 0 ? 32.0f : 1.0f);
    glNamedFramebufferTextureLayer(mapFbo_, GL_DEPTH_ATTACHMENT, mapTexture_, 0, 0);
    if (glCheckNamedFramebufferStatus(mapFbo_, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw std::runtime_error("Shadow framebuffer incomplete");
    const float clearDepth = 1;
    glClearNamedFramebufferfv(mapFbo_, GL_DEPTH, 0, &clearDepth);
    if (settings_.analyticMaps) {
        bindScene(shader, dropped);
        fullscreen_.draw();
    } else {
        glPatchParameteri(GL_PATCH_VERTICES, 3);
        for (const auto& b : scene_.boxes) {
            if (b.group == dropped)
                continue;
            shader.set_mat4("uModel", boxModel(b));
            glBindVertexArray(cube_.vao());
            glDrawElements(GL_PATCHES, GLsizei(cube_.index_count()), GL_UNSIGNED_INT, nullptr);
        }
    }
    glDisable(GL_DEPTH_TEST);
}
void Renderer::shadow(Method method, int dropped) {
    glViewport(0, 0, settings_.width, settings_.height);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    if (method == Method::Reference) {
        accum_.bind();
        const float zero[] = {0, 0, 0, 0};
        glClearNamedFramebufferfv(accum_.id(), GL_COLOR, 0, zero);
        referenceShader_.bind();
        bindScreen(referenceShader_);
        bindScene(referenceShader_, dropped);
        glEnable(GL_BLEND);
        glBlendEquation(GL_FUNC_ADD);
        glBlendFunc(GL_ONE, GL_ONE);
        for (int i = 0; i < settings_.referenceSamples; ++i) {
            const auto sample = sourceSample(unsigned(i), settings_.angle);
            referenceShader_.set_vec3("uDirection", glm::vec3(sample.direction));
            referenceShader_.set_float("uWeight", float(sample.weight));
            fullscreen_.draw();
        }
        glDisable(GL_BLEND);
        result_.bind();
        const GLenum attachment = GL_COLOR_ATTACHMENT0;
        glNamedFramebufferDrawBuffers(result_.id(), 1, &attachment);
        resolveShader_.bind();
        glBindTextureUnit(0, accum_.color());
        fullscreen_.draw();
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
Timing Renderer::render(Method method, int dropped, bool timed) {
    lastMethod_ = method;
    if (method != Method::Reference)
        (void)shadowProgram(method); // compilation stays outside GPU timing scopes
    Timing result;
    unsigned q[3] = {};
    if (timed) {
        glCreateQueries(GL_TIMESTAMP, 3, q);
        glQueryCounter(q[0], GL_TIMESTAMP);
    }
    maps(method, dropped);
    if (timed)
        glQueryCounter(q[1], GL_TIMESTAMP);
    shadow(method, dropped);
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
        lastMethod_ == Method::Hard || lastMethod_ == Method::Pcf || lastMethod_ == Method::Reference ? 0 : 1);
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
