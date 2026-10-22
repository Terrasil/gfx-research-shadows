#include <fstream>
#include <gfx/research/orbit_camera.hpp>
#include <gfx/research/primitives.hpp>
#include <gfx/research/screenshot.hpp>
#include <glad/gl.h>
#include <shadows/renderer.hpp>
#include <sstream>
#include <stdexcept>

namespace shadows {
namespace {
constexpr std::array<const char*, methodCount> ids = {"hard",     "pcf",       "pcss",          "pcss-min-gap",
                                                      "ndc-pcss", "split-min", "split-product", "binned-min",
                                                      "ray16",    "reference"};
constexpr std::array<const char*, methodCount> labels = {"Hard shadow",
                                                         "Fixed PCF",
                                                         "PCSS mean blocker",
                                                         "PCSS minimum gap",
                                                         "NDC + PCSS",
                                                         "Two caster groups + min",
                                                         "Two caster groups + product",
                                                         "Single-map depth bins + min (experimental)",
                                                         "16 angular rays (box scene)",
                                                         "Multipass angular reference"};
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
                std::initializer_list<std::pair<gfx::research::ShaderStage, const char*>> names) {
    std::vector<gfx::research::ShaderSource> sources;
    for (auto [stage, name] : names)
        sources.push_back({stage, loadSource(std::filesystem::path(GFX_SHADOWS_SOURCE_DIR) / "shaders" / name)});
    if (!target.load(sources))
        throw std::runtime_error(target.last_error());
}
bool split(Method m) {
    return m == Method::SplitMin || m == Method::SplitProduct;
}
bool ray(Method m) {
    return m == Method::Reference || m == Method::Ray16;
}
bool warped(Method m) {
    return int(m) >= int(Method::NdcPcss) && int(m) <= int(Method::BinnedMin);
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
Renderer::Renderer() = default;
Renderer::~Renderer() {
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
    loadShader(shadowShader_, {{S::Vertex, "fullscreen.vert"}, {S::Fragment, "shadow.frag"}});
    loadShader(referenceShader_, {{S::Vertex, "fullscreen.vert"}, {S::Fragment, "reference.frag"}});
    loadShader(resolveShader_, {{S::Vertex, "fullscreen.vert"}, {S::Fragment, "resolve.frag"}});
    loadShader(displayShader_, {{S::Vertex, "fullscreen.vert"}, {S::Fragment, "display.frag"}});
    cube_ = gfx::research::make_cube(1);
    plane_ = gfx::research::make_plane(30, 1);
    glCreateFramebuffers(1, &mapFbo_);
}
void Renderer::allocate() {
    const auto& s = settings_;
    if (s.width != allocatedWidth_ || s.height != allocatedHeight_) {
        using A = gfx::research::ColorAttachmentDesc;
        const A g[] = {{GL_RGBA8, GL_NEAREST, GL_NEAREST}, {GL_RGBA32F, GL_NEAREST, GL_NEAREST}};
        const A r[] = {{GL_R32F, GL_NEAREST, GL_NEAREST}};
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
        glTextureStorage3D(mapTexture_, 1, GL_DEPTH_COMPONENT32F, s.mapResolution, s.mapResolution, 2);
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
void Renderer::prepare(const Scene& scene, const Settings& settings) {
    if (scene.boxes.size() > maxCasters)
        throw std::runtime_error("Too many casters");
    scene_ = scene;
    settings_ = settings;
    allocate();
    gfx::research::OrbitCamera camera;
    camera.set_target(scene_.target);
    camera.set_clip(0.1f, 100);
    camera.set_distance(settings_.camera == 0 ? 16.0f : 14.0f);
    camera.set_yaw(-2.1f);
    camera.set_pitch(settings_.camera == 0 ? 0.65f : 1.25f);
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
void Renderer::configureShadow(gfx::research::Shader& shader, Method method) {
    shader.set_mat4("uLightVp", lightVp_);
    shader.set_mat4("uInverseLightVp", inverseLightVp_);
    shader.set_float("uWarp", warped(method) ? settings_.warp : 0);
    shader.set_vec2("uFocus", {0, 0});
    shader.set_int("uMapResolution", settings_.mapResolution);
}
void Renderer::maps(Method method, int dropped) {
    if (ray(method))
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
    bindScene(shader, dropped);
    shader.set_float("uTessellation", warped(method) && settings_.warp > 0 ? 32.0f : 1.0f);
    const int layers = split(method) ? 2 : 1;
    const float clearDepth = 1;
    for (int layer = 0; layer < layers; ++layer) {
        glNamedFramebufferTextureLayer(mapFbo_, GL_DEPTH_ATTACHMENT, mapTexture_, 0, layer);
        if (glCheckNamedFramebufferStatus(mapFbo_, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            throw std::runtime_error("Shadow framebuffer incomplete");
        glClearNamedFramebufferfv(mapFbo_, GL_DEPTH, 0, &clearDepth);
        const int group = split(method) ? layer : -1;
        if (settings_.analyticMaps) {
            shader.set_int("uGroup", group);
            fullscreen_.draw();
        } else {
            glPatchParameteri(GL_PATCH_VERTICES, 3);
            for (const auto& b : scene_.boxes) {
                if (b.group == dropped || (group >= 0 && b.group != group))
                    continue;
                shader.set_mat4("uModel", boxModel(b));
                glBindVertexArray(cube_.vao());
                glDrawElements(GL_PATCHES, GLsizei(cube_.index_count()), GL_UNSIGNED_INT, nullptr);
            }
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
        resolveShader_.bind();
        glBindTextureUnit(0, accum_.color());
        fullscreen_.draw();
    } else {
        result_.bind();
        shadowShader_.bind();
        bindScreen(shadowShader_);
        bindScene(shadowShader_, dropped);
        configureShadow(shadowShader_, method);
        glBindTextureUnit(2, mapTexture_);
        shadowShader_.set_int("uMethod", int(method));
        shadowShader_.set_int("uFilterSamples", settings_.filterSamples);
        shadowShader_.set_int("uBlockerSamples", settings_.blockerSamples);
        shadowShader_.set_int("uRotateSamples", settings_.rotateSamples ? 1 : 0);
        shadowShader_.set_vec3("uSun", glm::vec3(sunDirection()));
        shadowShader_.set_float("uTanAngle", std::tan(glm::radians(settings_.angle)));
        shadowShader_.set_float("uDepthRange", 80);
        shadowShader_.set_float("uExtent", 14);
        shadowShader_.set_float("uDepthBias", settings_.depthBias);
        shadowShader_.set_float("uMaxRadiusWorld", settings_.maxRadiusWorld);
        shadowShader_.set_float("uFixedPcfRadius", settings_.fixedPcfRadius);
        fullscreen_.draw();
    }
}
Timing Renderer::render(Method method, int dropped, bool timed) {
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
void Renderer::display(int width, int height, int view, unsigned reference) {
    glViewport(0, 0, width, height);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_FRAMEBUFFER_SRGB);
    displayShader_.bind();
    displayShader_.set_int("uView", view);
    displayShader_.set_vec3("uSun", glm::vec3(sunDirection()));
    glBindTextureUnit(0, result_.color());
    glBindTextureUnit(1, gbuffer_.color(1));
    glBindTextureUnit(2, gbuffer_.color());
    glBindTextureUnit(3, reference ? reference : result_.color());
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
