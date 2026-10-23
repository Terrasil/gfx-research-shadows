#include <GLFW/glfw3.h>
#include <gfx/research/application.hpp>
#include <glad/gl.h>
#include <imgui.h>
#include <iostream>
#include <type_traits>
#include <shadows/camera.hpp>
#include <shadows/experiment.hpp>

namespace {
template <typename T>
bool sliderSteps(const char* label, T* value, T minimum, T maximum, T step,
                 const char* format, ImGuiSliderFlags flags = 0) {
    ImGui::PushID(label);
    ImGui::BeginGroup();
    ImGui::TextUnformatted(label);
    const float buttonWidth = ImGui::GetFrameHeight();
    const float sliderWidth = std::max(1.0f, ImGui::GetContentRegionAvail().x -
                                               2.0f * (buttonWidth + ImGui::GetStyle().ItemSpacing.x));
    const auto button = [&](const char* text, T delta, bool disabled) {
        ImGui::BeginDisabled(disabled);
        ImGui::PushItemFlag(ImGuiItemFlags_ButtonRepeat, true);
        const bool pressed = ImGui::Button(text, ImVec2(buttonWidth, 0));
        ImGui::PopItemFlag();
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("Step: %.8g. Hold to repeat.", double(step));
        if (!pressed)
            return false;
        const T next = T(std::clamp(double(*value) + double(delta), double(minimum), double(maximum)));
        const bool changed = next != *value;
        *value = next;
        return changed;
    };
    bool changed = button("-", -step, *value <= minimum);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(sliderWidth);
    constexpr ImGuiDataType dataType = std::is_same_v<T, int> ? ImGuiDataType_S32 : ImGuiDataType_Float;
    changed |= ImGui::SliderScalar("##value", dataType, value, &minimum, &maximum, format,
                                  flags | ImGuiSliderFlags_AlwaysClamp);
    ImGui::SetItemTooltip("Ctrl+click the value to type it directly.");
    ImGui::SameLine();
    changed |= button("+", step, *value >= maximum);
    ImGui::EndGroup();
    ImGui::PopID();
    return changed;
}

bool sliderIntSteps(const char* label, int* value, int minimum, int maximum, int step = 1) {
    return sliderSteps(label, value, minimum, maximum, step, "%d");
}

bool sliderFloatSteps(const char* label, float* value, float minimum, float maximum, float step = 1.0f,
                      const char* format = "%.2f", ImGuiSliderFlags flags = 0) {
    // For scientific notation, one click adjusts the last displayed mantissa digit.
    if (flags & ImGuiSliderFlags_Logarithmic)
        step *= std::pow(10.0f, std::floor(std::log10(std::max(std::abs(*value), minimum))));
    return sliderSteps(label, value, minimum, maximum, step, format, flags);
}

class ShadowsApplication final : public gfx::research::Application {
  public:
    explicit ShadowsApplication(shadows::Options options)
        : Application(config(options)), options_(std::move(options)) {}
    int status() const {
        return status_;
    }

  private:
    static gfx::research::AppConfig config(const shadows::Options& o) {
        gfx::research::AppConfig c;
        c.title = "gfx-research-shadows";
        c.width = o.benchmark ? 640 : o.settings.width;
        c.height = o.benchmark ? 360 : o.settings.height;
        c.vsync = false;
        return c;
    }
    bool on_init() override {
        try {
            renderer_.initialize();
            method_ =
                options_.method == "all" ? int(shadows::Method::Geometric) : int(shadows::parseMethod(options_.method));
            const auto ids = shadows::sceneIds();
            for (std::size_t i = 0; i < ids.size(); ++i)
                if (ids[i] == options_.scene)
                    scene_ = int(i);
            pendingObjects_ = options_.settings.stressObjects;
            if (options_.settings.customCamera)
                camera_.set(options_.settings);
            else
                camera_.reset(shadows::makeScene(ids[std::size_t(scene_)], options_.settings.stressObjects), options_.settings.camera);
            cameraScene_ = scene_;
            cameraPreset_ = options_.settings.camera;
            if (options_.headless)
                glfwHideWindow(window());
            return true;
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            return false;
        }
    }
    void on_frame(const gfx::research::FrameInfo& frame) override {
        if (options_.benchmark) {
            try {
                shadows::runExperiment(renderer_, options_);
            } catch (const std::exception& e) {
                std::cerr << "Experiment failed: " << e.what() << '\n';
                status_ = 3;
            }
            close();
            return;
        }
        try {
            updateCamera(frame);
            auto settings = options_.settings;
            camera_.apply(settings);
            settings.width = frame.width;
            settings.height = frame.height;
            const auto scene = shadows::makeScene(shadows::sceneIds().at(std::size_t(scene_)), settings.stressObjects);
            if (scene.isStress() && scene.casterCount() > shadows::maxCasters) {
                settings.analyticMaps = options_.settings.analyticMaps = false;
            }
            gbufferMilliseconds_ = renderer_.prepare(scene, settings, true);
            auto t = renderer_.render(shadows::Method(method_), -1, true);
            milliseconds_ = t.mapMs + t.shadowMs;
            mapMilliseconds_ = t.mapMs;
            shadowMilliseconds_ = t.shadowMs;
            gfx::research::Framebuffer::bind_default();
            const int renderView = view_ >= 2 ? view_ + 1 : view_;
            renderer_.display(frame.width, frame.height, renderView);
            if (capture_) {
                std::filesystem::create_directories(options_.output);
                const auto path = options_.output /
                                  (std::string("interactive-") + shadows::methodId(shadows::Method(method_)) + ".png");
                renderer_.capture(path, renderView);
                std::cout << "Saved " << path.string() << '\n' << std::flush;
                capture_ = false;
                gfx::research::Framebuffer::bind_default();
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            status_ = 3;
            close();
        }
    }
    void on_gui(const gfx::research::FrameInfo&) override {
        if (options_.benchmark)
            return;
        ImGui::SetNextWindowSize(ImVec2(430, 640), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSizeConstraints(ImVec2(440, 400), ImVec2(1000, 1400));
        ImGui::Begin("Shadow research");
        ImGui::TextUnformatted("Method");
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::BeginCombo("##Method", shadows::methodLabel(shadows::Method(method_)))) {
            for (int i = 0; i < shadows::methodCount; ++i) {
                if (ImGui::Selectable(shadows::methodLabel(shadows::Method(i)), i == method_))
                    method_ = i;
            }
            ImGui::EndCombo();
        }
        const auto ids = shadows::sceneIds();
        if (ImGui::BeginCombo("Scene", ids.at(std::size_t(scene_)).c_str())) {
            for (std::size_t i = 0; i < ids.size(); ++i) {
                if (ImGui::Selectable(ids[i].c_str(), int(i) == scene_)) scene_ = int(i);
            }
            ImGui::EndCombo();
        }
        const bool stress = ids[std::size_t(scene_)] == "dense-stress";
        if (stress) {
            sliderSteps<int>("Requested objects", &pendingObjects_, 0, shadows::maxStressCasters, 1, "%d", ImGuiSliderFlags_Logarithmic);
            const int counts[] = {16, 1000, 10000, 100000, 1000000, 2000000, 10000000};
            const char* names[] = {"16", "1K", "10K", "100K", "1M", "2M", "10M"};
            for (int i = 0; i < 7; ++i) {
                if (i) ImGui::SameLine();
                if (ImGui::Button(names[i])) pendingObjects_ = counts[i];
            }
            if (ImGui::Button("Apply object count")) options_.settings.stressObjects = pendingObjects_;
            ImGui::Text("Submitted: %d casters, %llu base triangles", options_.settings.stressObjects,
                        12ull * static_cast<unsigned long long>(options_.settings.stressObjects));
            ImGui::TextWrapped("One instanced cube draw per geometry pass. Fixed 12x12 field: increasing count makes boxes thinner. No per-object CPU culling.");
            ImGui::Checkbox("Adaptive stress tessellation", &options_.settings.stressAdaptiveTessellation);
            sliderIntSteps("Stress tessellation limit", &options_.settings.stressTessellation, 1, 32);
            ImGui::TextWrapped("Both ARR methods test the full procedural population. Cost grows with object count and query budget. Analytic diagnostic maps remain limited to 32 objects.");
        }
        const auto method = shadows::Method(method_);
        const char* views[] = {"Shaded", "Visibility", "Filter radius / support", "Filter parameter"};
        const int viewsCount =
            method == shadows::Method::Hard || method == shadows::Method::Pcf || shadows::isAngularRayMethod(method) || shadows::usesPrefilter(method)
                ? 2
                : 4;
        if (view_ >= viewsCount)
            view_ = 0;
        ImGui::Combo("View", &view_, views, viewsCount);
        sliderFloatSteps("Sun angular radius (degrees)", &options_.settings.angle, 0.1f, 5.0f, 0.1f);
        sliderFloatSteps("NDC warp strength", &options_.settings.warp, 0, 16);
        if(shadows::isAngularRayMethod(method)){
            if(method==shadows::Method::Reference){
                int samples=shadows::angularSamples(method,options_.settings);
                if(sliderIntSteps("ARR directions",&samples,1,16384))options_.settings.arrSamples=samples;
                ImGui::TextWrapped("Deterministic low-discrepancy disk: radical-inverse radius and golden-angle azimuth. One evaluation pass.");
            }else{
                sliderIntSteps("Triangulation directions",&options_.settings.triSamples,3,65);
                options_.settings.triSamples=std::clamp(options_.settings.triSamples,3,65);
                sliderFloatSteps("Triangle sample radius",&options_.settings.triRadius,0.1f,1.0f,0.05f);
                sliderFloatSteps("Triangle rotation (degrees)",&options_.settings.triRotation,-180,180);
                const char* results[]={"Ray masks (weighted average)","Signed boundary reconstruction","Binary boundary reconstruction"};
                int result=options_.settings.triResultMasks?0:options_.settings.triSignedMargin?1:2;
                if(ImGui::Combo("Triangulation result",&result,results,3)){
                    options_.settings.triResultMasks=result==0;
                    options_.settings.triSignedMargin=result!=2;
                }
                if(options_.settings.triResultMasks)
                    ImGui::TextWrapped("Blend the sampled hard-shadow masks, then smooth the resulting visibility. No boundary reconstruction.");
                else
                    ImGui::TextWrapped("3 = triangle plane; 4 = triangle + center; more = radial triangle fan. Reconstructs visibility over the full circular disk.");
            }
            const char* labels[]={"Off / point","Bilinear visibility","Hermite visibility","B-spline visibility 4x4","Bicubic Catmull-Rom 4x4"};
            const int modes[]={0,1,3,4,5};
            int selected=0;for(int i=0;i<5;++i)if(options_.settings.arrSmoothing==modes[i])selected=i;
            if(ImGui::Combo("Result smoothing",&selected,labels,5))options_.settings.arrSmoothing=modes[selected];
            if(options_.settings.arrSmoothing!=0){
                sliderFloatSteps("ARR smoothing width (pixels)",&options_.settings.arrSmoothingRadius,0,3,0.05f);
                ImGui::TextWrapped("Spatial visibility filtering, guarded by receiver depth and normal. Adds one pass; may soften thin shadows.");
            }
        }
        if (shadows::supportsSmoothing(method)) {
            const char* modes[] = {"Original point sampling",        "Bilinear visibility",
                                   "Bilinear visibility + blockers", "Hermite visibility + blockers",
                                   "B-spline visibility 4x4",        "Bicubic Catmull-Rom 4x4",
                                   "Bilinear + random disk",         "Bilinear + stratified random disk"};
            const bool fixed = method == shadows::Method::Pcf;
            const int selected =
                fixed ? shadows::smoothingKernel(method, options_.settings) : options_.settings.smoothing;
            if (fixed) {
                modes[0] = "Point comparisons";
                modes[3] = "Hermite visibility";
            }
            if (ImGui::BeginCombo("Smoothing", modes[selected])) {
                for (int i = 0; i < 8; ++i) {
                    if (fixed && (i == 2 || i >= 6))
                        continue;
                    if (ImGui::Selectable(modes[i], i == selected)) {
                        options_.settings.smoothing = i;
                        if (i >= 6)
                            options_.settings.samplePattern = i - 5;
                    }
                }
                ImGui::EndCombo();
            }
            const int kernel = shadows::smoothingKernel(method, options_.settings);
            if (kernel == 4 || kernel == 5)
                ImGui::TextWrapped("Cubic reconstruction of comparison results. Four gathers per PCF tap.");
            if (method == shadows::Method::Pcss)
                ImGui::TextWrapped("Penumbra radius uses the arithmetic blocker mean.");
        }
        if (method == shadows::Method::Pcss || method == shadows::Method::Geometric || method == shadows::Method::GeometricLayered) {
            const char* extents[] = {"Full penumbra (original)", "Inside hard outline"};
            int extent = options_.settings.innerPenumbra ? 1 : 0;
            if (ImGui::Combo("Penumbra extent", &extent, extents, 2))
                options_.settings.innerPenumbra = extent != 0;
            if (options_.settings.innerPenumbra)
                ImGui::TextWrapped("Penumbra fades inward from the central hard-shadow outline.");
            sliderIntSteps("Filter samples", &options_.settings.filterSamples, 1, 256);
            sliderIntSteps("Blocker samples", &options_.settings.blockerSamples, 1, 256);
            const char* patterns[] = {"Vogel", "Random disk", "Stratified random disk"};
            int pattern = shadows::effectiveSamplePattern(method, options_.settings);
            if (ImGui::Combo("PCSS sample pattern", &pattern, patterns, 3)) {
                // Editing a preset's pattern keeps its bilinear reconstruction.
                if (options_.settings.smoothing >= 6)
                    options_.settings.smoothing = 1;
                options_.settings.samplePattern = pattern;
            }
            if (shadows::effectiveSamplePattern(method, options_.settings) != 0) {
                ImGui::InputInt("Sample seed", &options_.settings.sampleSeed);
                options_.settings.sampleSeed = std::max(0, options_.settings.sampleSeed);
                if (ImGui::Checkbox("Share random sample pattern", &options_.settings.sharedSamplePattern) && options_.settings.sharedSamplePattern)
                    options_.settings.rotateSamples = false;
                ImGui::TextWrapped(options_.settings.sharedSamplePattern
                    ? "Shared random disk: coherent integration, less grain. Blocker/filter streams remain independent. Keep rotation off."
                    : "Repeatable per-pixel jitter. Independent blocker and filter samples; may show spatial grain.");
            }
        }
        if (method == shadows::Method::GeometricLayered) {
            if (ImGui::Button("Apply soft packed preset"))
                shadows::applySoftPackedPreset(options_.settings);
            ImGui::SetItemTooltip("Packed min/max + Hermite visibility/blockers + 64 shared stratified filter and blocker samples + 1 texel minimum radius. Per-pixel rotation off.");
            ImGui::Checkbox("Packed min/max depth (R32UI)", &options_.settings.packedIntervals);
            if(options_.settings.packedIntervals)
                ImGui::TextWrapped("One R32UI map, two UNORM16 surface endpoints. Keeps some occluded blockers; loses intermediate depths.");
            sliderFloatSteps("Depth band separation", &options_.settings.layerSeparation, 1.5f, 8.0f, 0.1f);
            sliderFloatSteps("Minimum filter radius (texels)", &options_.settings.layerMinFilterTexels, 0, 4, 0.25f);
            ImGui::TextWrapped("Minimum spatial softness. Zero keeps physical radii; larger values also soften contact shadows.");
            ImGui::TextWrapped("Separate near/far radii; paired angular samples combine blockers before averaging.");
        }
        if (shadows::usesPrefilter(method)) {
            sliderIntSteps("Gaussian radius (texels)", &options_.settings.prefilterRadius, 0, 16);
            if (method == shadows::Method::Vsm) {
                sliderFloatSteps("Minimum variance", &options_.settings.vsmMinVariance, 1e-9f, 1e-3f, 0.01f, "%.2e", ImGuiSliderFlags_Logarithmic);
                sliderFloatSteps("Light bleeding reduction", &options_.settings.bleedReduction, 0, 0.95f, 0.01f);
            }
            if (method == shadows::Method::Csm) {
                sliderIntSteps("Fourier harmonics", &options_.settings.csmTerms, 4, 16, 2);
                options_.settings.csmTerms = (options_.settings.csmTerms / 2) * 2;
                sliderFloatSteps("CSM receiver offset", &options_.settings.csmBias, 0, 0.05f, 0.001f, "%.3f");
            }
            if (method == shadows::Method::Esm) {
                sliderFloatSteps("ESM exponent", &options_.settings.esmExponent, 1, 80);
                ImGui::Checkbox("ESM threshold fallback", &options_.settings.esmFallback);
            }
            ImGui::TextWrapped("Fixed Gaussian support in shadow texels. Encoding + two blur passes; no contact-hardening radius.");
        }
        if (method == shadows::Method::Pcf)
            sliderFloatSteps("Fixed PCF radius (texels)", &options_.settings.fixedPcfRadius, 0, 8, 1.0f, "%.0f");
        if (method == shadows::Method::Dpcf) {
            sliderIntSteps("Poisson taps", &options_.settings.dpcfTaps, 1, 64);
            sliderFloatSteps("DPCF support at 1.5 degrees (texels)", &options_.settings.dpcfRadius, 0.1f, 64, 0.1f);
            sliderFloatSteps("Penumbra ratio scale", &options_.settings.dpcfRatioScale, 1, 100);
            ImGui::TextWrapped(
                "Full directional DPCF filter from Filament v1.23.1. Original Poisson disk and hardened cubic kernel.");
        }
        if (method == shadows::Method::Chs) {
            const char* sizes[] = {"7", "9", "11", "13", "15"};
            int size = (options_.settings.chsKernel - 7) / 2;
            if (ImGui::Combo("CHS kernel", &size, sizes, 5))
                options_.settings.chsKernel = 7 + 2 * size;
            sliderFloatSteps("AMD sun size at 1.5 degrees", &options_.settings.chsSunSize, 0.1f, 32, 0.1f);
            ImGui::TextWrapped("AMD ShadowFX contactFixedGather4: original blocker search, weight tables and bilinear "
                               "reconstruction.");
        }
        ImGui::BeginDisabled(stress && options_.settings.stressObjects > shadows::maxCasters);
        ImGui::Checkbox("Analytic diagnostic maps", &options_.settings.analyticMaps);
        ImGui::EndDisabled();
        ImGui::Checkbox("Rotate sample pattern", &options_.settings.rotateSamples);
        sliderIntSteps("Camera preset", &options_.settings.camera, 0, 1);
        if (ImGui::Button("Reset camera (R)"))
            cameraScene_ = -1;
        ImGui::TextWrapped(
            "RMB: orbit. MMB: pan. Wheel: zoom. WASD: move, Q/E: down/up, Shift: faster. P: save image.");
        ImGui::Text("G-buffer GPU: %.3f ms", gbufferMilliseconds_);
        ImGui::Text("Map GPU: %.3f ms / shadow GPU: %.3f ms", mapMilliseconds_, shadowMilliseconds_);
        ImGui::Text("GPU components: %.3f ms / frame rate: %.1f FPS", gbufferMilliseconds_ + milliseconds_, ImGui::GetIO().Framerate);
        if (ImGui::Button("Save clean image"))
            capture_ = true;
        ImGui::End();
    }
    void updateCamera(const gfx::research::FrameInfo& frame) {
        if (cameraScene_ != scene_ || cameraPreset_ != options_.settings.camera) {
            camera_.reset(shadows::makeScene(shadows::sceneIds()[std::size_t(scene_)], options_.settings.stressObjects), options_.settings.camera);
            cameraScene_ = scene_;
            cameraPreset_ = options_.settings.camera;
        }
        const auto& io = ImGui::GetIO();
        if (!io.WantCaptureMouse) {
            if (ImGui::IsMouseDown(ImGuiMouseButton_Right))
                camera_.orbit(io.MouseDelta.x, io.MouseDelta.y);
            if (ImGui::IsMouseDown(ImGuiMouseButton_Middle))
                camera_.pan(io.MouseDelta.x, io.MouseDelta.y, float(frame.height));
            camera_.zoom(io.MouseWheel);
        }
        if (!io.WantCaptureKeyboard) {
            const auto key = [](ImGuiKey k) { return ImGui::IsKeyDown(k) ? 1.0f : 0.0f; };
            camera_.move(key(ImGuiKey_D) - key(ImGuiKey_A), key(ImGuiKey_W) - key(ImGuiKey_S),
                         key(ImGuiKey_E) - key(ImGuiKey_Q), frame.delta_seconds,
                         ImGui::IsKeyDown(ImGuiKey_LeftShift) || ImGui::IsKeyDown(ImGuiKey_RightShift));
            if (ImGui::IsKeyPressed(ImGuiKey_R))
                cameraScene_ = -1;
            if (ImGui::IsKeyPressed(ImGuiKey_P))
                capture_ = true;
        }
    }
    shadows::Options options_;
    shadows::Renderer renderer_;
    shadows::CameraController camera_;
    int cameraScene_ = -1, cameraPreset_ = -1;
    int method_ = int(shadows::Method::Geometric), scene_ = 2, view_ = 0, status_ = 0;
    int pendingObjects_ = 4096;
    bool capture_ = false;
    double milliseconds_ = 0;
    double gbufferMilliseconds_ = 0, mapMilliseconds_ = 0, shadowMilliseconds_ = 0;
};
} // namespace
int main(int argc, char** argv) {
    try {
        auto options = shadows::parseOptions(argc, argv);
#ifdef GFX_SHADOWS_HAS_EGL_HEADLESS
        if (options.headless)
            return shadows::runHeadlessEgl(options);
#endif
        glfwSetErrorCallback(
            [](int code, const char* description) { std::cerr << "GLFW " << code << ": " << description << '\n'; });
        ShadowsApplication app(std::move(options));
        int code = app.run();
        return code ? code : app.status();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
