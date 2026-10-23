#include <GLFW/glfw3.h>
#include <gfx/research/application.hpp>
#include <glad/gl.h>
#include <imgui.h>
#include <iostream>
#include <shadows/camera.hpp>
#include <shadows/experiment.hpp>

namespace {
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
            if (options_.settings.customCamera)
                camera_.set(options_.settings);
            else
                camera_.reset(shadows::makeScene(ids[std::size_t(scene_)]), options_.settings.camera);
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
            renderer_.prepare(shadows::makeScene(shadows::sceneIds().at(std::size_t(scene_))), settings);
            auto t = renderer_.render(shadows::Method(method_), -1, true);
            milliseconds_ = t.mapMs + t.shadowMs;
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
            for (int i = 0; i < shadows::methodCount; ++i)
                if (ImGui::Selectable(shadows::methodLabel(shadows::Method(i)), i == method_))
                    method_ = i;
            ImGui::EndCombo();
        }
        const auto ids = shadows::sceneIds();
        if (ImGui::BeginCombo("Scene", ids.at(std::size_t(scene_)).c_str())) {
            for (std::size_t i = 0; i < ids.size(); ++i) {
                if (ImGui::Selectable(ids[i].c_str(), int(i) == scene_))
                    scene_ = int(i);
            }
            ImGui::EndCombo();
        }
        const auto method = shadows::Method(method_);
        const char* views[] = {"Shaded", "Visibility", "Filter radius / support", "Filter parameter"};
        const int viewsCount =
            method == shadows::Method::Hard || method == shadows::Method::Pcf || method == shadows::Method::Reference
                ? 2
                : 4;
        if (view_ >= viewsCount)
            view_ = 0;
        ImGui::Combo("View", &view_, views, viewsCount);
        ImGui::SliderFloat("Sun angular radius (degrees)", &options_.settings.angle, 0.1f, 5.0f);
        ImGui::SliderFloat("NDC warp strength", &options_.settings.warp, 0, 16);
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
        if (method == shadows::Method::Pcss || method == shadows::Method::Geometric) {
            ImGui::SliderInt("Filter samples", &options_.settings.filterSamples, 1, 256);
            ImGui::SliderInt("Blocker samples", &options_.settings.blockerSamples, 1, 256);
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
                ImGui::TextWrapped("Repeatable per-pixel jitter. Independent blocker and filter samples.");
            }
        }
        if (method == shadows::Method::Pcf)
            ImGui::SliderFloat("Fixed PCF radius (texels)", &options_.settings.fixedPcfRadius, 0, 8, "%.0f");
        if (method == shadows::Method::Dpcf) {
            ImGui::SliderInt("Poisson taps", &options_.settings.dpcfTaps, 1, 64);
            ImGui::SliderFloat("DPCF support at 1.5 degrees (texels)", &options_.settings.dpcfRadius, 0.1f, 64);
            ImGui::SliderFloat("Penumbra ratio scale", &options_.settings.dpcfRatioScale, 1, 100);
            ImGui::TextWrapped(
                "Full directional DPCF filter from Filament v1.23.1. Original Poisson disk and hardened cubic kernel.");
        }
        if (method == shadows::Method::Chs) {
            const char* sizes[] = {"7", "9", "11", "13", "15"};
            int size = (options_.settings.chsKernel - 7) / 2;
            if (ImGui::Combo("CHS kernel", &size, sizes, 5))
                options_.settings.chsKernel = 7 + 2 * size;
            ImGui::SliderFloat("AMD sun size at 1.5 degrees", &options_.settings.chsSunSize, 0.1f, 32);
            ImGui::TextWrapped("AMD ShadowFX contactFixedGather4: original blocker search, weight tables and bilinear "
                               "reconstruction.");
        }
        ImGui::SliderInt("ARR reference directions", &options_.settings.referenceSamples, 16, 4096);
        ImGui::Checkbox("Analytic diagnostic maps", &options_.settings.analyticMaps);
        ImGui::Checkbox("Rotate sample pattern", &options_.settings.rotateSamples);
        ImGui::SliderInt("Camera preset", &options_.settings.camera, 0, 1);
        if (ImGui::Button("Reset camera (R)"))
            cameraScene_ = -1;
        ImGui::TextWrapped(
            "RMB: orbit. MMB: pan. Wheel: zoom. WASD: move, Q/E: down/up, Shift: faster. P: save image.");
        ImGui::Text("Map + shadow GPU: %.3f ms", milliseconds_);
        if (ImGui::Button("Save clean image"))
            capture_ = true;
        ImGui::End();
    }
    void updateCamera(const gfx::research::FrameInfo& frame) {
        if (cameraScene_ != scene_ || cameraPreset_ != options_.settings.camera) {
            camera_.reset(shadows::makeScene(shadows::sceneIds()[std::size_t(scene_)]), options_.settings.camera);
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
    bool capture_ = false;
    double milliseconds_ = 0;
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
