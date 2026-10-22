#include <GLFW/glfw3.h>
#include <gfx/research/application.hpp>
#include <glad/gl.h>
#include <imgui.h>
#include <iostream>
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
            method_ = options_.method == "all" ? 4 : int(shadows::parseMethod(options_.method));
            const auto ids = shadows::sceneIds();
            for (std::size_t i = 0; i < ids.size(); ++i)
                if (ids[i] == options_.scene)
                    scene_ = int(i);
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
            auto settings = options_.settings;
            settings.width = frame.width;
            settings.height = frame.height;
            renderer_.prepare(shadows::makeScene(shadows::sceneIds().at(std::size_t(scene_))), settings);
            auto t = renderer_.render(shadows::Method(method_), -1, true);
            milliseconds_ = t.mapMs + t.shadowMs;
            gfx::research::Framebuffer::bind_default();
            renderer_.display(frame.width, frame.height, view_);
            if (capture_) {
                std::filesystem::create_directories(options_.output);
                renderer_.capture(options_.output / "interactive.png", view_);
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
        ImGui::Begin("Shadow research");
        if (ImGui::BeginCombo("Method", shadows::methodLabel(shadows::Method(method_)))) {
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
        const char* views[] = {"Shaded", "Visibility"};
        ImGui::Combo("View", &view_, views, 2);
        ImGui::SliderFloat("Sun angular radius (degrees)", &options_.settings.angle, 0.1f, 5.0f);
        ImGui::SliderFloat("NDC warp strength", &options_.settings.warp, 0, 16);
        ImGui::SliderInt("Filter samples", &options_.settings.filterSamples, 8, 128);
        ImGui::SliderInt("Blocker samples", &options_.settings.blockerSamples, 8, 128);
        ImGui::SliderInt("Reference directions", &options_.settings.referenceSamples, 16, 1024);
        ImGui::Checkbox("Analytic diagnostic maps", &options_.settings.analyticMaps);
        ImGui::Checkbox("Rotate sample pattern", &options_.settings.rotateSamples);
        ImGui::SliderInt("Camera preset", &options_.settings.camera, 0, 1);
        ImGui::Text("Map + shadow GPU: %.3f ms", milliseconds_);
        ImGui::TextWrapped("Split methods use two annotated caster groups. Binned min uses one map and is "
                           "experimental. Reference uses exact box geometry.");
        if (ImGui::Button("Save clean image"))
            capture_ = true;
        ImGui::End();
    }
    shadows::Options options_;
    shadows::Renderer renderer_;
    int method_ = 4, scene_ = 2, view_ = 0, status_ = 0;
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
        ShadowsApplication app(std::move(options));
        int code = app.run();
        return code ? code : app.status();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
