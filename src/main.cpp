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
                const auto variant=method_==int(shadows::Method::AngularRgba)
                    ?std::string("-")+shadows::multiViewVariantId(options_.settings):std::string{};
                const auto path = options_.output /
                                  (std::string("interactive-") + shadows::methodId(shadows::Method(method_)) + variant + ".png");
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
        const char* groups[]={"Basic","PCSS","Filtered maps","Multiple light view","Geometry reference"};
        const std::array<std::vector<int>,5> order={{{0,1},{2,4,7,5,6},{8,9,10,13},{12},{3,11}}};
        int group=0;
        for(int i=0;i<5;++i)
            if(std::find(order[std::size_t(i)].begin(),order[std::size_t(i)].end(),method_)!=order[std::size_t(i)].end())group=i;
        if(ImGui::Combo("Group",&group,groups,5))method_=order[std::size_t(group)].front();
        ImGui::SetNextItemWidth(-1.0f);
        if(group==3){
            const int selected=shadows::multiViewVariant(options_.settings);
            const char* preview=selected>=0?shadows::multiViewVariantLabel(selected):
                (options_.settings.rgbaCascades?"Depth array cascaded (diagnostic)":"Depth array (diagnostic)");
            if(ImGui::BeginCombo("##Method",preview)){
                for(int variant=0;variant<shadows::multiViewVariantCount;++variant){
                    if(ImGui::Selectable(shadows::multiViewVariantLabel(variant),variant==selected))
                        shadows::applyMultiViewVariant(options_.settings,variant);
                    if(variant==selected)ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }else if(ImGui::BeginCombo("##Method",shadows::methodLabel(shadows::Method(method_)))){
            for(int i:order[std::size_t(group)]){
                const bool selected=i==method_;
                if(ImGui::Selectable(shadows::methodLabel(shadows::Method(i)),selected))method_=i;
                if(selected)ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        if(ImGui::CollapsingHeader("Scene and population",ImGuiTreeNodeFlags_DefaultOpen)){
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
        }
        const auto ids = shadows::sceneIds();
        const bool stress = ids[std::size_t(scene_)] == "dense-stress";
        const auto method = shadows::Method(method_);
        const char* views[] = {"Shaded", "Visibility", "Filter radius / support", "Filter parameter"};
        const int viewsCount =
            method == shadows::Method::AngularRgba || method == shadows::Method::Hard || method == shadows::Method::Pcf || shadows::isAngularRayMethod(method) || shadows::usesPrefilter(method)
                ? 2
                : 4;
        if (view_ >= viewsCount)
            view_ = 0;
        ImGui::Combo("View", &view_, views, viewsCount);
        if(ImGui::CollapsingHeader("Light and projection",ImGuiTreeNodeFlags_DefaultOpen)){
        sliderFloatSteps("Sun angular radius (degrees)", &options_.settings.angle, 0.1f, 5.0f, 0.1f);
        sliderFloatSteps("NDC warp strength", &options_.settings.warp, 0, 16);
        }
        if(ImGui::CollapsingHeader("Method settings",ImGuiTreeNodeFlags_DefaultOpen)){
        if(method==shadows::Method::AngularRgba){
            if(ImGui::CollapsingHeader("Storage and cost",ImGuiTreeNodeFlags_DefaultOpen)){
                ImGui::TextUnformatted(options_.settings.multiViewStorage==0?"Depth in RGBA color channels (MIN)":
                    options_.settings.multiViewStorage==2?"Separate depth texture for each view":"Depth array (diagnostic)");
                ImGui::TextUnformatted(options_.settings.rgbaCascades?"Cascaded coverage":"Equal coverage in all views");
                if(ImGui::TreeNode("Additional depth-array diagnostic")){
                    const char* storage[]={"RGBA channels","Depth array","Separate textures"};
                    ImGui::Combo("Diagnostic storage",&options_.settings.multiViewStorage,storage,3);
                    ImGui::TreePop();
                }
                ImGui::Text("%d texture objects, %d layers, %.2f MiB active map data",
                    shadows::mapTextureCount(method,options_.settings),shadows::activeMapLayers(method,options_.settings),
                    double(shadows::activeMapBytes(method,options_.settings))/(1024.0*1024.0));
                ImGui::TextUnformatted("4 geometry passes; same views, filtering and mixing.");
                ImGui::TextWrapped("Array versus separate isolates storage layout. RGBA also changes rasterization: color MIN blending instead of a depth attachment. Allocation and shader compilation are outside the GPU timings below.");
            }
            if(ImGui::CollapsingHeader("Light views",ImGuiTreeNodeFlags_DefaultOpen)){
            ImGui::Checkbox("Offset views on sun disk",&options_.settings.rgbaOffset);
            ImGui::BeginDisabled(!options_.settings.rgbaOffset);
            sliderFloatSteps("Ring radius / sun radius",&options_.settings.rgbaRadius,0,1,0.05f);
            sliderFloatSteps("Ring rotation (degrees)",&options_.settings.rgbaRotation,0,360,1);
            ImGui::EndDisabled();
            ImGui::TextWrapped("Four directions toward one sun disk. These are not four independent lights.");
            }
            if(ImGui::CollapsingHeader("Comparison presets")){
            const auto filterPreset=[&](auto apply){
                const int storage=options_.settings.multiViewStorage;
                const bool cascades=options_.settings.rgbaCascades;
                apply(options_.settings);
                options_.settings.multiViewStorage=storage;options_.settings.rgbaCascades=cascades;
            };
            if(ImGui::Button("Fixed PCF test baseline"))filterPreset(shadows::applyFixedAngularPreset);
            if(ImGui::Button("World-radius PCF test"))filterPreset(shadows::applyWorldAngularPreset);
            if(ImGui::Button("PCSS per-channel test"))filterPreset(shadows::applyPcssAngularPreset);
            ImGui::TextWrapped("The fixed-PCF presets differ in radius units. The PCSS preset uses a full blocker search in each channel. These presets reset mixing to weighted, moments and joint smoothing off. The selected storage/cascade variant, camera and light stay unchanged.");
            if(ImGui::Button("Apply angular PCF preset")){
                options_.settings.rgbaMoments=false;
                options_.settings.rgbaPcfMode=2;options_.settings.rgbaPcfSamples=16;
                options_.settings.rgbaPcfStrength=0.5f;options_.settings.rgbaPcfSharp=true;options_.settings.rgbaPcfMaxRadius=16;
                options_.settings.rgbaReconstruct=false;options_.settings.rgbaSmoothing=1;
                options_.settings.rgbaMixMode=0;options_.settings.rgbaAngularMix=1;
                options_.settings.resultBlend=0;
            }
            if(ImGui::Button("Apply smooth angular preset")){
                shadows::applySmoothAngularPreset(options_.settings);
            }
            if(ImGui::Button("Apply shared PCF preset")){
                shadows::applySharedAngularPreset(options_.settings);
            }
            if(ImGui::Button("Apply PCF moment preset")){
                shadows::applyMomentAngularPreset(options_.settings);
            }
            }
            if(ImGui::CollapsingHeader("Per-view filtering",ImGuiTreeNodeFlags_DefaultOpen)){
            const char* pcfModes[]={"Original per-view comparison","Fixed PCF per channel","Adaptive PCF per channel (legacy)","Shared world PCF (all channels)","PCSS per channel"};
            ImGui::Combo("Channel filter",&options_.settings.rgbaPcfMode,pcfModes,5);
            ImGui::BeginDisabled(options_.settings.rgbaPcfMode!=1&&options_.settings.rgbaPcfMode!=2&&options_.settings.rgbaPcfMode!=4);
            ImGui::Checkbox("Reconstruct edge from PCF moments",&options_.settings.rgbaMoments);
            if(options_.settings.rgbaMoments){
                sliderFloatSteps("PCF moment reconstruction strength",&options_.settings.rgbaMomentStrength,0,1,0.05f);
                ImGui::TextWrapped("Coverage and sample positions estimate a local edge. Fixed PCF also reads blocker gaps (center plus up to 8 search taps per view); PCSS reuses its blocker search. Unresolved edges or inconsistent observations fall back to the selected channel filter.");
            }
            ImGui::EndDisabled();
            if(options_.settings.rgbaMoments){
                if(shadows::usesMomentReconstruction(method,options_.settings))ImGui::TextWrapped("Moment reconstruction: enabled. Per-pixel confidence can still reject the fit.");
                else ImGui::TextWrapped("Moment reconstruction: INACTIVE. Requires fixed/adaptive PCF or per-channel PCSS, positive strength and radius, weighted mixing, angular offsets with ring radius >= 0.1, and angular mixture > 0 for cascades.");
            }
            if(ImGui::CollapsingHeader("Joint soft-shadow smoothing",ImGuiTreeNodeFlags_DefaultOpen)){
                ImGui::BeginDisabled(options_.settings.rgbaPcfMode==0);
                sliderFloatSteps("Joint blend strength",&options_.settings.resultBlend,0,1,0.05f);
                if(options_.settings.resultBlend>0){
                    sliderIntSteps("Joint neighborhood (pixels)",&options_.settings.resultBlendRadius,1,2);
                    sliderFloatSteps("Joint visibility tolerance",&options_.settings.resultBlendSigma,0.05f,0.5f,0.025f);
                }
                ImGui::EndDisabled();
                ImGui::TextWrapped("Uses penumbra widths and near-shadow protection from the channel views. Adds one screen pass; 3x3 or 5x5 samples. No temporal jitter.");
            }
            if(options_.settings.rgbaPcfMode!=0){
                const char* taps[]={"8","16","32","64"};
                int selected=options_.settings.rgbaPcfSamples==8?0:options_.settings.rgbaPcfSamples==16?1:options_.settings.rgbaPcfSamples==32?2:3;
                if(ImGui::Combo(options_.settings.rgbaPcfMode==3?"Shared PCF points":"PCF taps per channel",&selected,taps,4))options_.settings.rgbaPcfSamples=8<<selected;
                sliderFloatSteps(options_.settings.rgbaPcfMode==4?"PCSS penumbra scale":"Additional PCF strength",&options_.settings.rgbaPcfStrength,0,1,0.05f);
                if(options_.settings.rgbaPcfMode==1){
                    ImGui::Checkbox("Fixed radius in world meters (all channels)",&options_.settings.rgbaPcfWorldUnits);
                    if(options_.settings.rgbaPcfWorldUnits)
                        sliderFloatSteps("Fixed PCF radius (m)",&options_.settings.rgbaPcfWorldRadius,0,3,0.01f);
                    else sliderFloatSteps("Fixed PCF radius (texels)",&options_.settings.rgbaPcfRadius,0,32,0.25f);
                    if(options_.settings.rgbaMoments){
                        sliderFloatSteps("Moment blocker search cap (local texels)",&options_.settings.rgbaPcfMaxRadius,0,64,0.5f);
                        ImGui::Checkbox("Protect moment correction near contact",&options_.settings.rgbaPcfSharp);
                    }
                }else {
                    if(options_.settings.rgbaPcfMode==4){
                        sliderIntSteps("PCSS blocker samples per channel",&options_.settings.rgbaPcssBlockerSamples,1,128);
                        const char* means[]={"Arithmetic blocker gap","Geometric blocker gap"};
                        ImGui::Combo("PCSS blocker statistic",&options_.settings.rgbaPcssMean,means,2);
                        ImGui::Checkbox("Camera-independent blocker search",&options_.settings.rgbaPcssCasterBounds);
                        ImGui::TextWrapped("Uses receiver-to-caster bounds in each view. Camera translation and near/far placement no longer set the search radius. The world search limit still applies.");
                        sliderFloatSteps("PCSS blocker search limit (m)",&options_.settings.rgbaPcssSearchRadius,0,20,0.05f);
                    }
                    if(options_.settings.rgbaPcfMode==3){
                        ImGui::Checkbox("Adapt shared radius to blockers",&options_.settings.rgbaSharedAdaptive);
                        sliderFloatSteps(options_.settings.rgbaSharedAdaptive?"Shared radius limit (m)":"Shared radius (m)",&options_.settings.rgbaSharedRadius,0,3,0.01f);
                    }
                    ImGui::BeginDisabled(options_.settings.rgbaPcfMode==3&&!options_.settings.rgbaSharedAdaptive);
                    sliderFloatSteps("PCF radius cap (local texels)",&options_.settings.rgbaPcfMaxRadius,0,64,0.5f);
                    ImGui::EndDisabled();
                    ImGui::Checkbox("Preserve subpixel near shadows",&options_.settings.rgbaPcfSharp);
                }
                if(options_.settings.rgbaPcfMode==4)ImGui::TextWrapped("Each channel searches blockers, estimates its own penumbra from their mean gap, and filters visibility. Only the four finished PCSS results are mixed below. The center is one search sample, not an early exit. Near-shadow protection is applied separately per channel. World radius units for fixed PCF do not affect this mode.");
                else if(options_.settings.rgbaPcfMode==3)ImGui::TextWrapped("One world-space disk is projected onto the receiver plane. Each point queries every available channel at its own UV/depth, mixes visibility, then contributes to PCF. No extra map or screen pass. Radius is in meters before strength scaling.");
                else if(options_.settings.rgbaPcfMode==1)ImGui::TextWrapped("PCF filters each view before channel mixing. World units give the same radius in meters in each light-view plane, before strength scaling. Texel units follow each map's local density. Larger cascade coverage still reduces detail.");
                else ImGui::TextWrapped("PCF compares depth independently in each channel, then mixes visibility. Adaptive radius uses this view's local blocker gap; 8 search taps are needed only when the center is lit. No random jitter.");
            }
            }
            if(ImGui::CollapsingHeader("Legacy angular reconstruction")){
            if(ImGui::Button("Reconstruct soft gradients")){
                options_.settings.rgbaPcfMode=0;
                options_.settings.rgbaReconstruct=true;options_.settings.rgbaReconstructStrength=0.5f;
                options_.settings.rgbaMixMode=0;options_.settings.rgbaAngularMix=1;
                options_.settings.rgbaOffset=true;options_.settings.rgbaRadius=std::max(0.5f,options_.settings.rgbaRadius);
            }
            ImGui::Checkbox("Angular edge reconstruction (experimental)",&options_.settings.rgbaReconstruct);
            if(options_.settings.rgbaPcfMode!=0&&options_.settings.rgbaReconstruct)ImGui::TextWrapped("Angular edge reconstruction is inactive while channel PCF is selected.");
            if(options_.settings.rgbaReconstruct&&options_.settings.rgbaPcfMode==0){
                sliderFloatSteps("Reconstruction strength",&options_.settings.rgbaReconstructStrength,0,1,0.05f);
                sliderFloatSteps("Near blocker protection range (m)",&options_.settings.rgbaEdgeContactRange,0,3,0.05f);
                sliderIntSteps("Edge search radius (texels)",&options_.settings.rgbaEdgeSearch,2,128);
                ImGui::TextWrapped("Local silhouette distance and blocker gap reconstruct continuous source visibility. More depth reads, no extra geometry or screen pass. Needs all four views, angular offsets and weighted mixing.");
            }
            }
            if(ImGui::CollapsingHeader("Cascade coverage and blending",ImGuiTreeNodeFlags_DefaultOpen)){
                ImGui::Checkbox("Use channel views as cascades",&options_.settings.rgbaCascades);
                ImGui::Checkbox("Stable focus across coverage modes",&options_.settings.rgbaStableFocus);
                ImGui::TextWrapped("Keeps the same world focus with cascades on or off. Disable only to compare the previous camera/scene-target focus behavior.");
                sliderFloatSteps("First map half extent (m)",&options_.settings.rgbaExtent,0.5f,128,0.5f);
                if(options_.settings.rgbaCascades){
                    sliderFloatSteps("Extent multiplier per channel",&options_.settings.rgbaCascadeScale,1,4,0.1f);
                    sliderFloatSteps("Cascade transition width",&options_.settings.rgbaCascadeBlend,0,0.5f,0.025f);
                    sliderFloatSteps("Angular mixture over cascades",&options_.settings.rgbaAngularMix,0,1,0.05f);
                    ImGui::TextWrapped("Zero: nearest covering cascade with edge transitions. One: mix all available light directions. Larger maps trade texel density for coverage.");
                    ImGui::TextWrapped("Outside the smallest map fewer sun directions remain. Far shadows cannot retain four-direction quality with only four shared cascade channels.");
                }
                const float e=options_.settings.rgbaExtent,g=options_.settings.rgbaCascades?options_.settings.rgbaCascadeScale:1;
                ImGui::Text("Map widths R/G/B/A: %.1f / %.1f / %.1f / %.1f m",2*e,2*e*g,2*e*g*g,2*e*g*g*g);
            }
            if(ImGui::CollapsingHeader("Mixing final visibility",ImGuiTreeNodeFlags_DefaultOpen)){
            if(options_.settings.rgbaCascades&&options_.settings.rgbaAngularMix<=0)
                ImGui::TextWrapped("Enable Angular mixture over cascades to use angular mixing controls.");
            ImGui::BeginDisabled(options_.settings.rgbaCascades&&options_.settings.rgbaAngularMix<=0);
            const char* mixtures[]={"Weighted visibility (area-light estimate)","Darkest visibility (artistic)","Product visibility (artistic)","Weighted geometric visibility","Arithmetic / geometric blend"};
            ImGui::Combo("Angular shadow mixing",&options_.settings.rgbaMixMode,mixtures,5);
            if(options_.settings.rgbaMixMode==4)
                sliderFloatSteps("Geometric blend amount",&options_.settings.rgbaGeometricBlend,0,1,0.05f);
            sliderFloatSteps("Central view weight",&options_.settings.rgbaCenterWeight,0,1,0.05f);
            if(ImGui::Button("Equal R/G/B/A weights"))options_.settings.rgbaCenterWeight=0.25f;
            if(options_.settings.rgbaMixMode>=3)ImGui::TextWrapped("Geometric mixing favors darker views. A zero in any positively weighted channel gives geometric visibility zero. This can darken penumbrae; it is a comparison control, not recovered light visibility. Moment reconstruction is inactive for these modes.");
            ImGui::EndDisabled();
            const char* kernels[]={"Nearest","Bilinear","Hermite","Bicubic B-spline"};
            ImGui::Combo("Visibility reconstruction",&options_.settings.rgbaSmoothing,kernels,4);
            }
            ImGui::TextWrapped("View 0: center. Views 1/2/3: sun ring, 120 degrees apart. Each has its own projection and optional nested coverage.");
            ImGui::TextWrapped("Only four light directions: banding can remain. Reconstruction smooths map edges; it does not recover missing angular samples.");
        }
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
            if(ImGui::Button("Clean blue-noise preset"))
                shadows::applyCleanBlueNoisePreset(options_.settings);
            ImGui::SetItemTooltip("Keep sample counts. Enable Hermite, blue noise, packed endpoints, local contact protection and edge-aware result blend.");
            if(ImGui::CollapsingHeader("Result blend and contact protection")){
            sliderFloatSteps("Result blend", &options_.settings.resultBlend,0,1,0.1f);
            if(options_.settings.resultBlend>0){
                sliderIntSteps("Blend radius (pixels)",&options_.settings.resultBlendRadius,1,2);
                sliderFloatSteps("Blend visibility tolerance",&options_.settings.resultBlendSigma,0.05f,0.5f,0.025f);
                ImGui::TextWrapped("One extra pass filters finished visibility. Geometry, penumbra radius and supported contact cores guide the blend.");
            }
            sliderFloatSteps("Local contact protection",&options_.settings.contactGuard,0,1,0.1f);
            if(options_.settings.contactGuard>0)
                sliderFloatSteps("Contact blocker range (m)",&options_.settings.contactMaxGap,0.05f,3.0f,0.05f);
            ImGui::Checkbox("Packed min/max depth (R32UI)",&options_.settings.packedIntervals);
            if(options_.settings.contactGuard>0&&!options_.settings.packedIntervals)
                ImGui::TextWrapped("D32F can protect only visible blockers. Packed endpoints can retain a hidden lower caster.");
            }
            if(ImGui::CollapsingHeader("PCSS sampling",ImGuiTreeNodeFlags_DefaultOpen)){
            sliderIntSteps("Filter samples", &options_.settings.filterSamples, 1, 256);
            sliderIntSteps("Blocker samples", &options_.settings.blockerSamples, 1, 256);
            const char* patterns[] = {"Vogel", "Random disk", "Stratified random disk", "Vogel + blue noise (world)"};
            int pattern = shadows::effectiveSamplePattern(method, options_.settings);
            if (ImGui::Combo("PCSS sample pattern", &pattern, patterns, 4)) {
                // Editing a preset's pattern keeps its bilinear reconstruction.
                if (options_.settings.smoothing >= 6)
                    options_.settings.smoothing = 1;
                options_.settings.samplePattern = pattern;
            }
            if (shadows::effectiveSamplePattern(method, options_.settings) != 0) {
                ImGui::InputInt("Sample seed", &options_.settings.sampleSeed);
                options_.settings.sampleSeed = std::max(0, options_.settings.sampleSeed);
                if (pattern == 3) {
                    sliderFloatSteps("Blue-noise tile (m)", &options_.settings.blueNoiseTile, 0.01f, 64.0f, 0.1f, "%.2f");
                    ImGui::TextWrapped("World XZ tile, default 1 x 1 m. Two scalar blue-noise reads rotate the blocker/filter Vogel disks. Sample counts stay unchanged; screen rotation and shared pattern do not apply.");
                } else {
                    if (ImGui::Checkbox("Share random sample pattern", &options_.settings.sharedSamplePattern) && options_.settings.sharedSamplePattern)
                        options_.settings.rotateSamples = false;
                    ImGui::TextWrapped(options_.settings.sharedSamplePattern
                        ? "Shared random disk: coherent integration, less grain. Blocker/filter streams remain independent. Keep rotation off."
                        : "Repeatable per-pixel jitter. Independent blocker and filter samples; may show spatial grain.");
                }
            }
        }
        }
        if (method == shadows::Method::GeometricLayered) {
            if (ImGui::Button("Apply soft packed preset"))
                shadows::applySoftPackedPreset(options_.settings);
            ImGui::SetItemTooltip("Packed min/max + Hermite visibility/blockers + 64 shared stratified filter and blocker samples + 1 texel minimum radius. Per-pixel rotation off.");
            if(options_.settings.packedIntervals)
                ImGui::TextWrapped("One R32UI map, two UNORM16 surface endpoints. Keeps some occluded blockers; loses intermediate depths.");
            sliderFloatSteps("Depth band separation", &options_.settings.layerSeparation, 1.5f, 8.0f, 0.1f);
            sliderFloatSteps("Minimum filter radius (texels)", &options_.settings.layerMinFilterTexels, 0, 4, 0.25f);
            ImGui::TextWrapped("Minimum spatial softness. Zero keeps physical radii; larger values also soften contact shadows.");
            ImGui::TextWrapped("Separate near/far radii; paired angular samples combine blockers before averaging.");
        }
        if (shadows::usesPrefilter(method)) {
            sliderIntSteps("Gaussian radius (texels)", &options_.settings.prefilterRadius, 0, 16);
            if(method==shadows::Method::Gsm){
                sliderFloatSteps("GSM sharpness",&options_.settings.gsmSharpness,1,128,1);
                ImGui::TextWrapped("Project GSM: encode log(1+depth), Gaussian-prefilter the map, reconstruct geometric mean and apply a power-ratio visibility response. No blocker search or PCSS. Sharpness controls attenuation; Gaussian radius controls edge width. This approximation can bias overlapping shadows.");
            }
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
        }
        if(ImGui::CollapsingHeader("Advanced diagnostics")){
        ImGui::BeginDisabled(stress && options_.settings.stressObjects > shadows::maxCasters);
        ImGui::Checkbox("Analytic diagnostic maps", &options_.settings.analyticMaps);
        ImGui::EndDisabled();
        ImGui::BeginDisabled(shadows::effectiveSamplePattern(method, options_.settings) == 3);
        ImGui::Checkbox("Rotate sample pattern", &options_.settings.rotateSamples);
        ImGui::EndDisabled();
        }
        if(ImGui::CollapsingHeader("Camera and capture",ImGuiTreeNodeFlags_DefaultOpen)){
        sliderIntSteps("Camera preset", &options_.settings.camera, 0, 1);
        if (ImGui::Button("Reset camera (R)"))
            cameraScene_ = -1;
        ImGui::TextWrapped(
            "RMB: orbit. MMB: pan. Wheel: zoom. WASD: move, Q/E: down/up, Shift: faster. P: save image.");
        if (ImGui::Button("Save clean image"))capture_=true;
        }
        ImGui::SeparatorText("Performance");
        ImGui::Text("G-buffer GPU: %.3f ms", gbufferMilliseconds_);
        ImGui::Text("Map GPU: %.3f ms / shadow GPU: %.3f ms", mapMilliseconds_, shadowMilliseconds_);
        ImGui::Text("GPU components: %.3f ms / frame rate: %.1f FPS", gbufferMilliseconds_ + milliseconds_, ImGui::GetIO().Framerate);

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
