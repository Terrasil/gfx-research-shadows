#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <glad/gl.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <shadows/chs_weights.hpp>
#include <shadows/experiment.hpp>
#include <sstream>
#include <stdexcept>

namespace shadows {
namespace {
double median(std::vector<double> v) {
    if (v.empty())
        return 0;
    std::sort(v.begin(), v.end());
    auto n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) * 0.5;
}
struct Metrics {
    double rmse = 0, penumbraRmse = 0, gradientRmse = 0, maxError = 0, brightening = 0, brighteningRate = 0;
    std::size_t pixels = 0, penumbraPixels = 0, invalid = 0;
};
Metrics compare(const Image& a, const Image& reference, const Image* without = nullptr) {
    if (a.visibility.size() != reference.visibility.size())
        throw std::runtime_error("Image size mismatch");
    Metrics m;
    double sum = 0, pen = 0, gradient = 0;
    std::size_t gradientCount = 0, violations = 0;
    for (std::size_t i = 0; i < a.visibility.size(); ++i) {
        if (!std::isfinite(a.visibility[i]) || !std::isfinite(reference.visibility[i])) {
            ++m.invalid;
            continue;
        }
        if (a.visibility[i] < -1e-5 || a.visibility[i] > 1.00001f) {
            ++m.invalid;
            continue;
        }
        if (a.receiver[i] < 0.5f || a.depth[i] >= 1)
            continue;
        ++m.pixels;
        double error = double(a.visibility[i]) - reference.visibility[i];
        sum += error * error;
        m.maxError = std::max(m.maxError, std::abs(error));
        if (reference.visibility[i] > 0.01f && reference.visibility[i] < 0.99f) {
            ++m.penumbraPixels;
            pen += error * error;
        }
        if (without) {
            if (!std::isfinite(without->visibility[i])) {
                ++m.invalid;
                continue;
            }
            double delta = double(a.visibility[i]) - without->visibility[i];
            m.brightening = std::max(m.brightening, delta);
            if (delta > 1e-4)
                ++violations;
        }
        const int x = int(i % std::size_t(a.width)), y = int(i / std::size_t(a.width));
        if (x + 1 < a.width && y + 1 < a.height) {
            for (auto j : {i + 1, i + std::size_t(a.width)})
                if (a.receiver[j] > 0.5f) {
                    double d = (double(a.visibility[j]) - a.visibility[i]) -
                               (double(reference.visibility[j]) - reference.visibility[i]);
                    gradient += d * d;
                    ++gradientCount;
                }
        }
    }
    if (!m.pixels)
        throw std::runtime_error("No receiver pixels in experiment");
    m.rmse = std::sqrt(sum / double(m.pixels));
    m.penumbraRmse = m.penumbraPixels ? std::sqrt(pen / double(m.penumbraPixels)) : 0;
    m.gradientRmse = gradientCount ? std::sqrt(gradient / double(gradientCount)) : 0;
    m.brighteningRate = double(violations) / double(m.pixels);
    return m;
}
void pfm(const std::filesystem::path& path, const std::vector<float>& values, int width, int height) {
    if (values.size() != std::size_t(width) * std::size_t(height))
        throw std::runtime_error("PFM size mismatch");
    std::ofstream out(path, std::ios::binary);
    out << "Pf\n" << width << ' ' << height << "\n-1.0\n";
    out.write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size() * sizeof(float)));
    if (!out)
        throw std::runtime_error("Cannot write PFM");
}
std::string jsonString(std::string v) {
    std::string out = "\"";
    for (char c : v) {
        if (c == '"' || c == '\\')
            out += '\\';
        if (c == '\n') {
            out += "\\n";
            continue;
        }
        out += c;
    }
    return out + '"';
}
const char* glString(GLenum key) {
    auto p = glGetString(key);
    return p ? reinterpret_cast<const char*>(p) : "unknown";
}
struct Case {
    std::string scene;
    Settings settings;
};
std::vector<Case> cases(const Options& o) {
    std::vector<Case> out;
    auto scenes = o.full    ? sceneIds()
                  : o.quick ? std::vector<std::string>{"mixed-overlap", "crossing", "hidden-layer", "sloped"}
                  : o.smoke ? std::vector<std::string>{"mixed-overlap", "hidden-layer"}
                            : std::vector<std::string>{o.scene};
    auto resolutions = o.full ? std::vector<std::pair<int, int>>{{1280, 720}, {1920, 1080}}
                              : std::vector<std::pair<int, int>>{{o.settings.width, o.settings.height}};
    auto maps = o.full ? std::vector<int>{1024, 2048} : std::vector<int>{o.settings.mapResolution};
    auto angles = o.full ? std::vector<float>{0.27f, 1.5f, 3.0f} : std::vector<float>{o.settings.angle};
    for (const auto& scene : scenes)
        for (auto [w, h] : resolutions)
            for (int map : maps)
                for (float angle : angles) {
                    auto s = o.settings;
                    s.width = w;
                    s.height = h;
                    s.mapResolution = map;
                    s.angle = angle;
                    out.push_back({scene, s});
                }
    return out;
}
std::vector<Method> methods(const Options& o) {
    std::vector<Method> out;
    if (o.method == "all" || o.quick || o.full || o.smoke) {
        for (int i = 0; i < methodCount; ++i)
            out.push_back(Method(i));
    } else {
        out.push_back(parseMethod(o.method));
        if (out[0] != Method::Reference)
            out.push_back(Method::Reference);
    }
    return out;
}
void manifest(const std::filesystem::path& path, const Options& o, int planned, bool completed) {
    std::ofstream out(path);
    std::string gpu = glString(GL_RENDERER);
    bool software = gpu.find("llvmpipe") != std::string::npos || gpu.find("softpipe") != std::string::npos;
    out << "{\n\"version\":\"0.12.0\",\n\"completed\":" << (completed ? "true" : "false")
        << ",\n\"expected_cases\":" << planned << ",\n\"gpu\":" << jsonString(gpu)
        << ",\n\"software_renderer\":" << (software ? "true" : "false")
        << ",\n\"gl_version\":" << jsonString(glString(GL_VERSION))
        << ",\n\"base_revision\":" << jsonString(GFX_SHADOWS_BASE_REVISION)
        << ",\n\"compiler\":" << jsonString(GFX_SHADOWS_COMPILER)
        << ",\n\"build_type\":" << jsonString(GFX_SHADOWS_BUILD_TYPE) << ",\n\"profile\":"
        << jsonString(o.full    ? "publication"
                      : o.quick ? "quick"
                      : o.smoke ? "smoke"
                                : "manual")
        << ",\n\"method_count\":7,\n\"filter_validation_cases\":320,\n\"sampling_validation_cases\":1568"
        << ",\n\"dpcf_source\":\"Filament v1.23.1 a67d50b9e29cf7bf2f0a6f0b7bf4668cdb87770a\""
        << ",\n\"chs_source\":\"AMD ShadowFX c6d931c27275b1bb65ba0abfb71eaba86f355387 contactFixedGather4\""
        << ",\n\"chs_constant_buffer_bytes\":" << sizeof(chsKernelData)
        << ",\n\"shader_programs\":\"separate method programs; CHS kernel and PCF/PCSS smoothing specialized at "
           "compile time\""
        << ",\n\"smoothing\":" << o.settings.smoothing
        << ",\n\"configured_sample_pattern\":" << jsonString(samplePatternId(o.settings.samplePattern))
        << ",\n\"sample_seed\":" << o.settings.sampleSeed
        << ",\n\"rotate_samples\":" << (o.settings.rotateSamples ? "true" : "false")
        << ",\n\"reference_samples\":" << o.settings.referenceSamples << ",\n\"warmup\":" << o.warmup
        << ",\n\"measure\":" << o.measure << ",\n\"warp\":" << o.settings.warp
        << ",\n\"custom_camera\":" << (o.settings.customCamera ? "true" : "false")
        << ",\n\"camera_yaw\":" << o.settings.cameraYaw << ",\n\"camera_pitch\":" << o.settings.cameraPitch
        << ",\n\"camera_distance\":" << o.settings.cameraDistance << ",\n\"camera_target\":["
        << o.settings.cameraTarget.x << ',' << o.settings.cameraTarget.y << ',' << o.settings.cameraTarget.z << ']'
        << ",\n\"map_backend\":" << jsonString(o.settings.analyticMaps ? "analytic-box-oracle" : "raster-tessellated")
        << ",\n\"receiver_metrics_only\":true,\n\"temporal_reuse\":false,"
           "\n\"timing_scope\":\"map generation + shadow evaluation, excludes shared G-buffer, UI and readback\","
           "\n\"reference_model\":\"irradiance-weighted angular disk, exact AABB intersections\"\n}\n";
}
double cpuCheck(Renderer& renderer, const Image& reference) {
    const auto positions = renderer.positions(), normals = renderer.normals();
    const auto& s = renderer.settings();
    double maxError = 0;
    int tested = 0;
    const std::size_t stride = std::max<std::size_t>(1, reference.visibility.size() / 192);
    for (std::size_t i = stride / 2; i < reference.visibility.size(); i += stride) {
        if (reference.receiver[i] < 0.5f || reference.depth[i] >= 1)
            continue;
        glm::dvec3 p(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]),
            n(normals[i * 4], normals[i * 4 + 1], normals[i * 4 + 2]);
        double value =
            cpuVisibility(renderer.scene(), p, glm::normalize(n), s.referenceSamples, s.angle, -1, s.normalBias);
        maxError = std::max(maxError, std::abs(value - double(reference.visibility[i])));
        ++tested;
    }
    if (!tested)
        throw std::runtime_error("CPU reference oracle tested no receiver pixels");
    return maxError;
}
} // namespace
void printHelp() {
    std::cout
        << "gfx-research-shadows 0.12.0\n"
           "Methods: hard | pcf | pcss | arr | geometric | dpcf | chs\n"
           "Interactive: --scene=crossing --method=geometric\n"
           "Experiments: --benchmark | --smoke | --publication-quick | --publication-suite, --method=all\n"
           "--width=1280 --height=720 --shadow-resolution=1024 --angle=1.5 --warp=8\n"
           "--samples=48 --blocker-samples=48 --reference-samples=512 --pcf-radius=2\n"
           "--dpcf-taps=12 --dpcf-radius=6 --dpcf-ratio-scale=1 --chs-kernel=11 --chs-sun-size=4\n"
           "DPCF radius and AMD sun size are calibrated at 1.5 degrees; scaled with tan(angle).\n"
           "CHS kernels: 7,9,11,13,15. --smoothing=0|1|2|3|4|5|6|7 (PCF, PCSS, geometric).\n"
           "Smoothing: point | bilinear | bilinear+blockers | Hermite | B-spline | bicubic Catmull-Rom.\n"
           "Presets 6/7: bilinear + random/stratified samples (PCSS and geometric).\n"
           "--geometric-smoothing is a compatibility alias for --smoothing.\n"
           "--sample-pattern=vogel|random|stratified --sample-seed=0 (PCSS and geometric).\n"
           "--depth-bias=0.00015 --normal-bias=0.002 --max-radius=3 --rotate-samples\n"
           "--map-backend=raster|analytic --camera=0|1 --warmup=12 --measure=30 --output=results/manual\n"
           "--camera-yaw=-2.1 --camera-pitch=0.65 --camera-distance=16\n"
           "--camera-target-x=-1.4 --camera-target-y=1 --camera-target-z=-0.8 --headless\n"
           "Controls: RMB orbit, MMB pan, wheel zoom, WASD move, QE vertical, Shift faster, R reset, P capture.\n";
}
Options parseOptions(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto equal = arg.find('=');
        std::string key = arg.substr(0, equal), value = equal == std::string::npos ? "" : arg.substr(equal + 1);
        if (key == "--help" || key == "-h") {
            printHelp();
            std::exit(0);
        }
        if (key == "--version") {
            std::cout << "gfx-research-shadows 0.12.0, " << GFX_SHADOWS_COMPILER << ", " << GFX_SHADOWS_BUILD_TYPE
                      << ", base " << GFX_SHADOWS_BASE_REVISION << '\n';
            std::exit(0);
        }
        if (key == "--publication-quick" || key == "--publication-suite" || key == "--smoke") {
            o.benchmark = true;
            o.quick = key == "--publication-quick";
            o.full = key == "--publication-suite";
            o.smoke = key == "--smoke";
            o.settings.width = o.full ? 1280 : o.quick ? 640 : 320;
            o.settings.height = o.full ? 720 : o.quick ? 360 : 180;
            o.settings.mapResolution = o.full ? 1024 : o.quick ? 512 : 256;
            o.settings.referenceSamples = o.full ? 1024 : o.quick ? 512 : 128;
            o.warmup = o.full ? 120 : o.quick ? 3 : 0;
            o.measure = o.full ? 300 : o.quick ? 8 : 1;
            o.output = o.full ? "results/publication" : o.quick ? "results/quick" : "results/smoke";
            continue;
        }
        if (key == "--benchmark") {
            o.benchmark = true;
            continue;
        }
        if (key == "--headless") {
            o.headless = true;
            continue;
        }
        if (key == "--rotate-samples") {
            o.settings.rotateSamples = true;
            continue;
        }
        if (value.empty() && i + 1 < argc)
            value = argv[++i];
        if (value.empty())
            throw std::runtime_error("Missing value: " + key);
        auto& t = o.settings;
        if (key == "--width")
            t.width = std::stoi(value);
        else if (key == "--height")
            t.height = std::stoi(value);
        else if (key == "--shadow-resolution")
            t.mapResolution = std::stoi(value);
        else if (key == "--samples")
            t.filterSamples = std::stoi(value);
        else if (key == "--blocker-samples")
            t.blockerSamples = std::stoi(value);
        else if (key == "--reference-samples")
            t.referenceSamples = std::stoi(value);
        else if (key == "--dpcf-taps")
            t.dpcfTaps = std::stoi(value);
        else if (key == "--chs-kernel")
            t.chsKernel = std::stoi(value);
        else if (key == "--smoothing" || key == "--geometric-smoothing")
            t.smoothing = std::stoi(value);
        else if (key == "--sample-pattern")
            t.samplePattern = parseSamplePattern(value);
        else if (key == "--sample-seed")
            t.sampleSeed = std::stoi(value);
        else if (key == "--camera")
            t.camera = std::stoi(value);
        else if (key == "--pcf-radius")
            t.fixedPcfRadius = std::stof(value);
        else if (key == "--angle")
            t.angle = std::stof(value);
        else if (key == "--warp")
            t.warp = std::stof(value);
        else if (key == "--depth-bias")
            t.depthBias = std::stof(value);
        else if (key == "--normal-bias")
            t.normalBias = std::stof(value);
        else if (key == "--max-radius")
            t.maxRadiusWorld = std::stof(value);
        else if (key == "--dpcf-radius")
            t.dpcfRadius = std::stof(value);
        else if (key == "--dpcf-ratio-scale")
            t.dpcfRatioScale = std::stof(value);
        else if (key == "--chs-sun-size")
            t.chsSunSize = std::stof(value);
        else if (key == "--warmup")
            o.warmup = std::stoi(value);
        else if (key == "--measure")
            o.measure = std::stoi(value);
        else if (key == "--scene")
            o.scene = value;
        else if (key == "--method")
            o.method = value;
        else if (key == "--output")
            o.output = value;
        else if (key == "--map-backend") {
            if (value != "raster" && value != "analytic")
                throw std::runtime_error("Unknown map backend");
            t.analyticMaps = value == "analytic";
        } else if (key == "--camera-yaw" || key == "--camera-pitch" || key == "--camera-distance" ||
                   key == "--camera-target-x" || key == "--camera-target-y" || key == "--camera-target-z") {
            t.customCamera = true;
            float v = std::stof(value);
            if (key == "--camera-yaw")
                t.cameraYaw = v;
            if (key == "--camera-pitch")
                t.cameraPitch = v;
            if (key == "--camera-distance")
                t.cameraDistance = v;
            if (key == "--camera-target-x")
                t.cameraTarget.x = v;
            if (key == "--camera-target-y")
                t.cameraTarget.y = v;
            if (key == "--camera-target-z")
                t.cameraTarget.z = v;
        } else
            throw std::runtime_error("Unknown option: " + key);
    }
    const auto& t = o.settings;
    for (float v : {t.angle, t.warp, t.depthBias, t.normalBias, t.maxRadiusWorld, t.fixedPcfRadius, t.dpcfRadius,
                    t.dpcfRatioScale, t.chsSunSize, t.cameraYaw, t.cameraPitch, t.cameraDistance, t.cameraTarget.x,
                    t.cameraTarget.y, t.cameraTarget.z})
        if (!std::isfinite(v))
            throw std::runtime_error("Non-finite argument");
    if (t.width < 16 || t.width > 8192 || t.height < 16 || t.height > 8192 || t.mapResolution < 32 ||
        t.mapResolution > 8192 || !std::has_single_bit(unsigned(t.mapResolution)) || t.filterSamples < 1 ||
        t.filterSamples > 256 || t.blockerSamples < 1 || t.blockerSamples > 256 || t.referenceSamples < 16 ||
        t.referenceSamples > 16384 || t.angle <= 0 || t.angle > 10 || t.warp < 0 || t.warp > 16 || t.camera < 0 ||
        t.camera > 1 || o.measure < 1 || o.warmup < 0 || t.fixedPcfRadius < 0 || t.fixedPcfRadius > 8 ||
        t.depthBias < 0 || t.depthBias > 0.01 || t.normalBias < 0 || t.normalBias > 0.1 || t.maxRadiusWorld <= 0 ||
        t.maxRadiusWorld > 20 || t.smoothing < 0 || t.smoothing > 7 || t.samplePattern < 0 || t.samplePattern > 2 ||
        t.sampleSeed < 0 || t.dpcfTaps < 1 || t.dpcfTaps > 64 || t.dpcfRadius <= 0 || t.dpcfRadius > 128 ||
        t.dpcfRatioScale < 1 || t.dpcfRatioScale > 1000 || t.chsSunSize <= 0 || t.chsSunSize > 100 ||
        (t.chsKernel != 7 && t.chsKernel != 9 && t.chsKernel != 11 && t.chsKernel != 13 && t.chsKernel != 15) ||
        t.cameraPitch < -1.45 || t.cameraPitch > 1.45 || t.cameraDistance < 0.5 || t.cameraDistance > 60)
        throw std::runtime_error("Argument outside supported range");
    if (o.headless && !o.benchmark)
        throw std::runtime_error("--headless requires an experiment mode");
    (void)makeScene(o.scene);
    if (o.method != "all")
        (void)parseMethod(o.method);
    return o;
}
void runExperiment(Renderer& renderer, const Options& o) {
    std::filesystem::create_directories(o.output / "captures");
    auto configs = cases(o);
    auto selected = methods(o);
    int expected = int(configs.size() * selected.size()), done = 0, index = 0;
    manifest(o.output / "run_manifest.json", o, expected, false);
    renderer.validateFilters(o.output / "filter_gpu.csv");
    renderer.validateSampling(o.output / "sampling_gpu.csv");
    std::ofstream raw(o.output / "raw.csv"), timings(o.output / "timings.csv"),
        captures(o.output / "capture_manifest.csv"), convergence(o.output / "reference_convergence.csv");
    if (!raw || !timings || !captures || !convergence)
        throw std::runtime_error("Cannot create experiment outputs");
    raw << "case_id,scene,method,width,height,map_resolution,angle,camera,map_backend,map_count,map_bytes,shadow_"
           "passes,timing_samples,map_gpu_ms,shadow_gpu_ms,total_gpu_ms,rmse,penumbra_rmse,gradient_rmse,max_abs_error,"
           "receiver_pixels,penumbra_pixels,invalid_pixels,monotonic_max_brightening,monotonic_violation_fraction,map_"
           "generation_passes,auxiliary_map_bytes,depth_layers,configured_filter_samples,configured_blocker_samples,"
           "dpcf_taps,dpcf_radius,"
           "dpcf_ratio_scale,chs_kernel,chs_sun_size,warp,smoothing,smoothing_kernel,configured_sample_pattern,"
           "sample_pattern,sample_seed,rotate_"
           "samples\n";
    timings << "case_id,sample,map_gpu_ms,shadow_gpu_ms,total_gpu_ms\n";
    captures << "case_id,kind,width,height,path\n";
    convergence << "config_id,lower_samples,reference_samples,rmse,max_abs_error,cpu_gpu_max_abs_error\n";
    for (const auto& c : configs) {
        renderer.prepare(makeScene(c.scene), c.settings);
        std::string config = "c" + std::to_string(index++);
        renderer.render(Method::Reference);
        auto reference = renderer.read();
        unsigned refTexture = renderer.copyResult();
        double cpuError = cpuCheck(renderer, reference);
        pfm(o.output / "captures" / (config + "-reference.pfm"), reference.visibility, c.settings.width,
            c.settings.height);
        pfm(o.output / "captures" / (config + "-receiver.pfm"), reference.receiver, c.settings.width,
            c.settings.height);
        for (const auto& suffix : {"reference", "receiver"})
            captures << config << ',' << suffix << ',' << c.settings.width << ',' << c.settings.height << ",captures/"
                     << config << '-' << suffix << ".pfm\n";
        auto lower = c.settings;
        lower.referenceSamples = std::max(8, c.settings.referenceSamples / 2);
        renderer.prepare(makeScene(c.scene), lower);
        renderer.render(Method::Reference);
        auto delta = compare(renderer.read(), reference);
        convergence << config << ',' << lower.referenceSamples << ',' << c.settings.referenceSamples << ','
                    << delta.rmse << ',' << delta.maxError << ',' << cpuError << '\n';
        renderer.prepare(makeScene(c.scene), c.settings);
        std::rotate(selected.begin(), selected.begin() + 1, selected.end());
        for (Method method : selected) {
            std::string id = config + "-" + methodId(method);
            bool arr = method == Method::Reference;
            for (int i = 0; i < (arr ? 0 : o.warmup); ++i)
                renderer.render(method);
            glFinish();
            std::vector<double> mt, st, tt;
            int samples = arr ? 1 : o.measure;
            for (int i = 0; i < samples; ++i) {
                auto t = renderer.render(method, -1, true);
                mt.push_back(t.mapMs);
                st.push_back(t.shadowMs);
                tt.push_back(t.mapMs + t.shadowMs);
                timings << id << ',' << i << ',' << std::setprecision(10) << t.mapMs << ',' << t.shadowMs << ','
                        << t.mapMs + t.shadowMs << '\n';
            }
            auto image = renderer.read();
            pfm(o.output / "captures" / (id + ".pfm"), image.visibility, c.settings.width, c.settings.height);
            captures << id << ",visibility-pfm," << c.settings.width << ',' << c.settings.height << ",captures/" << id
                     << ".pfm\n";
            for (int view = 0; view < 3; ++view) {
                std::string suffix = view == 0   ? "shaded"
                                     : view == 1 ? "visibility"
                                                 : "difference",
                            name = id + "-" + suffix + ".png";
                renderer.capture(o.output / "captures" / name, view, refTexture);
                captures << id << ',' << suffix << ',' << c.settings.width << ',' << c.settings.height << ",captures/"
                         << name << '\n';
            }
            renderer.render(method, 0);
            auto without = renderer.read();
            auto m = compare(image, reference, &without);
            int count = activeMapLayers(method, c.settings);
            raw << id << ',' << c.scene << ',' << methodId(method) << ',' << c.settings.width << ','
                << c.settings.height << ',' << c.settings.mapResolution << ',' << c.settings.angle << ','
                << c.settings.camera << ',' << (c.settings.analyticMaps ? "analytic" : "raster") << ',' << count << ','
                << activeMapBytes(method, c.settings) << ',' << (arr ? c.settings.referenceSamples + 1 : 1) << ','
                << samples << ',' << std::setprecision(10) << median(mt) << ',' << median(st) << ',' << median(tt)
                << ',' << m.rmse << ',' << m.penumbraRmse << ',' << m.gradientRmse << ',' << m.maxError << ','
                << m.pixels << ',' << m.penumbraPixels << ',' << m.invalid << ',' << m.brightening << ','
                << m.brighteningRate << ',' << mapGenerationPasses(method, c.settings) << ",0," << count << ','
                << c.settings.filterSamples << ',' << c.settings.blockerSamples << ',' << c.settings.dpcfTaps << ','
                << c.settings.dpcfRadius << ',' << c.settings.dpcfRatioScale << ',' << c.settings.chsKernel << ','
                << c.settings.chsSunSize << ',' << c.settings.warp << ',' << c.settings.smoothing << ','
                << smoothingKernel(method, c.settings) << ',' << samplePatternId(c.settings.samplePattern) << ','
                << samplePatternId(effectiveSamplePattern(method, c.settings)) << ',' << c.settings.sampleSeed << ','
                << (c.settings.rotateSamples ? 1 : 0) << '\n';
            raw.flush();
            std::cout << '[' << ++done << '/' << expected << "] " << c.scene << " / " << methodId(method)
                      << " RMSE=" << m.rmse << " GPU=" << median(tt) << " ms\n"
                      << std::flush;
            if (m.invalid)
                throw std::runtime_error("Invalid visibility");
            if (cpuError > 0.025)
                throw std::runtime_error("ARR disagrees with CPU oracle");
        }
        glDeleteTextures(1, &refTexture);
    }
    if (done != expected)
        throw std::runtime_error("Incomplete case matrix");
    raw.close();
    timings.close();
    captures.close();
    convergence.close();
    manifest(o.output / "run_manifest.json", o, expected, true);
}
} // namespace shadows
