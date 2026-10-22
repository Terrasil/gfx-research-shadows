#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <glad/gl.h>
#include <iomanip>
#include <iostream>
#include <numeric>
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
    out << "{\n  \"project\": \"gfx-research-shadows\",\n  \"version\": \"0.1.0\",\n  \"base_revision\": "
        << jsonString(GFX_SHADOWS_BASE_REVISION) << ",\n  \"profile\": "
        << jsonString(o.full    ? "publication"
                      : o.quick ? "quick"
                      : o.smoke ? "smoke"
                                : "manual")
        << ",\n  \"completed\": " << (completed ? "true" : "false") << ",\n  \"expected_cases\": " << planned
        << ",\n  \"gpu\": " << jsonString(gpu) << ",\n  \"vendor\": " << jsonString(glString(GL_VENDOR))
        << ",\n  \"gl_version\": " << jsonString(glString(GL_VERSION))
        << ",\n  \"software_renderer\": " << (software ? "true" : "false")
        << ",\n  \"compiler\": " << jsonString(GFX_SHADOWS_COMPILER)
        << ",\n  \"build_type\": " << jsonString(GFX_SHADOWS_BUILD_TYPE) << ",\n  \"map_backend\": "
        << jsonString(o.settings.analyticMaps ? "analytic-box-oracle" : "raster-tessellated")
        << ",\n  \"reference_samples\": " << o.settings.referenceSamples
        << ",\n  \"filter_samples\": " << o.settings.filterSamples
        << ",\n  \"blocker_samples\": " << o.settings.blockerSamples << ",\n  \"warmup\": " << o.warmup
        << ",\n  \"measure\": " << o.measure << ",\n  \"warp\": " << o.settings.warp
        << ",\n  \"normal_bias\": " << o.settings.normalBias << ",\n  \"depth_bias\": " << o.settings.depthBias
        << ",\n  \"max_radius_world\": " << o.settings.maxRadiusWorld << ",\n  \"camera\": " << o.settings.camera
        << ",\n  \"rotated_samples\": " << (o.settings.rotateSamples ? "true" : "false")
        << ",\n  \"receiver_metrics_only\": true,\n  \"timing_scope\": \"map generation + shadow evaluation; excludes "
           "shared G-buffer, UI and readback\",\n  \"reference_model\": \"irradiance-weighted angular disk, exact AABB "
           "intersections, one direction per fullscreen pass\",\n  \"temporal_reuse\": false\n}\n";
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
        << "gfx-research-shadows\n"
        << "Interactive: gfx-research-shadows [--scene=mixed-overlap] [--method=ndc-pcss]\n"
        << "Experiments: --publication-quick | --publication-suite | --smoke | --benchmark\n"
        << "  --method=all|hard|pcf|pcss|pcss-min-gap|ndc-pcss|split-min|split-product|binned-min|ray16|reference\n"
        << "  --scene=single-high|single-low|mixed-overlap|crossing|hidden-layer|sloped|dense\n"
        << "  --width=1280 --height=720 --shadow-resolution=1024 --angle=1.5 --warp=8\n"
        << "  --samples=48 --blocker-samples=24 --reference-samples=256 --warmup=12 --measure=30\n"
        << "  --map-backend=raster|analytic --camera=0|1 --rotate-samples --headless --output=results/manual\n";
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
            std::cout << "gfx-research-shadows 0.1.0, " << GFX_SHADOWS_COMPILER << ", " << GFX_SHADOWS_BUILD_TYPE
                      << ", base " << GFX_SHADOWS_BASE_REVISION << '\n';
            std::exit(0);
        }
        if (key == "--publication-quick") {
            o.quick = true;
            o.benchmark = true;
            o.settings.width = 640;
            o.settings.height = 360;
            o.settings.mapResolution = 512;
            o.settings.referenceSamples = 128;
            o.warmup = 3;
            o.measure = 8;
            o.output = "results/quick";
            continue;
        }
        if (key == "--publication-suite") {
            o.full = true;
            o.benchmark = true;
            o.settings.referenceSamples = 1024;
            o.warmup = 120;
            o.measure = 300;
            o.output = "results/publication";
            continue;
        }
        if (key == "--smoke") {
            o.smoke = true;
            o.benchmark = true;
            o.settings.width = 320;
            o.settings.height = 180;
            o.settings.mapResolution = 256;
            o.settings.referenceSamples = 64;
            o.warmup = 0;
            o.measure = 1;
            o.output = "results/smoke";
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
        if (key == "--width")
            o.settings.width = std::stoi(value);
        else if (key == "--height")
            o.settings.height = std::stoi(value);
        else if (key == "--shadow-resolution")
            o.settings.mapResolution = std::stoi(value);
        else if (key == "--samples")
            o.settings.filterSamples = std::stoi(value);
        else if (key == "--blocker-samples")
            o.settings.blockerSamples = std::stoi(value);
        else if (key == "--reference-samples")
            o.settings.referenceSamples = std::stoi(value);
        else if (key == "--warmup")
            o.warmup = std::stoi(value);
        else if (key == "--measure")
            o.measure = std::stoi(value);
        else if (key == "--angle")
            o.settings.angle = std::stof(value);
        else if (key == "--warp")
            o.settings.warp = std::stof(value);
        else if (key == "--camera")
            o.settings.camera = std::stoi(value);
        else if (key == "--scene")
            o.scene = value;
        else if (key == "--method")
            o.method = value;
        else if (key == "--output")
            o.output = value;
        else if (key == "--map-backend") {
            if (value != "raster" && value != "analytic")
                throw std::runtime_error("Unknown map backend");
            o.settings.analyticMaps = value == "analytic";
        } else
            throw std::runtime_error("Unknown option: " + key);
    }
    if (int(o.quick) + int(o.full) + int(o.smoke) > 1)
        throw std::runtime_error("Select one suite profile");
    if (o.headless && !o.benchmark)
        throw std::runtime_error("--headless requires an experiment mode");
    const auto& s = o.settings;
    if (s.width < 16 || s.width > 8192 || s.height < 16 || s.height > 8192 || s.mapResolution < 32 ||
        s.mapResolution > 8192 || s.filterSamples < 1 || s.filterSamples > 256 || s.blockerSamples < 1 ||
        s.blockerSamples > 256 || s.referenceSamples < 16 || s.referenceSamples > 16384 || s.angle <= 0 ||
        s.angle > 10 || s.warp < 0 || s.warp > 16 || s.camera < 0 || s.camera > 1 || o.measure < 1 || o.warmup < 0)
        throw std::runtime_error("Argument outside supported range");
    (void)makeScene(o.scene);
    if (o.method != "all")
        (void)parseMethod(o.method);
    return o;
}
void runExperiment(Renderer& renderer, const Options& o) {
    std::filesystem::create_directories(o.output / "captures");
    const auto configs = cases(o);
    auto selected = methods(o);
    const int expected = int(configs.size() * selected.size());
    manifest(o.output / "run_manifest.json", o, expected, false);
    std::ofstream raw(o.output / "raw.csv"), timings(o.output / "timings.csv"),
        captures(o.output / "capture_manifest.csv"), convergence(o.output / "reference_convergence.csv");
    if (!raw || !timings || !captures || !convergence)
        throw std::runtime_error("Cannot create experiment outputs");
    raw << "case_id,scene,method,width,height,map_resolution,angle,camera,map_backend,map_count,map_bytes,shadow_"
           "passes,timing_samples,map_gpu_ms,shadow_gpu_ms,total_gpu_ms,rmse,penumbra_rmse,gradient_rmse,max_abs_error,"
           "receiver_pixels,penumbra_pixels,invalid_pixels,monotonic_max_brightening,monotonic_violation_fraction\n";
    timings << "case_id,sample,map_gpu_ms,shadow_gpu_ms,total_gpu_ms\n";
    captures << "case_id,kind,width,height,path\n";
    convergence << "config_id,lower_samples,reference_samples,rmse,max_abs_error,cpu_gpu_max_abs_error\n";
    int configIndex = 0, done = 0;
    for (const auto& c : configs) {
        renderer.prepare(makeScene(c.scene), c.settings);
        const std::string config = "c" + std::to_string(configIndex++);
        renderer.render(Method::Reference);
        Image reference = renderer.read();
        unsigned referenceTexture = renderer.copyResult();
        pfm(o.output / "captures" / (config + "-receiver.pfm"), reference.receiver, c.settings.width,
            c.settings.height);
        captures << config << ",receiver-mask," << c.settings.width << ',' << c.settings.height << ",captures/"
                 << config << "-receiver.pfm\n";
        const double cpuError = cpuCheck(renderer, reference);
        auto lower = c.settings;
        lower.referenceSamples = std::max(8, c.settings.referenceSamples / 2);
        renderer.prepare(makeScene(c.scene), lower);
        renderer.render(Method::Reference);
        auto lowerImage = renderer.read();
        const auto delta = compare(lowerImage, reference);
        convergence << config << ',' << lower.referenceSamples << ',' << c.settings.referenceSamples << ','
                    << delta.rmse << ',' << delta.maxError << ',' << cpuError << '\n';
        renderer.prepare(makeScene(c.scene), c.settings);
        // Rotate the order between configurations to reduce fixed-order thermal bias.
        std::rotate(selected.begin(), selected.begin() + 1, selected.end());
        for (Method method : selected) {
            const std::string id = config + "-" + methodId(method);
            for (int i = 0; i < (method == Method::Reference ? 0 : o.warmup); ++i)
                renderer.render(method);
            glFinish();
            std::vector<double> mapTimes, shadowTimes, totalTimes;
            const int samples = method == Method::Reference ? 1 : o.measure;
            for (int i = 0; i < samples; ++i) {
                auto t = renderer.render(method, -1, true);
                mapTimes.push_back(t.mapMs);
                shadowTimes.push_back(t.shadowMs);
                totalTimes.push_back(t.mapMs + t.shadowMs);
                timings << id << ',' << i << ',' << std::setprecision(10) << t.mapMs << ',' << t.shadowMs << ','
                        << t.mapMs + t.shadowMs << '\n';
            }
            const Image image = renderer.read();
            pfm(o.output / "captures" / (id + ".pfm"), image.visibility, c.settings.width, c.settings.height);
            captures << id << ",visibility-pfm," << c.settings.width << ',' << c.settings.height << ",captures/" << id
                     << ".pfm\n";
            for (int view = 0; view < 3; ++view) {
                const std::string name = id + (view == 0   ? "-shaded.png"
                                               : view == 1 ? "-visibility.png"
                                                           : "-difference.png");
                renderer.capture(o.output / "captures" / name, view, referenceTexture);
                captures << id << ','
                         << (view == 0   ? "shaded-png"
                             : view == 1 ? "visibility-png"
                                         : "difference-png")
                         << ',' << c.settings.width << ',' << c.settings.height << ",captures/" << name << '\n';
            }
            renderer.render(method, 0);
            const Image without = renderer.read();
            const auto m = compare(image, reference, &without);
            const int mapCount = (method == Method::Reference || method == Method::Ray16)         ? 0
                                 : (method == Method::SplitMin || method == Method::SplitProduct) ? 2
                                                                                                  : 1;
            raw << id << ',' << c.scene << ',' << methodId(method) << ',' << c.settings.width << ','
                << c.settings.height << ',' << c.settings.mapResolution << ',' << c.settings.angle << ','
                << c.settings.camera << ',' << (c.settings.analyticMaps ? "analytic" : "raster") << ',' << mapCount
                << ','
                << std::uint64_t(mapCount) * std::uint64_t(c.settings.mapResolution) *
                       std::uint64_t(c.settings.mapResolution) * 4u
                << ',' << (method == Method::Reference ? c.settings.referenceSamples + 1 : 1) << ',' << samples << ','
                << std::setprecision(10) << median(mapTimes) << ',' << median(shadowTimes) << ',' << median(totalTimes)
                << ',' << m.rmse << ',' << m.penumbraRmse << ',' << m.gradientRmse << ',' << m.maxError << ','
                << m.pixels << ',' << m.penumbraPixels << ',' << m.invalid << ',' << m.brightening << ','
                << m.brighteningRate << '\n';
            raw.flush();
            ++done;
            std::cout << '[' << done << '/' << expected << "] " << c.scene << " / " << methodId(method)
                      << " RMSE=" << m.rmse << " GPU=" << median(totalTimes) << " ms\n"
                      << std::flush;
            if (m.invalid)
                throw std::runtime_error("Non-finite or out-of-range visibility detected");
            if (cpuError > 0.025)
                throw std::runtime_error(
                    "GPU reference disagrees with independent CPU oracle; inspect reference_convergence.csv");
        }
        glDeleteTextures(1, &referenceTexture);
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
