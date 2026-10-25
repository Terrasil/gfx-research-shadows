#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace shadows {
constexpr int maxCasters = 32;
constexpr int maxStressCasters = 10'000'000;
constexpr double pi = 3.14159265358979323846;

struct Box {
    glm::vec3 lo{}, hi{}, color{0.7f};
    int group = 0;
};
struct Scene {
    std::string id;
    std::vector<Box> boxes;
    glm::vec3 target{-1.4f, 1.0f, -0.8f};
    float receiverTilt = 0.0f;
    int stressObjects = 0;
    bool isStress() const { return id == "dense-stress"; }
    int casterCount() const { return isStress() ? stressObjects : int(boxes.size()); }
};
inline int stressGrid(int count) {
    if (count < 0 || count > maxStressCasters)
        throw std::runtime_error("Stress object count must be 0..10000000");
    return std::max(1, int(std::ceil(std::sqrt(double(count)))));
}
inline Box stressBox(int index, int count) {
    const int grid = stressGrid(count);
    if (index < 0 || index >= count)
        throw std::runtime_error("Stress instance index outside population");
    const int x = index % grid, z = index / grid, group = (x + z) % 2;
    const float cell = 12.0f / float(grid), width = cell * 0.28f;
    const glm::vec3 center{(float(x) + 0.5f) * cell - 6.0f, 0, (float(z) + 0.5f) * cell - 6.0f};
    return {center + glm::vec3{-width * 0.5f, 0.1f, -width * 0.5f},
            center + glm::vec3{width * 0.5f, group ? 3.6f : 0.75f, width * 0.5f},
            group ? glm::vec3{0.82f, 0.55f, 0.19f} : glm::vec3{0.18f, 0.46f, 0.75f}, group};
}
inline float casterNearDepth(const Scene& scene,const glm::mat4& lightVp,int droppedGroup=-1) {
    double nearest=1.0;
    const glm::dmat4 vp(lightVp);
    auto include=[&](const Box& box){
        for(int k=0;k<8;++k){
            const glm::dvec3 p{(k&1)?box.hi.x:box.lo.x,(k&2)?box.hi.y:box.lo.y,(k&4)?box.hi.z:box.lo.z};
            const auto clip=vp*glm::dvec4(p,1);
            nearest=std::min(nearest,(clip.z/clip.w)*0.5+0.5);
        }
    };
    if(scene.isStress()&&scene.casterCount()>maxCasters){
        // Conservative field bound, O(1) even for ten million procedural boxes.
        include(Box{{-6,0.1f,-6},{6,droppedGroup==1?0.75f:3.6f,6},{},0});
    }else for(const auto& box:scene.boxes)if(box.group!=droppedGroup)include(box);
    if(nearest>=1.0)return 1.0f;
    // Round toward the light so the bound cannot exclude a stored surface.
    return std::nextafter(float(std::clamp(nearest,0.0,1.0)),0.0f);
}
inline std::vector<std::string> sceneIds(bool includeStress = true) {
    std::vector<std::string> ids = {"single-high", "single-low", "mixed-overlap", "crossing", "hidden-layer", "sloped", "dense", "crossing-four"};
    if (includeStress) ids.push_back("dense-stress");
    return ids;
}
inline Scene makeScene(const std::string& id, int objects = 4096) {
    Scene s;
    s.id = id;
    if (s.isStress()) {
        (void)stressGrid(objects);
        s.stressObjects = objects;
        s.target = {0, 1, 0};
        // Only small populations need an explicit list for the CPU/AABB oracle.
        // Large scenes remain O(1) in CPU memory; the vertex shader generates them.
        if (objects <= maxCasters)
            for (int i = 0; i < objects; ++i) s.boxes.push_back(stressBox(i, objects));
        return s;
    }
    const glm::vec3 gold{0.82f, 0.55f, 0.19f}, blue{0.18f, 0.46f, 0.75f};
    auto add = [&](glm::vec3 lo, glm::vec3 hi, int group) { s.boxes.push_back({lo, hi, group ? gold : blue, group}); };
    if (id == "single-high" || id == "mixed-overlap" || id == "sloped")
        add({-1.8f, 4.5f, -1.6f}, {1.8f, 5.0f, 1.6f}, 1);
    if (id == "single-low" || id == "mixed-overlap" || id == "sloped")
        add({-5.2f, 0.15f, -3.15f}, {-0.6f, 0.70f, -2.75f}, 0);
    if (id == "crossing") {
        add({-3, 4.5f, -0.5f}, {3, 4.8f, 0.5f}, 1);
        add({-2.6f, 0.2f, -4}, {-2.0f, 0.7f, 1.6f}, 0);
    }
    if (id == "crossing-four") {
        // Four separate heights, alternating X/Z beams. Their projected shadows
        // cross at (-2.5, 0, -2), and the central light ray meets all four beams.
        const glm::vec3 colors[] = {blue, {0.25f, 0.66f, 0.42f}, gold, {0.65f, 0.35f, 0.72f}};
        const float heights[] = {0.5f, 1.8f, 3.3f, 5.0f};
        for (int i = 0; i < 4; ++i) {
            const float h = heights[i];
            const glm::vec3 center{-2.5f + 0.55f * h, h, -2.0f + 0.35f * h};
            const glm::vec3 halfSize = i % 2 == 0 ? glm::vec3{3.0f, 0.15f, 0.22f}
                                                 : glm::vec3{0.22f, 0.15f, 3.0f};
            s.boxes.push_back({center - halfSize, center + halfSize, colors[i], i < 2 ? 0 : 1});
        }
        s.target = {-1.5f, 1.5f, -1.0f};
    }
    if (id == "hidden-layer") {
        add({-0.6f, 4.1f, -0.5f}, {0.6f, 4.6f, 0.5f}, 1);
        add({-3.0f, 0.15f, -2.4f}, {-1.7f, 0.7f, -0.6f}, 0);
    }
    if (id == "dense") {
        for (int z = 0; z < 4; ++z)
            for (int x = 0; x < 4; ++x) {
                const float h = ((x + z) % 2) ? 3.5f : 0.65f;
                glm::vec3 p{float(x) * 1.6f - 2.5f, 0.1f, float(z) * 1.6f - 2.5f};
                add(p, p + glm::vec3{0.45f, h, 0.45f}, (x + z) % 2);
            }
    }
    if (id == "sloped")
        s.receiverTilt = glm::radians(9.0f);
    if (s.boxes.empty())
        throw std::runtime_error("Unknown scene: " + id);
    return s;
}
inline double radicalInverse(unsigned n) {
    double v = 0, scale = 0.5;
    while (n) {
        v += double(n & 1u) * scale;
        n >>= 1u;
        scale *= 0.5;
    }
    return v;
}
inline glm::dvec2 angularDisk(unsigned index) {
    const double r = std::sqrt(radicalInverse(index + 1u));
    const double a = 2 * pi * std::fmod((double(index) + 0.5) * 0.6180339887498948482, 1.0);
    return {r * std::cos(a), r * std::sin(a)};
}
inline glm::dvec3 sunDirection() {
    return glm::normalize(glm::dvec3(0.55, 1.0, 0.35));
}
struct AngularSample {
    glm::dvec3 direction;
    double weight;
};
inline AngularSample sourceSample(unsigned index, double angularRadiusDegrees) {
    const auto l = sunDirection();
    const auto u = glm::normalize(glm::cross(glm::dvec3(0, 0, 1), l));
    const auto v = glm::cross(l, u);
    const auto disk = angularDisk(index) * std::tan(angularRadiusDegrees * pi / 180.0);
    return {glm::normalize(l + u * disk.x + v * disk.y), std::pow(1.0 + glm::dot(disk, disk), -1.5)};
}
inline bool rayBox(const glm::dvec3& o, const glm::dvec3& d, const Box& box, double& hit) {
    double near = 0, far = std::numeric_limits<double>::infinity();
    for (int k = 0; k < 3; ++k) {
        if (std::abs(d[k]) < 1e-14) {
            if (o[k] < box.lo[k] || o[k] > box.hi[k])
                return false;
        } else {
            double a = (double(box.lo[k]) - o[k]) / d[k], b = (double(box.hi[k]) - o[k]) / d[k];
            if (a > b)
                std::swap(a, b);
            near = std::max(near, a);
            far = std::min(far, b);
            if (near > far)
                return false;
        }
    }
    hit = near;
    return far > 1e-8;
}
inline double cpuVisibility(const Scene& scene, glm::dvec3 position, glm::dvec3 normal, int samples, double angle,
                            int droppedGroup = -1, double offset = 0.002) {
    if (scene.isStress() && scene.casterCount() > maxCasters)
        throw std::runtime_error("CPU reference is limited to 32 stress objects");
    double lit = 0, total = 0;
    position += normal * offset;
    for (int i = 0; i < samples; ++i) {
        const auto a = sourceSample(unsigned(i), angle);
        const double w = a.weight * std::max(glm::dot(normal, a.direction), 0.0);
        bool blocked = false;
        for (const auto& b : scene.boxes) {
            if (b.group == droppedGroup)
                continue;
            double hit = 0;
            if (rayBox(position, a.direction, b, hit)) {
                blocked = true;
                break;
            }
        }
        total += w;
        if (!blocked)
            lit += w;
    }
    return total > 1e-14 ? lit / total : 1.0;
}
} // namespace shadows
