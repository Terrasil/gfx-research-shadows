#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace shadows {
constexpr int maxCasters = 32;
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
};
inline std::vector<std::string> sceneIds() {
    return {"single-high", "single-low", "mixed-overlap", "crossing", "hidden-layer", "sloped", "dense"};
}
inline Scene makeScene(const std::string& id) {
    Scene s;
    s.id = id;
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
