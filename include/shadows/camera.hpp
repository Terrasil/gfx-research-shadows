#pragma once
#include <algorithm>
#include <cmath>
#include <gfx/research/orbit_camera.hpp>
#include "renderer.hpp"

namespace shadows {
class CameraController {
  public:
    void reset(const Scene& scene, int preset) {
        camera_.set_target(scene.target);
        camera_.set_yaw(-2.1f);
        camera_.set_pitch(preset == 0 ? 0.65f : 1.25f);
        camera_.set_distance(preset == 0 ? 16.0f : 14.0f);
    }
    void set(const Settings& settings) {
        camera_.set_target(settings.cameraTarget);
        camera_.set_yaw(settings.cameraYaw);
        camera_.set_pitch(settings.cameraPitch);
        camera_.set_distance(settings.cameraDistance);
    }
    void orbit(float dx, float dy) {
        camera_.yaw() -= dx * 0.006f;
        camera_.pitch() = std::clamp(camera_.pitch() + dy * 0.006f, -1.45f, 1.45f);
    }
    void zoom(float steps) {
        camera_.distance() = std::clamp(camera_.distance() * std::exp(-steps * 0.12f), 0.5f, 60.0f);
    }
    void pan(float dx, float dy, float viewportHeight) {
        const auto forward = glm::normalize(camera_.target() - camera_.position());
        const auto right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
        const auto up = glm::cross(right, forward);
        const float scale = 2.0f * camera_.distance() * std::tan(glm::radians(26.0f)) /
                            std::max(viewportHeight, 1.0f);
        camera_.target() += (-dx * right + dy * up) * scale;
    }
    void move(float rightAmount, float forwardAmount, float upAmount, float seconds, bool fast) {
        const auto forward = glm::normalize(camera_.target() - camera_.position());
        const auto right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
        auto delta = right * rightAmount + forward * forwardAmount + glm::vec3(0, 1, 0) * upAmount;
        if (glm::dot(delta, delta) > 0.0f)
            camera_.target() += glm::normalize(delta) * std::clamp(seconds, 0.0f, 0.1f) * (fast ? 18.0f : 5.0f);
    }
    void apply(Settings& settings) {
        settings.customCamera = true;
        settings.cameraTarget = camera_.target();
        settings.cameraYaw = camera_.yaw();
        settings.cameraPitch = camera_.pitch();
        settings.cameraDistance = camera_.distance();
    }
  private:
    gfx::research::OrbitCamera camera_;
};
} // namespace shadows
