#include <iostream>
#include <shadows/camera.hpp>
#include <stdexcept>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
int main() {
    try {
        shadows::CameraController camera;
        shadows::Settings initial, moved;
        const auto scene = shadows::makeScene("crossing");
        camera.reset(scene, 0);
        camera.apply(initial);
        camera.orbit(100, -30);
        camera.pan(40, 20, 720);
        camera.zoom(2);
        camera.move(1, 1, 0, 0.05f, false);
        camera.apply(moved);
        require(moved.customCamera && moved.cameraYaw != initial.cameraYaw, "Orbit was not exported to renderer settings");
        require(glm::distance(moved.cameraTarget, initial.cameraTarget) > 0.01f, "Pan and movement left camera fixed");
        require(moved.cameraDistance < initial.cameraDistance, "Zoom did not approach target");
        camera.orbit(0, 1e9f);
        camera.zoom(-1e6f);
        camera.apply(moved);
        require(std::isfinite(moved.cameraPitch) && moved.cameraPitch <= 1.45f, "Pitch reached a singular orientation");
        require(moved.cameraDistance == 60.0f, "Zoom exceeded supported far distance");
        camera.reset(scene, 0);
        camera.apply(moved);
        require(moved.cameraTarget == initial.cameraTarget && moved.cameraDistance == initial.cameraDistance &&
                moved.cameraYaw == initial.cameraYaw && moved.cameraPitch == initial.cameraPitch, "Reset changed preset");
        camera.reset(scene, 1);
        camera.apply(moved);
        require(moved.cameraPitch > initial.cameraPitch, "Second preset was not restored");
        std::cout << "Camera input, bounds and reset: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
