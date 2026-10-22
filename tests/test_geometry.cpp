#include <iostream>
#include <shadows/scene.hpp>
#include <stdexcept>
using namespace shadows;
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
int main() {
    try {
        Box b{{-1, -1, -1}, {1, 1, 1}, {1, 1, 1}, 0};
        double t = 0;
        require(rayBox({0, 0, -3}, {0, 0, 1}, b, t) && std::abs(t - 2) < 1e-12, "Front-face intersection");
        require(!rayBox({2, 0, -3}, {0, 0, 1}, b, t), "Parallel ray outside slab");
        require(!rayBox({0, 0, -3}, {0, 0, -1}, b, t), "Box behind ray origin");
        require(rayBox({0, 0, 0}, {0, 1, 0}, b, t), "Origin inside solid box");
        for (unsigned i = 0; i < 2048; ++i) {
            const auto d = angularDisk(i);
            require(glm::dot(d, d) <= 1.0, "Angular sample outside disk");
            auto s = sourceSample(i, 3);
            require(std::abs(glm::length(s.direction) - 1) < 1e-12 && s.weight > 0, "Invalid angular sample");
        }
        Scene empty;
        empty.id = "empty";
        require(cpuVisibility(empty, {0, 0, 0}, {0, 1, 0}, 128, 1.5) == 1.0, "Empty scene must be lit");
        Scene ceiling;
        ceiling.id = "ceiling";
        ceiling.boxes.push_back({{-5, 1, -5}, {5, 2, 5}, {1, 1, 1}, 0});
        require(cpuVisibility(ceiling, {0, 0, 0}, {0, 1, 0}, 128, 1.5) == 0.0, "Opaque ceiling must block source");
        for (const auto& name : sceneIds()) {
            const auto scene = makeScene(name);
            for (int x = -8; x <= 8; ++x) {
                glm::dvec3 p{double(x) * 0.5, 0, -1.5};
                double all = cpuVisibility(scene, p, {0, 1, 0}, 128, 3),
                       without = cpuVisibility(scene, p, {0, 1, 0}, 128, 3, 0);
                require(all <= without + 1e-12, "Added opaque caster increased reference visibility");
            }
        }
        std::cout << "Geometry and reference invariants: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
