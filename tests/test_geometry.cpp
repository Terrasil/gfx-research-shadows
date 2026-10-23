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
        const auto four = makeScene("crossing-four");
        require(four.boxes.size() == 4, "Four-beam scene must contain four separate casters");
        for (const auto& beam : four.boxes)
            require(rayBox({-2.5, 0, -2.0}, sunDirection(), beam, t),
                    "Central light ray must cross every height in the four-beam scene");
        empty.id = "empty";
        require(cpuVisibility(empty, {0, 0, 0}, {0, 1, 0}, 128, 1.5) == 1.0, "Empty scene must be lit");
        Scene ceiling;
        ceiling.id = "ceiling";
        ceiling.boxes.push_back({{-5, 1, -5}, {5, 2, 5}, {1, 1, 1}, 0});
        require(cpuVisibility(ceiling, {0, 0, 0}, {0, 1, 0}, 128, 1.5) == 0.0, "Opaque ceiling must block source");
        for (int count : {0, 1, 17, 32, 4096, 1'000'000, 10'000'000}) {
            const auto stress = makeScene("dense-stress", count);
            require(stress.casterCount() == count, "Stress population count changed");
            require(stress.boxes.size() == (count <= maxCasters ? std::size_t(count) : 0u), "Large stress scenes must not allocate per-object CPU data");
            if (count == 0) continue;
            const int grid = stressGrid(count);
            require(std::int64_t(grid) * grid >= count && std::int64_t(grid - 1) * (grid - 1) < count, "Stress grid does not cover exact population");
            for (int i : {0, count / 2, count - 1}) {
                const auto box = stressBox(i, count);
                require(box.lo.x >= -6.001f && box.hi.x <= 6.001f && box.lo.z >= -6.001f && box.hi.z <= 6.001f, "Stress caster outside fixed field");
                require(box.lo.x < box.hi.x && box.lo.y < box.hi.y && box.lo.z < box.hi.z, "Degenerate stress caster");
            }
        }
        bool referenceRejected = false;
        try { (void)cpuVisibility(makeScene("dense-stress", 1000), {0, 0, 0}, {0, 1, 0}, 16, 1); }
        catch (const std::runtime_error&) { referenceRejected = true; }
        require(referenceRejected, "Large procedural populations must not silently use an empty reference");
        for (const auto& name : sceneIds(false)) {
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
