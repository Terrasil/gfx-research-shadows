#pragma once
#include "renderer.hpp"
#include <filesystem>
#include <string>
namespace shadows {
struct Options {
    Settings settings;
    bool benchmark = false, quick = false, full = false, smoke = false, headless = false;
    std::string scene = "mixed-overlap", method = "ndc-pcss";
    std::filesystem::path output = "results/manual";
    int warmup = 12, measure = 30;
};
Options parseOptions(int argc, char** argv);
void printHelp();
void runExperiment(Renderer&, const Options&);
int runHeadlessEgl(const Options&);
} // namespace shadows
