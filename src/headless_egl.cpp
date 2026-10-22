#include <dlfcn.h>
#include <glad/gl.h>
#include <iostream>
#include <shadows/experiment.hpp>
#include <stdexcept>

namespace shadows {
namespace {
using Handle = void*;
using Proc = void (*)();
using GetProc = Proc (*)(const char*);
GetProc eglGetProc = nullptr;
GLADapiproc loadGl(const char* name) {
    return reinterpret_cast<GLADapiproc>(eglGetProc(name));
}
template <class T> T symbol(void* library, const char* name) {
    auto p = dlsym(library, name);
    if (!p)
        throw std::runtime_error(std::string("Missing EGL symbol: ") + name);
    return reinterpret_cast<T>(p);
}
struct Context {
    void* library = nullptr;
    Handle display = nullptr, context = nullptr, surface = nullptr;
    unsigned (*makeCurrent)(Handle, Handle, Handle, Handle) = nullptr;
    unsigned (*destroyContext)(Handle, Handle) = nullptr;
    unsigned (*destroySurface)(Handle, Handle) = nullptr;
    unsigned (*terminate)(Handle) = nullptr;
    ~Context() {
        if (display && makeCurrent)
            makeCurrent(display, nullptr, nullptr, nullptr);
        if (display && context && destroyContext)
            destroyContext(display, context);
        if (display && surface && destroySurface)
            destroySurface(display, surface);
        if (display && terminate)
            terminate(display);
        if (library)
            dlclose(library);
    }
    void create() {
        library = dlopen("libEGL.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!library)
            throw std::runtime_error("Linux --headless requires libEGL.so.1");
        eglGetProc = symbol<GetProc>(library, "eglGetProcAddress");
        using PlatformDisplay = Handle (*)(unsigned, void*, const int*);
        auto platformDisplay = reinterpret_cast<PlatformDisplay>(eglGetProc("eglGetPlatformDisplayEXT"));
        if (platformDisplay)
            display = platformDisplay(0x31DD, nullptr, nullptr);
        if (!display)
            display = symbol<Handle (*)(void*)>(library, "eglGetDisplay")(nullptr);
        auto initialize = symbol<unsigned (*)(Handle, int*, int*)>(library, "eglInitialize");
        int major = 0, minor = 0;
        if (!display || !initialize(display, &major, &minor))
            throw std::runtime_error("Cannot initialize surfaceless EGL display");
        makeCurrent = symbol<decltype(makeCurrent)>(library, "eglMakeCurrent");
        destroyContext = symbol<decltype(destroyContext)>(library, "eglDestroyContext");
        destroySurface = symbol<decltype(destroySurface)>(library, "eglDestroySurface");
        terminate = symbol<decltype(terminate)>(library, "eglTerminate");
        if (!symbol<unsigned (*)(unsigned)>(library, "eglBindAPI")(0x30A2))
            throw std::runtime_error("Cannot bind desktop OpenGL API");
        const int attributes[] = {0x3033, 1, 0x3040, 8, 0x3024, 8, 0x3023, 8, 0x3022, 8, 0x3038};
        Handle config = nullptr;
        int count = 0;
        auto choose = symbol<unsigned (*)(Handle, const int*, Handle*, int, int*)>(library, "eglChooseConfig");
        if (!choose(display, attributes, &config, 1, &count) || count != 1)
            throw std::runtime_error("No EGL OpenGL configuration");
        const int size[] = {0x3057, 1, 0x3056, 1, 0x3038};
        surface =
            symbol<Handle (*)(Handle, Handle, const int*)>(library, "eglCreatePbufferSurface")(display, config, size);
        const int version[] = {0x3098, 4, 0x30FB, 6, 0x30FD, 1, 0x3038};
        context = symbol<Handle (*)(Handle, Handle, Handle, const int*)>(library, "eglCreateContext")(display, config,
                                                                                                      nullptr, version);
        if (!surface || !context || !makeCurrent(display, surface, surface, context))
            throw std::runtime_error("Cannot create OpenGL 4.6 EGL context");
        if (!gladLoadGL(loadGl))
            throw std::runtime_error("Cannot load OpenGL functions from EGL");
    }
};
} // namespace
int runHeadlessEgl(const Options& options) {
    try {
        Context context;
        context.create();
        Renderer renderer;
        renderer.initialize();
        runExperiment(renderer, options);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Headless experiment failed: " << e.what() << '\n';
        return 3;
    }
}
} // namespace shadows
