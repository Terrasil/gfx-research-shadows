#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <glad/gl.h>
#include <iomanip>
#include <iostream>
#include <shadows/experiment.hpp>
#include <stdexcept>

namespace shadows {
namespace {
double middle(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto n = values.size();
    return n % 2 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}
void performanceManifest(const Options& o, const std::vector<int>& counts, int cases, bool completed) {
    const std::string gpu = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    std::ofstream out(o.output / "run_manifest.json");
    out << "{\n\"version\":\"0.31.0\",\"experiment_kind\":\"performance\",\"profile\":\"stress\","
        << "\n\"completed\":" << (completed ? "true" : "false") << ",\"expected_cases\":" << cases
        << ",\n\"gpu\":" << std::quoted(gpu) << ",\"software_renderer\":"
        << (gpu.find("llvmpipe") != std::string::npos || gpu.find("softpipe") != std::string::npos ? "true" : "false")
        << ",\n\"method_count\":14,\"scene\":\"dense-stress\",\"object_counts\":[";
    for (std::size_t i = 0; i < counts.size(); ++i) out << (i ? "," : "") << counts[i];
    out << "],\n\"width\":" << o.settings.width << ",\"height\":" << o.settings.height
        << ",\"map_resolution\":" << o.settings.mapResolution << ",\"warmup\":" << o.warmup << ",\"measure\":" << o.measure
        << ",\"warp\":" << o.settings.warp << ",\"angle\":" << o.settings.angle
        << ",\"stress_tessellation\":" << o.settings.stressTessellation
        << ",\"adaptive_tessellation\":" << (o.settings.stressAdaptiveTessellation ? "true" : "false")
        << ",\"configured_method\":" << std::quoted(o.method)
        << ",\"configured_sample_pattern\":" << std::quoted(samplePatternId(o.settings.samplePattern))
        << ",\"sample_seed\":" << o.settings.sampleSeed << ",\"blue_noise_tile_m\":" << o.settings.blueNoiseTile
        << ",\"result_blend\":" << o.settings.resultBlend << ",\"result_blend_radius\":" << o.settings.resultBlendRadius
        << ",\"result_blend_sigma\":" << o.settings.resultBlendSigma << ",\"contact_guard\":" << o.settings.contactGuard
        << ",\"configured_layer_depth\":" << std::quoted(o.settings.packedIntervals?"interval":"nearest")
        << ",\"contact_max_gap\":" << o.settings.contactMaxGap
        << ",\"gsm_sharpness\":" << o.settings.gsmSharpness
        << ",\"gsm_prefilter_radius\":" << o.settings.prefilterRadius
        << ",\"multiview_storage\":" << std::quoted(multiViewStorageId(o.settings.multiViewStorage))
        << ",\"multiview_variant\":" << std::quoted(multiViewVariantId(o.settings))
        << ",\"rgba_offset\":" << (o.settings.rgbaOffset?1:0)
        << ",\"rgba_radius\":" << o.settings.rgbaRadius
        << ",\"rgba_rotation\":" << o.settings.rgbaRotation
        << ",\"rgba_center_weight\":" << o.settings.rgbaCenterWeight
        << ",\"rgba_smoothing\":" << o.settings.rgbaSmoothing
        << ",\"rgba_cascades\":" << (o.settings.rgbaCascades?1:0)
        << ",\"rgba_extent\":" << o.settings.rgbaExtent
        << ",\"rgba_cascade_scale\":" << o.settings.rgbaCascadeScale
        << ",\"rgba_cascade_blend\":" << o.settings.rgbaCascadeBlend
        << ",\"rgba_angular_mix\":" << o.settings.rgbaAngularMix
        << ",\"rgba_mix_mode\":" << o.settings.rgbaMixMode
        << ",\"rgba_geometric_blend\":" << o.settings.rgbaGeometricBlend
        << ",\"rgba_reconstruct\":" << (o.settings.rgbaReconstruct?1:0)
        << ",\"rgba_reconstruct_strength\":" << o.settings.rgbaReconstructStrength
        << ",\"rgba_edge_search\":" << o.settings.rgbaEdgeSearch
        << ",\"rgba_edge_contact\":" << o.settings.rgbaEdgeContactRange
        << ",\"rgba_pcf_mode\":" << o.settings.rgbaPcfMode
        << ",\"rgba_pcf_samples\":" << o.settings.rgbaPcfSamples
        << ",\"rgba_pcf_strength\":" << o.settings.rgbaPcfStrength
        << ",\"rgba_pcf_radius\":" << o.settings.rgbaPcfRadius
        << ",\"rgba_pcf_world_units\":" << (o.settings.rgbaPcfWorldUnits?1:0)
        << ",\"rgba_pcf_world_radius_m\":" << o.settings.rgbaPcfWorldRadius
        << ",\"rgba_pcf_max_radius\":" << o.settings.rgbaPcfMaxRadius
        << ",\"rgba_pcf_sharp\":" << (o.settings.rgbaPcfSharp?1:0)
        << ",\"rgba_pcss_blocker_samples\":" << o.settings.rgbaPcssBlockerSamples
        << ",\"rgba_pcss_mean\":" << o.settings.rgbaPcssMean
        << ",\"rgba_pcss_search_radius_m\":" << o.settings.rgbaPcssSearchRadius
        << ",\"rgba_pcss_bounds\":" << (o.settings.rgbaPcssCasterBounds?1:0)
        << ",\"rgba_stable_focus\":" << (o.settings.rgbaStableFocus?1:0)
        << ",\"rgba_shared_adaptive\":" << (o.settings.rgbaSharedAdaptive?1:0)
        << ",\"rgba_shared_radius_m\":" << o.settings.rgbaSharedRadius
        << ",\"rgba_moments\":" << (o.settings.rgbaMoments?1:0)
        << ",\"rgba_moment_strength\":" << o.settings.rgbaMomentStrength
        << ",\"base_revision\":" << std::quoted(std::string(GFX_SHADOWS_BASE_REVISION))
        << ",\"quality_reference\":null,\"quality_metrics\":false,\"temporal_reuse\":false"
        << ",\"instance_storage_bytes\":0,\"field_width\":12,\"cpu_object_culling\":false"
        << ",\"excluded_from_all\":[],\"angular_scene_backend\":\"procedural full-population box scan\""
        << ",\"arr_samples\":" << angularSamples(Method::Reference,o.settings)
        << ",\"tri_samples\":" << o.settings.triSamples
        << ",\"arr_smoothing\":" << o.settings.arrSmoothing
        << ",\"arr_smoothing_width\":" << o.settings.arrSmoothingRadius
        << ",\"stress_validation_cases\":12,\"tri_signed_margin\":" << (o.settings.triSignedMargin ? 1 : 0)
        << ",\"tri_result_masks\":" << (o.settings.triResultMasks ? 1 : 0)
        << ",\"timing_scope\":\"GPU G-buffer + map generation/prefilter + shadow evaluation; excludes allocation, shader compilation, UI, present, captures\""
        << ",\"wall_scope\":\"prepare + render including CPU submission and synchronous GPU timestamp waits\"\n}\n";
    if (!out) throw std::runtime_error("Cannot write stress manifest");
}
} // namespace

void runStressBenchmark(Renderer& renderer, const Options& o) {
    const auto counts = o.objectCounts.empty() ? std::vector<int>{o.settings.stressObjects} : o.objectCounts;
    std::vector<Method> methods;
    if (o.method == "all") {
        for (int i = 0; i < methodCount; ++i) methods.push_back(Method(i));
    } else methods.push_back(parseMethod(o.method));
    std::filesystem::create_directories(o.output / "captures");
    renderer.validateStress(o.output / "stress_gpu.csv");
    const int expected = int(counts.size() * methods.size());
    performanceManifest(o, counts, expected, false);
    std::ofstream raw(o.output / "raw.csv"), timings(o.output / "timings.csv"), captures(o.output / "capture_manifest.csv");
    raw << "case_id,scene,method,objects,width,height,map_resolution,timing_samples,caster_base_triangles,gbuffer_cube_draws,map_cube_draws,instance_storage_bytes,map_bytes,map_storage,prefilter_passes,gbuffer_gpu_ms,map_gpu_ms,shadow_gpu_ms,total_gpu_ms,wall_ms,invalid_pixels,stress_tessellation,adaptive_tessellation,warp,angle,smoothing,filter_samples,blocker_samples,configured_sample_pattern,shared_sample_pattern,csm_terms,angular_queries,angular_smoothing,shadow_passes,angular_direction_bytes,screen_filter_bytes,tri_result_masks,tri_signed_margin,sample_pattern,blue_noise_tile_m,sampling_texture_bytes,result_blend,result_blend_radius,result_blend_sigma,result_blend_bytes,contact_guard,contact_max_gap,rgba_pcf_mode,rgba_pcf_samples,rgba_pcf_strength,rgba_pcf_radius,rgba_pcf_max_radius,rgba_shared_adaptive,rgba_shared_radius_m,rgba_moments,rgba_moment_strength,rgba_pcf_world_units,rgba_pcf_world_radius_m,rgba_mix_mode,rgba_geometric_blend,rgba_pcss_blocker_samples,rgba_pcss_mean,rgba_pcss_search_radius_m,rgba_pcss_bounds,rgba_stable_focus,multiview_storage,map_texture_count,multiview_variant,gsm_sharpness,gsm_prefilter_radius\n";
    timings << "case_id,sample,gbuffer_gpu_ms,map_gpu_ms,shadow_gpu_ms,total_gpu_ms,wall_ms\n";
    captures << "case_id,kind,width,height,path\n";
    if (!raw || !timings || !captures) throw std::runtime_error("Cannot create stress output files");
    int index = 0;
    for (int count : counts) {
        auto settings = o.settings;
        settings.stressObjects = count;
        const auto scene = makeScene("dense-stress", count);
        for (Method method : methods) {
            const std::string id = "s" + std::to_string(index++) + "-" + methodId(method);
            // Prime allocations and specialized shaders before measurement, even
            // when warmup=0. Warmup iterations include both geometry passes.
            renderer.prepare(scene, settings);
            renderer.render(method);
            for (int i = 0; i < o.warmup; ++i) {
                renderer.prepare(scene, settings);
                renderer.render(method);
            }
            glFinish();
            std::vector<double> gbuffer, maps, shadows, totals, walls;
            for (int i = 0; i < o.measure; ++i) {
                const auto start = std::chrono::steady_clock::now();
                const double g = renderer.prepare(scene, settings, true);
                const auto t = renderer.render(method, -1, true);
                const double total = g + t.mapMs + t.shadowMs;
                const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                gbuffer.push_back(g); maps.push_back(t.mapMs); shadows.push_back(t.shadowMs); totals.push_back(total); walls.push_back(wall);
                timings << id << ',' << i << ',' << std::setprecision(12) << g << ',' << t.mapMs << ',' << t.shadowMs << ',' << total << ',' << wall << '\n';
            }
            const auto image = renderer.read();
            int invalid = 0;
            for (float v : image.visibility) if (!std::isfinite(v) || v < -1e-5f || v > 1.00001f) ++invalid;
            if (invalid || glGetError() != GL_NO_ERROR) throw std::runtime_error("Invalid stress rendering or OpenGL error");
            raw << id << ",dense-stress," << methodId(method) << ',' << count << ',' << settings.width << ',' << settings.height << ','
                << settings.mapResolution << ',' << o.measure << ',' << 12ull * static_cast<unsigned long long>(count) << ','
                << (count > 0 ? 1 : 0) << ',' << (count > 0 && !isAngularRayMethod(method) ? mapGenerationPasses(method,settings) : 0) << ",0,"
                << activeMapBytes(method, settings) << ',' << mapStorageId(method,settings) << ','
                << prefilterPasses(method) << ',' << std::setprecision(12) << middle(gbuffer) << ',' << middle(maps) << ',' << middle(shadows) << ','
                << middle(totals) << ',' << middle(walls) << ',' << invalid << ',' << settings.stressTessellation << ','
                << (settings.stressAdaptiveTessellation ? 1 : 0) << ',' << settings.warp << ',' << settings.angle << ',' << settings.smoothing << ','
                << settings.filterSamples << ',' << settings.blockerSamples << ',' << samplePatternId(settings.samplePattern) << ','
                << (settings.sharedSamplePattern ? 1 : 0) << ',' << settings.csmTerms << ','
                << angularSamples(method, settings) << ',' << angularSmoothing(method, settings) << ',' << shadowPasses(method, settings) << ','
                << (method == Method::Reference ? 16384 * 16 : 0) << ','
                << (angularSmoothing(method, settings) ? std::uint64_t(settings.width) * settings.height * 8 : 0) << ','
                << (method == Method::ArrTriangulation && settings.triResultMasks ? 1 : 0) << ',' << (settings.triSignedMargin ? 1 : 0) << ',' << samplePatternId(effectiveSamplePattern(method,settings)) << ','
                << (effectiveSamplePattern(method,settings)==3?settings.blueNoiseTile:0.0f) << ',' << samplingTextureBytes(method,settings) << ',' << (usesResultBlend(method,settings)?settings.resultBlend:0.0f) << ','
                << (usesResultBlend(method,settings)?settings.resultBlendRadius:0) << ',' << (usesResultBlend(method,settings)?settings.resultBlendSigma:0.0f) << ','
                << resultBlendBytes(method,settings) << ',' << (usesPcss(method)?settings.contactGuard:0.0f) << ','
                << (usesPcss(method)&&settings.contactGuard>0?settings.contactMaxGap:0.0f) << ',' << (method==Method::AngularRgba&&settings.rgbaPcfMode!=0?settings.rgbaPcfMode:0) << ',' << (method==Method::AngularRgba&&settings.rgbaPcfMode!=0?settings.rgbaPcfSamples:0) << ',' << (method==Method::AngularRgba&&settings.rgbaPcfMode!=0?settings.rgbaPcfStrength:0) << ',' << (method==Method::AngularRgba&&settings.rgbaPcfMode!=0?settings.rgbaPcfRadius:0) << ',' << (method==Method::AngularRgba&&settings.rgbaPcfMode!=0?settings.rgbaPcfMaxRadius:0) << ',' << (method==Method::AngularRgba&&settings.rgbaPcfMode==3&&settings.rgbaSharedAdaptive?1:0) << ',' << (method==Method::AngularRgba&&settings.rgbaPcfMode==3?settings.rgbaSharedRadius:0) << ',' << (usesMomentReconstruction(method,settings)?1:0) << ',' << (usesMomentReconstruction(method,settings)?settings.rgbaMomentStrength:0)
                << ',' << ((method==Method::AngularRgba&&settings.rgbaPcfMode==1&&settings.rgbaPcfWorldUnits)?1:0) << ',' << ((method==Method::AngularRgba&&settings.rgbaPcfMode==1&&settings.rgbaPcfWorldUnits)?settings.rgbaPcfWorldRadius:0)
                << ',' << ((method==Method::AngularRgba&&(!settings.rgbaCascades||settings.rgbaAngularMix>0))?settings.rgbaMixMode:0) << ',' << ((method==Method::AngularRgba&&(!settings.rgbaCascades||settings.rgbaAngularMix>0))&&settings.rgbaMixMode==4?settings.rgbaGeometricBlend:0)
                << ',' << (method==Method::AngularRgba&&settings.rgbaPcfMode==4?settings.rgbaPcssBlockerSamples:0) << ',' << (method==Method::AngularRgba&&settings.rgbaPcfMode==4?settings.rgbaPcssMean:0)
                << ',' << (method==Method::AngularRgba&&settings.rgbaPcfMode==4?settings.rgbaPcssSearchRadius:0)
                << ',' << (method==Method::AngularRgba&&settings.rgbaPcfMode==4&&settings.rgbaPcssCasterBounds?1:0)
                << ',' << (method==Method::AngularRgba&&settings.rgbaStableFocus?1:0)
                << ',' << (method==Method::AngularRgba?multiViewStorageId(settings.multiViewStorage):"none")
                << ',' << mapTextureCount(method,settings)
                << ',' << (method==Method::AngularRgba?multiViewVariantId(settings):"none")
                << ',' << (method==Method::Gsm?settings.gsmSharpness:0.0f)
                << ',' << (method==Method::Gsm?settings.prefilterRadius:0) << '\n';
            for (int view = 0; view < 2; ++view) {
                const std::string kind = view == 0 ? "shaded" : "visibility", name = id + '-' + kind + ".png";
                renderer.capture(o.output / "captures" / name, view);
                captures << id << ',' << kind << ',' << settings.width << ',' << settings.height << ",captures/" << name << '\n';
            }
            raw.flush(); timings.flush(); captures.flush();
            std::cout << '[' << index << '/' << expected << "] " << count << " objects / " << methodId(method)
                      << " GPU components=" << middle(totals) << " ms\n" << std::flush;
        }
    }
    raw.close(); timings.close(); captures.close();
    performanceManifest(o, counts, expected, true);
}
} // namespace shadows
