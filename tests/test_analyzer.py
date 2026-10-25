import importlib.util
import csv
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "tools/analyze_results.py"
spec = importlib.util.spec_from_file_location("analyzer", MODULE)
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


class AnalyzerRejectionTests(unittest.TestCase):
    def test_incomplete_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "run_manifest.json").write_text(json.dumps({"completed": False}))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                analyzer.analyze(root)

    def test_software_timings_rejected_for_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "run_manifest.json").write_text(json.dumps({"completed": True, "software_renderer": True}))
            with self.assertRaisesRegex(ValueError, "Software"):
                analyzer.analyze(root, require_hardware=True)

    def test_nonfinite_pfm_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            import struct
            path = Path(temp) / "bad.pfm"
            path.write_bytes(b"Pf\n1 1\n-1.0\n" + struct.pack("<f", float("nan")))
            with self.assertRaisesRegex(ValueError, "payload"):
                analyzer.read_pfm(path)

    def test_old_catalog_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "run_manifest.json").write_text(json.dumps({"completed":True,"version":"0.7.0","method_count":31}))
            with self.assertRaisesRegex(ValueError,"catalog"):
                analyzer.analyze(root)

    def test_catalog_contains_comparison_methods(self):
        self.assertEqual(len(analyzer.METHODS),14)
        self.assertEqual(set(analyzer.METHODS),{"hard","pcf","pcss","arr","geometric","dpcf","chs","geometric-layered","vsm","csm","esm","arr-triangulation","angular-rgba","gsm"})


class StressAnalysisTests(unittest.TestCase):
    def fixture(self, root):
        manifest = {"version": "0.16.0", "completed": True, "experiment_kind": "performance", "quality_metrics": False,
                    "quality_reference": None, "software_renderer": True, "object_counts": [1_000_000],
                    "configured_method": "hard", "expected_cases": 1, "measure": 1}
        row = {"case_id": "s0-hard", "scene": "dense-stress", "method": "hard", "objects": 1_000_000,
               "caster_base_triangles": 12_000_000, "gbuffer_cube_draws": 1, "map_cube_draws": 1,
               "instance_storage_bytes": 0, "map_bytes": 4096, "map_resolution": 32, "csm_terms": 16,
               "prefilter_passes": 0, "invalid_pixels": 0, "timing_samples": 1,
               "gbuffer_gpu_ms": 2, "map_gpu_ms": 3, "shadow_gpu_ms": 1, "total_gpu_ms": 6, "wall_ms": 7}
        (root / "run_manifest.json").write_text(json.dumps(manifest))
        self.write_csv(root / "raw.csv", row)
        self.write_csv(root / "timings.csv", {"case_id": "s0-hard", "sample": 0, **{k: row[k] for k in
                       ("gbuffer_gpu_ms", "map_gpu_ms", "shadow_gpu_ms", "total_gpu_ms", "wall_ms")}})
        return row

    def write_csv(self, path, row):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, list(row));writer.writeheader();writer.writerow(row)

    def test_performance_does_not_invent_quality(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);self.fixture(root)
            summary, _ = analyzer.analyze(root, validate_captures=False)
            self.assertFalse(summary["quality_metrics"])
            self.assertNotIn("mean_rmse", summary["methods"]["hard"])

    def test_incorrect_instance_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);row = self.fixture(root)
            row["caster_base_triangles"] = 12;self.write_csv(root / "raw.csv", row)
            with self.assertRaisesRegex(ValueError, "instance accounting"):
                analyzer.analyze(root, validate_captures=False)

    def test_gbuffer_must_be_included_in_total(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);self.fixture(root)
            self.write_csv(root / "timings.csv", {"case_id": "s0-hard", "sample": 0, "gbuffer_gpu_ms": 2,
                           "map_gpu_ms": 3, "shadow_gpu_ms": 1, "total_gpu_ms": 4, "wall_ms": 7})
            with self.assertRaisesRegex(ValueError, "scopes disagree"):
                analyzer.analyze(root, validate_captures=False)

    def test_arr_supports_full_million_population(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);row = self.fixture(root)
            manifest = json.loads((root / "run_manifest.json").read_text())
            manifest.update(version="0.16.1", configured_method="arr", arr_samples=4, tri_samples=4,
                            arr_smoothing=0, arr_smoothing_width=.75, tri_result_masks=0, stress_validation_cases=12)
            (root / "run_manifest.json").write_text(json.dumps(manifest))
            row.update(method="arr", map_cube_draws=0, map_bytes=0, angular_queries=4, angular_smoothing=0,
                       shadow_passes=1, screen_filter_bytes=0, angular_direction_bytes=262144, tri_result_masks=0)
            self.write_csv(root / "raw.csv", row)
            fields = ["objects", "instance", "mode", "expected_blocked", "gpu_blocked", "bounds_max_abs_error"]
            with (root / "stress_gpu.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fields);writer.writeheader()
                for count in (33, 65, 1024, 1_000_000):
                    for mode in range(3):
                        writer.writerow(dict(objects=count, instance=count-1, mode=mode, expected_blocked=int(mode==0), gpu_blocked=int(mode==0), bounds_max_abs_error=0))
            summary, _ = analyzer.analyze(root, validate_captures=False)
            self.assertIn("arr", summary["methods"])


class ContactBlendAccountingTests(unittest.TestCase):
    def fixture(self):
        manifest=dict(result_blend=.8,result_blend_radius=1,result_blend_sigma=.2,contact_guard=1,contact_max_gap=1)
        row=dict(method='geometric',width=320,height=240,result_blend=.8,result_blend_radius=1,
                 result_blend_sigma=.2,result_blend_bytes=320*240*4,contact_guard=1,contact_max_gap=1)
        return row,manifest

    def test_blend_memory_is_required(self):
        row,manifest=self.fixture()
        self.assertEqual(analyzer.verify_result_options(row,manifest),1)
        row['result_blend_bytes']=0
        with self.assertRaisesRegex(ValueError,'resource'):
            analyzer.verify_result_options(row,manifest)

    def test_angular_reference_does_not_inherit_pcss_blend(self):
        row,manifest=self.fixture()
        row.update(method='arr',result_blend=0,result_blend_radius=0,result_blend_sigma=0,
                   result_blend_bytes=0,contact_guard=0,contact_max_gap=0)
        self.assertEqual(analyzer.verify_result_options(row,manifest),0)

    def test_nonfinite_contact_metadata_is_rejected(self):
        row,manifest=self.fixture();row['contact_guard']=float('nan')
        with self.assertRaisesRegex(ValueError,'contact'):
            analyzer.verify_result_options(row,manifest)

    def test_rgba_joint_pass_requires_its_output_buffer(self):
        row,manifest=self.fixture()
        row.update(method='angular-rgba',contact_guard=0,contact_max_gap=0)
        manifest.update(version='0.23.0',rgba_pcf_mode=2,rgba_pcf_strength=.5,
                        rgba_pcf_radius=2,rgba_pcf_max_radius=16)
        self.assertEqual(analyzer.verify_result_options(row,manifest),1)
        row['result_blend_bytes']=0
        with self.assertRaisesRegex(ValueError,'resource'):
            analyzer.verify_result_options(row,manifest)

    def test_rgba_disabled_pcf_does_not_add_joint_pass(self):
        for mode,strength,cap in ((0,.5,16),(2,0,16),(2,.5,0)):
            row,manifest=self.fixture()
            row.update(method='angular-rgba',result_blend=0,result_blend_radius=0,result_blend_sigma=0,
                       result_blend_bytes=0,contact_guard=0,contact_max_gap=0)
            manifest.update(version='0.23.0',rgba_pcf_mode=mode,rgba_pcf_strength=strength,
                            rgba_pcf_radius=2,rgba_pcf_max_radius=cap)
            self.assertEqual(analyzer.verify_result_options(row,manifest),0)

    def test_shared_manual_radius_is_independent_of_adaptive_cap(self):
        row,manifest=self.fixture()
        row.update(method='angular-rgba',contact_guard=0,contact_max_gap=0)
        manifest.update(version='0.24.0',rgba_pcf_mode=3,rgba_pcf_strength=.5,
                        rgba_pcf_radius=2,rgba_pcf_max_radius=0,
                        rgba_shared_adaptive=0,rgba_shared_radius_m=.25)
        self.assertEqual(analyzer.verify_result_options(row,manifest),1)
        manifest['rgba_shared_adaptive']=1
        row.update(result_blend=0,result_blend_radius=0,result_blend_sigma=0,result_blend_bytes=0)
        self.assertEqual(analyzer.verify_result_options(row,manifest),0)


class MomentAccountingTests(unittest.TestCase):
    def fixture(self):
        m=dict(version='0.25.0',rgba_moments=1,rgba_moment_strength=.5,
               rgba_pcf_mode=2,rgba_pcf_samples=32,rgba_pcf_strength=.5,rgba_pcf_radius=2,rgba_pcf_max_radius=16,
               rgba_shared_adaptive=1,rgba_shared_radius_m=.25,rgba_offset=1,rgba_radius=1,
               rgba_mix_mode=0,rgba_cascades=0,rgba_angular_mix=1)
        r={k:v for k,v in m.items() if k.startswith('rgba_')}
        r.update(method='angular-rgba',rgba_shared_adaptive=0,rgba_shared_radius_m=0)
        return r,m

    def test_moment_metadata_cannot_claim_disabled_reconstruction(self):
        r,m=self.fixture();analyzer.verify_rgba_pcf_options(r,m)
        m['rgba_moment_strength']=0
        with self.assertRaisesRegex(ValueError,'moment'):
            analyzer.verify_rgba_pcf_options(r,m)
        r.update(rgba_moments=0,rgba_moment_strength=0)
        analyzer.verify_rgba_pcf_options(r,m)

    def test_artistic_mix_does_not_enable_physical_reconstruction(self):
        r,m=self.fixture();m['rgba_mix_mode']=1
        with self.assertRaisesRegex(ValueError,'moment'):
            analyzer.verify_rgba_pcf_options(r,m)
        r.update(rgba_moments=0,rgba_moment_strength=0)
        analyzer.verify_rgba_pcf_options(r,m)


class FixedWorldMixAccountingTests(unittest.TestCase):
    def fixture(self):
        r,m=MomentAccountingTests.fixture(self)
        m.update(version='0.26.0',rgba_pcf_mode=1,rgba_pcf_world_units=1,rgba_pcf_world_radius_m=.25,
                 rgba_geometric_blend=.25)
        r.update(rgba_pcf_mode=1,rgba_pcf_world_units=1,rgba_pcf_world_radius_m=.25,rgba_geometric_blend=0)
        return r,m

    def test_fixed_world_moments_are_effective(self):
        r,m=self.fixture();analyzer.verify_rgba_pcf_options(r,m)

    def test_zero_world_radius_cannot_claim_active_moments(self):
        r,m=self.fixture();m['rgba_pcf_world_radius_m']=0
        with self.assertRaisesRegex(ValueError,'moment'):analyzer.verify_rgba_pcf_options(r,m)
        r.update(rgba_moments=0,rgba_moment_strength=0,rgba_pcf_world_radius_m=0)
        analyzer.verify_rgba_pcf_options(r,m)

    def test_hybrid_cannot_claim_moment_reconstruction(self):
        r,m=self.fixture();m['rgba_mix_mode']=4
        with self.assertRaisesRegex(ValueError,'moment'):analyzer.verify_rgba_pcf_options(r,m)
        r.update(rgba_moments=0,rgba_moment_strength=0,rgba_mix_mode=4,rgba_geometric_blend=.25)
        analyzer.verify_rgba_pcf_options(r,m)
        r['rgba_geometric_blend']=.5
        with self.assertRaisesRegex(ValueError,'mixing'):analyzer.verify_rgba_pcf_options(r,m)

    def test_other_method_does_not_export_angular_settings(self):
        r,m=self.fixture();r.update(method='arr')
        with self.assertRaisesRegex(ValueError,'PCF'):analyzer.verify_rgba_pcf_options(r,m)
        for key in r:
            if key.startswith('rgba_'):r[key]=0
        analyzer.verify_rgba_pcf_options(r,m)

    def test_world_radius_controls_joint_buffer_accounting(self):
        r,m=ContactBlendAccountingTests.fixture(self)
        r.update(method='angular-rgba',contact_guard=0,contact_max_gap=0)
        m.update(version='0.26.0',rgba_pcf_mode=1,rgba_pcf_strength=.5,rgba_pcf_world_units=1,
                 rgba_pcf_world_radius_m=.25,rgba_pcf_radius=0)
        self.assertEqual(analyzer.verify_result_options(r,m),1)
        m['rgba_pcf_world_radius_m']=0;m['rgba_pcf_radius']=2
        r.update(result_blend=0,result_blend_radius=0,result_blend_sigma=0,result_blend_bytes=0)
        self.assertEqual(analyzer.verify_result_options(r,m),0)


class PerChannelPcssAccountingTests(unittest.TestCase):
    def fixture(self):
        r,m=FixedWorldMixAccountingTests.fixture(self)
        m.update(version='0.27.0',rgba_pcf_mode=4,rgba_pcss_blocker_samples=32,rgba_pcss_mean=0,rgba_pcss_search_radius_m=3)
        r.update(rgba_pcf_mode=4,rgba_pcf_world_units=0,rgba_pcf_world_radius_m=0,
                 rgba_pcss_blocker_samples=32,rgba_pcss_mean=0,rgba_pcss_search_radius_m=3)
        return r,m

    def test_pcss_reuses_blockers_for_moments(self):
        r,m=self.fixture();analyzer.verify_rgba_pcf_options(r,m)

    def test_blocker_statistic_is_independent_of_result_mixing(self):
        r,m=self.fixture();m['rgba_pcss_mean']=1
        with self.assertRaisesRegex(ValueError,'PCSS'):analyzer.verify_rgba_pcf_options(r,m)
        r['rgba_pcss_mean']=1;analyzer.verify_rgba_pcf_options(r,m)

    def test_legacy_filter_cannot_claim_pcss_blocker_settings(self):
        r,m=self.fixture();m['rgba_pcf_mode']=2;r['rgba_pcf_mode']=2
        with self.assertRaisesRegex(ValueError,'PCSS'):analyzer.verify_rgba_pcf_options(r,m)
        r.update(rgba_pcss_blocker_samples=0,rgba_pcss_mean=0,rgba_pcss_search_radius_m=0)
        analyzer.verify_rgba_pcf_options(r,m)

    def test_disabled_pcss_cannot_claim_active_moments(self):
        r,m=self.fixture();m['rgba_pcf_max_radius']=0;r['rgba_pcf_max_radius']=0
        with self.assertRaisesRegex(ValueError,'moment'):analyzer.verify_rgba_pcf_options(r,m)
        r.update(rgba_moments=0,rgba_moment_strength=0);analyzer.verify_rgba_pcf_options(r,m)

    def test_pcss_metadata_rejects_unsupported_budget_and_search(self):
        _,m=self.fixture()
        m.update(rgba_pcf_sharp=1,rgba_reconstruct=0,rgba_edge_search=64,rgba_reconstruct_strength=.5,rgba_edge_contact=1,
                 rgba_extent=14,rgba_cascade_scale=4,rgba_cascade_blend=0)
        analyzer.verify_rgba_options(m)
        for key,value in [('rgba_pcss_blocker_samples',0),('rgba_pcss_blocker_samples',129),('rgba_pcss_blocker_samples',1.5),
                          ('rgba_pcss_mean',2),('rgba_pcss_search_radius_m',-.1),('rgba_pcss_search_radius_m',20.1),
                          ('rgba_pcss_search_radius_m',float('nan')),('rgba_pcss_search_radius_m',float('inf'))]:
            with self.subTest(key=key,value=value):
                with self.assertRaisesRegex(ValueError,'PCSS'):analyzer.verify_rgba_options(dict(m,**{key:value}))


class CascadeCorrectionAccountingTests(unittest.TestCase):
    def fixture(self):
        r,m=PerChannelPcssAccountingTests.fixture(self)
        m.update(version='0.28.0',rgba_pcss_bounds=1,rgba_stable_focus=1)
        r.update(rgba_pcss_bounds=1,rgba_stable_focus=1)
        return r,m

    def test_bounded_search_is_reported(self):
        r,m=self.fixture();analyzer.verify_rgba_pcf_options(r,m)
        m['rgba_pcss_bounds']=0
        with self.assertRaisesRegex(ValueError,'corrections'):analyzer.verify_rgba_pcf_options(r,m)
        r['rgba_pcss_bounds']=0;analyzer.verify_rgba_pcf_options(r,m)

    def test_focus_setting_cannot_be_silently_changed(self):
        r,m=self.fixture();m['rgba_stable_focus']=0
        with self.assertRaisesRegex(ValueError,'corrections'):analyzer.verify_rgba_pcf_options(r,m)
        r['rgba_stable_focus']=0;analyzer.verify_rgba_pcf_options(r,m)

    def test_non_pcss_filter_has_no_caster_bound_setting(self):
        r,m=self.fixture();m['rgba_pcf_mode']=2;r['rgba_pcf_mode']=2
        r.update(rgba_pcss_blocker_samples=0,rgba_pcss_mean=0,rgba_pcss_search_radius_m=0)
        with self.assertRaisesRegex(ValueError,'corrections'):analyzer.verify_rgba_pcf_options(r,m)
        r['rgba_pcss_bounds']=0;analyzer.verify_rgba_pcf_options(r,m)


class MultiViewStorageAccountingTests(unittest.TestCase):
    def test_storage_mismatch_cannot_masquerade_as_array(self):
        manifest = dict(version='0.29.0', multiview_storage='array')
        row = dict(method='angular-rgba', multiview_storage='rgba', map_texture_count=1)
        with self.assertRaisesRegex(ValueError, 'differs'):
            analyzer.multiview_storage(row, manifest)
        row['multiview_storage'] = 'array'
        self.assertEqual(analyzer.multiview_storage(row, manifest), 'd32f-array-angular')

    def test_four_separate_maps_cannot_claim_one_texture(self):
        manifest = dict(version='0.29.0', multiview_storage='separate')
        row = dict(method='angular-rgba', multiview_storage='separate', map_texture_count=1)
        with self.assertRaisesRegex(ValueError, 'texture count'):
            analyzer.multiview_storage(row, manifest)
        row['map_texture_count'] = 4
        self.assertEqual(analyzer.multiview_storage(row, manifest), 'd32f-separate-angular')

    def test_arr_has_no_shadow_texture_under_any_storage_setting(self):
        manifest = dict(version='0.29.0', multiview_storage='separate')
        row = dict(method='arr', multiview_storage='none', map_texture_count=0)
        self.assertIsNone(analyzer.multiview_storage(row, manifest))
        row['map_texture_count'] = 4
        with self.assertRaisesRegex(ValueError, 'texture count'):
            analyzer.multiview_storage(row, manifest)

    def test_truncated_backend_validation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'multiview.csv'
            path.write_text('case_id,storage\n')
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                analyzer.verify_multiview_storage(path)


class ExplicitMultiViewVariantTests(unittest.TestCase):
    def test_all_four_requested_variants_are_distinct(self):
        variants = set()
        for storage in ('rgba', 'separate'):
            for cascades in (0, 1):
                manifest = dict(version='0.30.0', multiview_storage=storage, rgba_cascades=cascades)
                variant = analyzer.multiview_variant(manifest)
                variants.add(variant)
                manifest['multiview_variant'] = variant
                row = dict(method='angular-rgba', multiview_storage=storage, multiview_variant=variant,
                           map_texture_count=1 if storage=='rgba' else 4)
                analyzer.multiview_storage(row, manifest)
        self.assertEqual(variants, {'rgba', 'separate', 'rgba-cascaded', 'separate-cascaded'})

    def test_misreported_cascade_variant_is_rejected(self):
        manifest = dict(version='0.30.0', multiview_storage='separate', rgba_cascades=1,
                        multiview_variant='separate-cascaded')
        row = dict(method='angular-rgba', multiview_storage='separate', multiview_variant='separate', map_texture_count=4)
        with self.assertRaisesRegex(ValueError, 'explicit'):
            analyzer.multiview_storage(row, manifest)


class GsmMapAccountingTests(unittest.TestCase):
    def test_gsm_has_separate_parameters_from_pcss(self):
        m=dict(version='0.31.0',gsm_sharpness=64,gsm_prefilter_radius=4)
        r=dict(method='gsm',gsm_sharpness=64,gsm_prefilter_radius=4)
        analyzer.verify_gsm_options(r,m)
        r['method']='geometric'
        with self.assertRaisesRegex(ValueError,'effective GSM'):
            analyzer.verify_gsm_options(r,m)
        r.update(gsm_sharpness=0,gsm_prefilter_radius=0)
        analyzer.verify_gsm_options(r,m)

    def test_invalid_gsm_map_controls_are_rejected(self):
        r=dict(method='gsm',gsm_sharpness=64,gsm_prefilter_radius=4)
        for key,value in [('gsm_sharpness',0),('gsm_sharpness',129),('gsm_sharpness',float('nan')),
                          ('gsm_prefilter_radius',17),('gsm_prefilter_radius',-1),('gsm_prefilter_radius',1.5)]:
            m=dict(version='0.31.0',gsm_sharpness=64,gsm_prefilter_radius=4,**{})
            m[key]=value
            with self.subTest(key=key,value=value):
                with self.assertRaisesRegex(ValueError,'GSM'):
                    analyzer.verify_gsm_options(r,m)


if __name__ == "__main__":
    unittest.main()
