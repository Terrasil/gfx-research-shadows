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
        self.assertEqual(len(analyzer.METHODS),12)
        self.assertEqual(set(analyzer.METHODS),{"hard","pcf","pcss","arr","geometric","dpcf","chs","geometric-layered","vsm","csm","esm","arr-triangulation"})


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


if __name__ == "__main__":
    unittest.main()
