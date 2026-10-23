import importlib.util
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

    def test_catalog_has_only_seven_methods(self):
        self.assertEqual(len(analyzer.METHODS),7)
        self.assertEqual(set(analyzer.METHODS),{"hard","pcf","pcss","arr","geometric","dpcf","chs"})


if __name__ == "__main__":
    unittest.main()
