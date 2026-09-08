import json
import tempfile
import unittest
from pathlib import Path

from hg_mc_auto.quality_control import (
    build_qc_packet,
    packet_from_result_file,
    normalize_qc_result,
    validate_llm_assessment,
)
from hg_mc_auto.monitor import changed_result_files


class TestQualityControl(unittest.TestCase):
    def test_packet_contains_metrics_flags_and_human_gate(self):
        packet = build_qc_packet(
            element="hg",
            run_id="run-001",
            metrics={"internal_precision": 0.12, "data_quality_score": 0.42},
            results={"delta202": -0.8},
            flags=[{"code": "LOW_QUALITY_SCORE", "severity": "warning"}],
        )

        payload = json.loads(packet.to_json())
        self.assertEqual(payload["schema_version"], "qc.v1")
        self.assertEqual(payload["element"], "hg")
        self.assertEqual(payload["human_confirmation"]["required"], True)
        self.assertEqual(payload["flags"][0]["code"], "LOW_QUALITY_SCORE")

    def test_normalize_qc_result_rejects_unknown_severity(self):
        with self.assertRaises(ValueError):
            normalize_qc_result({"code": "X", "severity": "urgent"})

    def test_llm_assessment_is_advisory_only(self):
        assessment = validate_llm_assessment(
            {
                "summary": "Review precision",
                "anomalies": [{"flag_code": "LOW_QUALITY_SCORE", "cause": "unstable signal"}],
                "recommendations": ["Inspect standard and amplifier stability"],
                "confidence": 0.7,
            }
        )
        self.assertTrue(assessment["human_confirmation_required"])
        self.assertNotIn("execute", assessment)

    def test_result_file_adapter_maps_hg_prediction_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "internal_combined_results.json"
            path.write_text(json.dumps({
                "run_id": "run-002",
                "Data_Quality_Score": 0.41,
                "Total_StdErr": 0.22,
                "ML_Predicted_Cause": "unstable signal",
                "ML_Prediction_Confidence": 0.88,
                "delta202": -0.4,
            }), encoding="utf-8")

            packet = packet_from_result_file(path, element="hg")

        self.assertEqual(packet.run_id, "run-002")
        self.assertEqual(packet.metrics["data_quality_score"], 0.41)
        flag_codes = {flag.code for flag in packet.flags}
        self.assertIn("LOW_QUALITY_SCORE", flag_codes)
        self.assertIn("ML_ANOMALY_PREDICTION", flag_codes)
        self.assertTrue(all(flag.severity == "warning" for flag in packet.flags))

    def test_result_file_adapter_rejects_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.txt"
            path.write_text("not supported", encoding="utf-8")
            with self.assertRaises(ValueError):
                packet_from_result_file(path, element="hg")

    def test_monitor_detects_new_result_file_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text("{}", encoding="utf-8")
            seen = {}
            self.assertEqual(changed_result_files(directory, seen), [path])
            self.assertEqual(changed_result_files(directory, seen), [])


if __name__ == "__main__":
    unittest.main()