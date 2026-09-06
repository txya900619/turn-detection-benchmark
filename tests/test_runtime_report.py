from __future__ import annotations

import json
from pathlib import Path

import pytest

from turn_eval.reports.runtime import runtime_profile_row, write_runtime_report


def _profile() -> dict[str, object]:
    return {
        "schema_version": 1,
        "profile_type": "isolated-cold-start-and-warm-streaming-ticks",
        "model": {
            "name": "tiny-streamer",
            "artifact_size_bytes": 1_048_576,
        },
        "execution": {
            "provider": "cpu",
            "threads": 2,
            "frame_interval_ms": 80.0,
        },
        "measurement": {
            "process_start_to_model_constructed_s": 1.0,
            "process_start_to_first_output_s": 1.1,
            "cold_first_tick_latency_ms": 100.0,
            "measurement_frames": 3,
            "warm_tick_latencies_ms": [1.0, 2.0, 3.0],
            "warm_inference_latency_ms": {
                "count": 3,
                "mean": 2.0,
                "p50": 2.0,
                "p90": 2.8,
                "p95": 2.9,
                "p99": 2.98,
                "minimum": 1.0,
                "maximum": 3.0,
            },
            "rtf": 0.025,
            "realtime_multiplier": 40.0,
        },
        "resources": {
            "cpu_utilization_percent": {"mean_over_profile": 150.0},
            "ram": {"peak_rss_bytes": 2_097_152},
            "gpu_vram": {"peak_bytes": None},
        },
    }


def test_runtime_report_recomputes_raw_trace_and_writes_all_formats(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(_profile()))

    report = write_runtime_report([profile], tmp_path / "report")
    row = runtime_profile_row(profile)

    assert report.is_file()
    assert row["warm_p95_ms"] == pytest.approx(2.9)
    assert "tiny-streamer" in report.read_text()
    assert (report.parent / "summary.json").is_file()
    assert (report.parent / "summary.csv").is_file()


def test_runtime_report_rejects_summary_not_matching_raw_trace(tmp_path: Path) -> None:
    value = _profile()
    value["measurement"]["warm_inference_latency_ms"]["p50"] = 999.0
    profile = tmp_path / "bad.json"
    profile.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="summary mismatch"):
        runtime_profile_row(profile)
