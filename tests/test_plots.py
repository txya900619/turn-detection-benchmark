from __future__ import annotations

import json
from pathlib import Path

import pytest

from turn_eval.reports.plots import extract_plot_data, write_suite_plots


def _write_summary(root: Path) -> Path:
    eot_run = root / "eot"
    eot_run.mkdir()
    (eot_run / "metrics.json").write_text(
        json.dumps(
            {
                "pareto_frontier": [
                    {
                        "threshold": 0.5,
                        "action_delay_ms": 300,
                        "timeout_ms": 1_000,
                        "false_cutoff_rate": 0.2,
                        "mean_latency_ms": 300,
                    },
                    {
                        "threshold": 0.6,
                        "action_delay_ms": 600,
                        "timeout_ms": 1_000,
                        "false_cutoff_rate": 0.1,
                        "mean_latency_ms": 600,
                    },
                ],
                "operating_points": {
                    "false_cutoff_at_300ms": {
                        "false_cutoff_rate": 0.2,
                        "mean_latency_ms": 300,
                    },
                    "false_cutoff_at_600ms": {
                        "false_cutoff_rate": 0.1,
                        "mean_latency_ms": 600,
                    },
                },
            }
        )
    )
    summary = root / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "no_overall_score": True,
                "sections": {
                    "eot_bench": [
                        {
                            "language": "en",
                            "model": "smart-turn-v3.2-cpu",
                            "row_role": "baseline",
                            "channel_mode": "native-single-channel",
                            "signal_provenance": "native",
                            "run_path": str(eot_run),
                        }
                    ],
                    "sid_bench": [
                        {
                            "language": "en",
                            "model": "dualturn-endpointing",
                            "channel_mode": "single-channel-fallback",
                            "signal_provenance": "derived",
                            "threshold": 0.9,
                            "recall": 0.8,
                            "false_positive_rate": 0.1,
                            "latency_p50_ms": 200,
                            "latency_p90_ms": 400,
                            "average_penalty_time_s": 0.5,
                            "nonbreak_false_positive_rate": 0.12,
                            "run_path": str(root / "sid"),
                        }
                    ],
                },
            }
        )
    )
    return summary


def test_extract_plot_data_preserves_frozen_sid_point(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path)

    data, inputs = extract_plot_data(summary)

    assert data["sid_bench"][0]["threshold"] == pytest.approx(0.9)
    assert data["sid_bench"][0]["recall"] == pytest.approx(0.8)
    assert "no SID test-set threshold sweep" in data["policy_notes"]["sid_bench"]
    assert inputs == [
        (tmp_path / "eot" / "metrics.json").resolve(),
        summary.resolve(),
    ]


def test_write_suite_plots_emits_png_data_and_provenance(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path)

    output = write_suite_plots(summary, tmp_path / "plots")

    png = output / "eot_pareto_en.png"
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (output / "sid_interruption_tradeoff.png").is_file()
    assert (output / "plot_data.json").is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["sid_threshold_policy"] == "external-frozen-only; no test-set sweep"
    assert len(manifest["inputs"]) == 2


def test_plot_output_is_immutable(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path)
    output = tmp_path / "plots"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        write_suite_plots(summary, output)
