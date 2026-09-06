from __future__ import annotations

from time import monotonic

import pytest

from turn_eval.runtime_profile import (
    ProcessResourceSampler,
    process_age_seconds,
    summarize_latencies_seconds,
)


def test_latency_summary_uses_milliseconds_and_named_percentiles() -> None:
    summary = summarize_latencies_seconds([0.001, 0.002, 0.003, 0.004])

    assert summary["count"] == 4
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["p50"] == pytest.approx(2.5)
    assert summary["p95"] == pytest.approx(3.85)
    assert summary["minimum"] == pytest.approx(1.0)
    assert summary["maximum"] == pytest.approx(4.0)


def test_empty_latency_summary_is_explicit() -> None:
    summary = summarize_latencies_seconds([])

    assert summary["count"] == 0
    assert summary["mean"] is None
    assert summary["p50"] is None


def test_process_sampler_reports_process_scoped_cpu_and_ram() -> None:
    sampler = ProcessResourceSampler(interval_s=0.005).start()
    deadline = monotonic() + 0.03
    while monotonic() < deadline:
        sum(index * index for index in range(100))
    result = sampler.stop()

    assert result["sample_count"] >= 2
    assert str(result["process_scope"]).startswith("pid:")
    assert result["wall_s"] > 0
    assert result["cpu_time_s"] >= 0
    assert result["ram"]["baseline_rss_bytes"] > 0
    assert result["ram"]["peak_rss_bytes"] >= result["ram"]["baseline_rss_bytes"]
    assert result["gpu_vram"]["peak_bytes"] is None


def test_process_age_is_available_on_linux() -> None:
    age = process_age_seconds()

    assert age is not None
    assert age >= 0
