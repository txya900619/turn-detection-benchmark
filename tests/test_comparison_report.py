from __future__ import annotations

from pathlib import Path

from turn_eval.reports.comparison import (
    _empty_row,
    _model_license,
    _render_markdown,
    _select_latest,
)


def test_model_license_renders_missing_and_null_as_na() -> None:
    assert _model_license({}) == "N/A"
    assert _model_license({"model_license": None}) == "N/A"
    assert _model_license({"license": "Apache-2.0"}) == "Apache-2.0"


def test_latest_selection_uses_canonical_bonus_for_timestamp_ties(tmp_path: Path) -> None:
    merged = tmp_path / "merged"
    final = tmp_path / "final"
    older = tmp_path / "older"
    candidates = [
        (merged, {"identity": "model", "created_at": "2026-01-02T00:00:00Z"}),
        (final, {"identity": "model", "created_at": "2026-01-02T00:00:00Z"}),
        (older, {"identity": "model", "created_at": "2026-01-01T00:00:00Z"}),
    ]

    selected, rejected = _select_latest(
        candidates,
        identity=lambda value: value["identity"],
        canonical_bonus=lambda path: 2 if path.name == "final" else 1,
    )

    assert selected == [(final, candidates[1][1])]
    assert {Path(row["path"]).name for row in rejected} == {"merged", "older"}
    assert all(row["reason"] == "superseded canonical run" for row in rejected)


def test_markdown_renders_fastturn_rows_without_shadowing_coverage() -> None:
    eot_row = _empty_row(
        benchmark="fastturn-eot",
        language="zh",
        task="turn_state_2state",
        model="tiny-model",
        row_role="baseline",
        channel_mode="native_single",
        execution_mode="offline",
        signal_provenance="native",
        model_license="Apache-2.0",
        threshold=0.5,
        complete_precision=0.9,
        complete_recall=0.8,
        complete_f1=0.85,
        incomplete_precision=0.7,
        incomplete_recall=0.6,
        incomplete_f1=0.65,
        macro_f1=0.75,
        incomplete_to_complete_error_rate=0.2,
        abstention_rate=0.1,
        latency_p50_ms=40,
        rtf=0.02,
    )
    full_row = _empty_row(
        benchmark="fastturn-full",
        language="zh",
        task="turn_state_4state",
        model="four-state-model",
        row_role="baseline",
        channel_mode="native_single",
        execution_mode="offline",
        signal_provenance="native",
        model_license="Apache-2.0",
        complete_f1=0.85,
        incomplete_f1=0.65,
        backchannel_f1=0.8,
        wait_f1=0.9,
        macro_f1=0.8,
        incomplete_to_complete_error_rate=0.2,
        backchannel_to_complete_error_rate=0.03,
        abstention_rate=0.01,
        latency_p50_ms=400,
        rtf=0.2,
    )

    rendered = _render_markdown(
        [eot_row, full_row],
        {
            "status": "binary EOT and native four-state coverage complete in separate tables",
            "excluded_runs": [],
        },
        {"zh"},
    )

    assert "FastTurn ZH — binary EOT subset" in rendered
    assert "| tiny-model | native_single / offline / native | 0.5 |" in rendered
    assert "FastTurn ZH — native four-state" in rendered
    assert "not comparable to the binary EOT subset" in rendered
    assert "binary EOT and native four-state coverage complete" in rendered
