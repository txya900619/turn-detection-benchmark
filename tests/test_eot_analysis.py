from __future__ import annotations

from turn_eval.reports.eot_analysis import reconstruct_hold_outcomes


def _row(
    span: str,
    end: float,
    at: float,
    probability: float,
) -> dict[str, object]:
    return {
        "span_id": span,
        "group_id": span,
        "label": "hold",
        "silence_start_s": 0.0,
        "silence_end_s": end,
        "target_at_s": at,
        "available_at_s": at,
        "p_eot": probability,
    }


def test_reconstruct_hold_outcomes_separates_model_timeout_and_safe() -> None:
    rows = [
        _row("model", 0.4, 0.2, 0.9),
        _row("timeout", 1.2, 0.2, 0.1),
        _row("safe", 0.3, 0.2, 0.1),
    ]
    outcomes = reconstruct_hold_outcomes(
        rows,
        {"threshold": 0.5, "action_delay_ms": 200, "timeout_ms": 1_000},
        {"min_hold_span_ms": 200, "max_hold_span_ms": 5_000},
    )

    assert outcomes["model"]["cause"] == "model"
    assert outcomes["timeout"]["cause"] == "timeout"
    assert outcomes["safe"]["cause"] == "safe"
