from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np

from turn_eval.core.types import InterruptionPredictionRecord


def _quantile(values: list[float], q: float) -> float | None:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else None


def _score_rows(
    rows: tuple[InterruptionPredictionRecord, ...], tolerance_s: float
) -> dict[str, object]:
    outcomes: Counter[str] = Counter()
    penalties: list[float] = []
    correct_latencies: list[float] = []
    official_irl_delays: list[float] = []
    nonbreak_predictions = 0
    for row in rows:
        gold = row.gold_break_time_s
        predicted = row.predicted_break_time_s
        notebook_gold = row.duration_s if gold is None else gold
        if predicted is not None and abs(predicted - notebook_gold) <= tolerance_s + 1e-12:
            if gold is None:
                # The reference notebook performs its ±50 ms check before checking
                # total_nonbreak, so an end-boundary trigger receives zero penalty.
                outcome, penalty = "notebook_end_boundary_tolerance", 0.0
                nonbreak_predictions += 1
            else:
                outcome, penalty = "within_tolerance", 0.0
                correct_latencies.append(0.0)
        elif gold is None and predicted is None:
            outcome, penalty = "true_negative", 0.0
        elif gold is None:
            outcome, penalty = "false_interruption_nonbreak", row.duration_s
            nonbreak_predictions += 1
        elif predicted is None:
            outcome = "missed_interruption"
            penalty = row.duration_s - gold
        else:
            delta = predicted - gold
            if delta < 0:
                outcome, penalty = "early_interruption", row.duration_s
            else:
                outcome, penalty = "late_interruption", delta
                correct_latencies.append(delta)
                official_irl_delays.append(delta)
        outcomes[outcome] += 1
        penalties.append(penalty)

    total = len(rows)
    positives = sum(row.gold_break_time_s is not None for row in rows)
    nonbreak = total - positives
    false_interruptions = outcomes["false_interruption_nonbreak"] + outcomes["early_interruption"]
    strict_paper_false_interruptions = nonbreak_predictions + outcomes["early_interruption"]
    true_interruptions = outcomes["within_tolerance"] + outcomes["late_interruption"]
    return {
        "false_interruption_rate": false_interruptions / total,
        "official_mean_irl_s": (
            sum(official_irl_delays) / len(official_irl_delays) if official_irl_delays else None
        ),
        "average_penalty_time_s": sum(penalties) / total,
        "interruption_recall": true_interruptions / positives if positives else None,
        "response_latency_s": {
            "p50": _quantile(correct_latencies, 0.50),
            "p90": _quantile(correct_latencies, 0.90),
            "p95": _quantile(correct_latencies, 0.95),
        },
        "diagnostics": {
            "nonbreak_false_positive_rate": (nonbreak_predictions / nonbreak if nonbreak else None),
            "early_interruption_rate": (
                outcomes["early_interruption"] / positives if positives else None
            ),
            "strict_paper_fir": strict_paper_false_interruptions / total,
            "notebook_end_boundary_exemptions": outcomes["notebook_end_boundary_tolerance"],
        },
        "counts": {
            "examples": total,
            "groups": len({row.group_id for row in rows}),
            "positive_interruptions": positives,
            "nonbreak": nonbreak,
            "false_interruptions": false_interruptions,
            "strict_paper_false_interruptions": strict_paper_false_interruptions,
            "correct_interruptions": true_interruptions,
            "latency_samples": len(correct_latencies),
            "official_irl_samples": len(official_irl_delays),
            "outcomes": dict(sorted(outcomes.items())),
        },
    }


def evaluate_sid_bench(
    records: Iterable[InterruptionPredictionRecord],
    *,
    tolerance_s: float = 0.05,
) -> dict[str, object]:
    """Reproduce SID-Bench's paper/notebook metrics and add named diagnostics."""
    rows = tuple(records)
    if not rows:
        raise ValueError("cannot evaluate an empty SID-Bench prediction set")
    if tolerance_s < 0:
        raise ValueError("SID-Bench tolerance must be non-negative")
    metrics = _score_rows(rows, tolerance_s)
    categories = sorted({row.category for row in rows})
    metrics.update(
        {
            "metric_protocol": "sid-bench-paper-and-official-notebook-v1",
            "tolerance_s": tolerance_s,
            "fir_denominator": "all evaluated examples",
            "official_irl_population": "late correct detections beyond tolerance only",
            "by_category": {
                category: _score_rows(
                    tuple(row for row in rows if row.category == category), tolerance_s
                )
                for category in categories
            },
        }
    )
    return metrics
