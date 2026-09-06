from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from itertools import product

import numpy as np

from turn_eval.core.types import DecisionLabel, PredictionRecord


@dataclass(frozen=True, slots=True)
class PolicyGrid:
    thresholds: tuple[float, ...]
    action_delays_ms: tuple[float, ...]
    timeouts_ms: tuple[float, ...]
    calibration_source: str
    split_role: str = "development"
    min_hold_span_ms: float = 0.0
    max_hold_span_ms: float | None = None
    score_point_ms: float | None = None
    bootstrap_resamples: int = 0
    bootstrap_seed: int = 0

    def __post_init__(self) -> None:
        if not self.thresholds or not self.action_delays_ms or not self.timeouts_ms:
            raise ValueError("policy grid dimensions must not be empty")
        if any(not 0 <= threshold <= 1 for threshold in self.thresholds):
            raise ValueError("policy thresholds must be probabilities")
        if any(delay < 0 for delay in self.action_delays_ms):
            raise ValueError("action delays cannot be negative")
        if any(timeout <= 0 for timeout in self.timeouts_ms):
            raise ValueError("timeouts must be positive")
        if self.split_role == "test":
            raise ValueError("policy fitting on a test split is forbidden")
        if not self.calibration_source:
            raise ValueError("calibration source is required")
        if self.min_hold_span_ms < 0:
            raise ValueError("minimum hold span cannot be negative")
        if self.max_hold_span_ms is not None and self.max_hold_span_ms < self.min_hold_span_ms:
            raise ValueError("maximum hold span must be at least the minimum")
        if self.score_point_ms is not None and self.score_point_ms < 0:
            raise ValueError("score point cannot be negative")
        if self.bootstrap_resamples < 0:
            raise ValueError("bootstrap resamples cannot be negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def default_policy_grid() -> PolicyGrid:
    return PolicyGrid(
        thresholds=(0.5, 0.9),
        action_delays_ms=(0.0, 100.0, 300.0, 600.0, 1_000.0),
        timeouts_ms=(300.0, 600.0, 1_000.0),
        calibration_source="synthetic-m0-development",
    )


def eot_bench_policy_grid(*, score_point_ms: float | None = None) -> PolicyGrid:
    """Return the policy grid pinned by the LiveKit eot-bench harness."""

    return PolicyGrid(
        thresholds=tuple(value / 100 for value in range(101)),
        action_delays_ms=tuple(float(value) for value in range(200, 1_001, 100)),
        timeouts_ms=tuple(float(value) for value in range(1_000, 3_501, 500)),
        calibration_source="livekit-eot-bench-prescribed-sweep",
        split_role="validation",
        min_hold_span_ms=200.0,
        max_hold_span_ms=5_000.0,
        score_point_ms=score_point_ms,
        bootstrap_resamples=1_000,
        bootstrap_seed=0,
    )


def eot_bench_vad_policy_grid() -> PolicyGrid:
    """Return the official comparison report's fine-grained VAD delay grid."""

    return PolicyGrid(
        thresholds=(0.0,),
        action_delays_ms=tuple(float(value) for value in range(200, 5_001, 100)),
        timeouts_ms=(5_000.0,),
        calibration_source="livekit-eot-bench-fine-vad-baseline",
        split_role="validation",
        min_hold_span_ms=200.0,
        max_hold_span_ms=5_000.0,
        bootstrap_resamples=1_000,
        bootstrap_seed=0,
    )


def _group_spans(
    records: Iterable[PredictionRecord],
    *,
    min_hold_span_ms: float,
    max_hold_span_ms: float | None,
) -> dict[str, list[PredictionRecord]]:
    spans: dict[str, list[PredictionRecord]] = {}
    for record in records:
        spans.setdefault(record.span_id, []).append(record)
    if not spans:
        raise ValueError("cannot evaluate an empty prediction set")
    for span_id, rows in spans.items():
        rows.sort(key=lambda row: row.available_at_s)
        first = rows[0]
        for row in rows[1:]:
            if (
                row.label is not first.label
                or row.silence_start_s != first.silence_start_s
                or row.silence_end_s != first.silence_end_s
            ):
                raise ValueError(f"inconsistent annotations for span {span_id}")
    return {
        span_id: rows
        for span_id, rows in spans.items()
        if rows[0].label is DecisionLabel.EOT
        or (
            (rows[0].silence_end_s - rows[0].silence_start_s) * 1_000 >= min_hold_span_ms - 1e-6
            and (
                max_hold_span_ms is None
                or (rows[0].silence_end_s - rows[0].silence_start_s) * 1_000
                <= max_hold_span_ms + 1e-6
            )
        )
    }


def _evaluate_policy(
    spans: Mapping[str, list[PredictionRecord]],
    *,
    threshold: float,
    action_delay_ms: float,
    timeout_ms: float,
    score_point_ms: float | None,
) -> dict[str, float | int | None]:
    hold_count = 0
    false_cutoffs = 0
    eot_latencies_ms: list[float] = []
    action_delay_s = action_delay_ms / 1_000
    timeout_s = timeout_ms / 1_000

    for rows in spans.values():
        first = rows[0]
        eligible_rows = rows
        if score_point_ms is not None:
            eligible_rows = [
                row
                for row in rows
                if abs((row.target_at_s - first.silence_start_s) * 1_000 - score_point_ms) <= 1e-3
            ]
            if not eligible_rows:
                raise ValueError(f"span {first.span_id} has no prediction at {score_point_ms}ms")
        positive_rows = [row for row in eligible_rows if row.p_eot > threshold]
        earliest_model_commit = None
        if positive_rows:
            first_fire_s = min(row.available_at_s for row in positive_rows)
            earliest_model_commit = max(first_fire_s, first.silence_start_s + action_delay_s)
        forced_commit = first.silence_start_s + timeout_s
        commit = (
            forced_commit
            if earliest_model_commit is None
            else min(earliest_model_commit, forced_commit)
        )

        if first.label is DecisionLabel.HOLD:
            hold_count += 1
            if commit < first.silence_end_s - 1e-9:
                false_cutoffs += 1
        else:
            eot_latencies_ms.append(max(0.0, (commit - first.silence_start_s) * 1_000))

    false_cutoff_rate = false_cutoffs / hold_count if hold_count else 0.0
    latency_sum_ms = sum(eot_latencies_ms)
    mean_latency_ms = latency_sum_ms / len(eot_latencies_ms) if eot_latencies_ms else None
    return {
        "threshold": threshold,
        "action_delay_ms": action_delay_ms,
        "timeout_ms": timeout_ms,
        "false_cutoff_rate": false_cutoff_rate,
        "mean_latency_ms": mean_latency_ms,
        "eot_latency_sum_ms": latency_sum_ms,
        "false_cutoffs": false_cutoffs,
        "hold_spans": hold_count,
        "eot_spans": len(eot_latencies_ms),
    }


def _bootstrap_confidence_intervals(
    spans: Mapping[str, list[PredictionRecord]],
    *,
    policy_grid: PolicyGrid,
    operating_points: Mapping[str, dict[str, float | int] | None],
) -> dict[str, object]:
    grouped: dict[str, dict[str, list[PredictionRecord]]] = {}
    for span_id, rows in spans.items():
        grouped.setdefault(rows[0].group_id, {})[span_id] = rows
    group_spans = tuple(grouped.values())
    rng = np.random.default_rng(policy_grid.bootstrap_seed)
    intervals: dict[str, object] = {}

    for name, point in operating_points.items():
        if point is None:
            intervals[name] = None
            continue
        contributions = [
            _evaluate_policy(
                item,
                threshold=float(point["threshold"]),
                action_delay_ms=float(point["action_delay_ms"]),
                timeout_ms=float(point["timeout_ms"]),
                score_point_ms=policy_grid.score_point_ms,
            )
            for item in group_spans
        ]
        is_cutoff = name.startswith("false_cutoff")
        samples: list[float] = []
        for _ in range(policy_grid.bootstrap_resamples):
            selected = rng.integers(0, len(contributions), size=len(contributions))
            if is_cutoff:
                denominator = sum(int(contributions[index]["hold_spans"]) for index in selected)
                if denominator:
                    numerator = sum(
                        int(contributions[index]["false_cutoffs"]) for index in selected
                    )
                    samples.append(numerator / denominator)
            else:
                denominator = sum(int(contributions[index]["eot_spans"]) for index in selected)
                if denominator:
                    numerator = sum(
                        float(contributions[index]["eot_latency_sum_ms"]) for index in selected
                    )
                    samples.append(numerator / denominator)
        if not samples:
            intervals[name] = None
            continue
        values = np.asarray(samples, dtype=np.float64)
        intervals[name] = {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "confidence": 0.95,
            "effective_resamples": len(samples),
        }

    return {
        "method": "fixed-operating-policy cluster bootstrap",
        "group_field": "group_id",
        "groups": len(group_spans),
        "resamples": policy_grid.bootstrap_resamples,
        "seed": policy_grid.bootstrap_seed,
        "intervals": intervals,
    }


def _pareto_frontier(points: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    ordered = sorted(
        points,
        key=lambda point: (
            float(point["mean_latency_ms"]),
            float(point["false_cutoff_rate"]),
        ),
    )
    frontier: list[dict[str, float | int]] = []
    best_cutoff = float("inf")
    for point in ordered:
        cutoff = float(point["false_cutoff_rate"])
        if cutoff < best_cutoff:
            frontier.append(point)
            best_cutoff = cutoff
    return frontier


def _diagnostic_classification(
    spans: Mapping[str, list[PredictionRecord]], *, score_point_ms: float | None
) -> dict[str, float | int | None]:
    labels: list[int] = []
    scores: list[float] = []
    for rows in spans.values():
        first = rows[0]
        eligible_rows = rows
        if score_point_ms is not None:
            eligible_rows = [
                row
                for row in rows
                if abs((row.target_at_s - first.silence_start_s) * 1_000 - score_point_ms) <= 1e-3
            ]
        if not eligible_rows:
            raise ValueError(f"span {first.span_id} has no classification score")
        labels.append(int(first.label is DecisionLabel.EOT))
        scores.append(max(row.p_eot for row in eligible_rows))

    y_true = np.asarray(labels, dtype=np.int8)
    y_score = np.asarray(scores, dtype=np.float64)
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if not positives or not negatives:
        return {
            "auc": None,
            "average_precision": None,
            "scored_spans": len(labels),
            "positive_spans": positives,
            "negative_spans": negatives,
        }

    order = np.argsort(y_score, kind="stable")
    rank_sum = 0.0
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and y_score[order[end]] == y_score[order[index]]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        rank_sum += average_rank * int(y_true[order[index:end]].sum())
        index = end
    auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

    descending = np.argsort(-y_score, kind="stable")
    true_positives = 0
    predicted_positives = 0
    average_precision = 0.0
    index = 0
    while index < len(descending):
        end = index + 1
        while end < len(descending) and y_score[descending[end]] == y_score[descending[index]]:
            end += 1
        group_positives = int(y_true[descending[index:end]].sum())
        true_positives += group_positives
        predicted_positives += end - index
        average_precision += (group_positives / positives) * (true_positives / predicted_positives)
        index = end

    return {
        "auc": float(auc),
        "average_precision": float(average_precision),
        "scored_spans": len(labels),
        "positive_spans": positives,
        "negative_spans": negatives,
    }


def _best_at_latency(
    points: list[dict[str, float | int]], budget_ms: float
) -> dict[str, float | int] | None:
    eligible = [point for point in points if float(point["mean_latency_ms"]) <= budget_ms + 1e-6]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda point: (
            float(point["false_cutoff_rate"]),
            float(point["mean_latency_ms"]),
        ),
    )


def _best_at_cutoff(
    points: list[dict[str, float | int]], budget: float
) -> dict[str, float | int] | None:
    eligible = [point for point in points if float(point["false_cutoff_rate"]) <= budget + 1e-12]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda point: (
            float(point["mean_latency_ms"]),
            float(point["false_cutoff_rate"]),
        ),
    )


def evaluate_eot_sweep(
    records: Iterable[PredictionRecord], policy_grid: PolicyGrid
) -> dict[str, object]:
    spans = _group_spans(
        records,
        min_hold_span_ms=policy_grid.min_hold_span_ms,
        max_hold_span_ms=policy_grid.max_hold_span_ms,
    )
    if not spans:
        raise ValueError("no spans remain after EOT benchmark filtering")
    points = [
        _evaluate_policy(
            spans,
            threshold=threshold,
            action_delay_ms=action_delay_ms,
            timeout_ms=timeout_ms,
            score_point_ms=policy_grid.score_point_ms,
        )
        for threshold, action_delay_ms, timeout_ms in product(
            policy_grid.thresholds,
            policy_grid.action_delays_ms,
            policy_grid.timeouts_ms,
        )
        if timeout_ms + 1e-9 >= action_delay_ms
    ]
    operating_points = {
        "false_cutoff_at_300ms": _best_at_latency(points, 300.0),
        "false_cutoff_at_600ms": _best_at_latency(points, 600.0),
        "latency_at_5pct_cutoff": _best_at_cutoff(points, 0.05),
        "latency_at_10pct_cutoff": _best_at_cutoff(points, 0.10),
    }
    result = {
        "schema_version": 2,
        "protocol": "decision_point",
        "task": "eot_hold",
        "counts": {
            "prediction_rows": sum(len(rows) for rows in spans.values()),
            "spans": len(spans),
            "hold_spans": sum(rows[0].label is DecisionLabel.HOLD for rows in spans.values()),
            "eot_spans": sum(rows[0].label is DecisionLabel.EOT for rows in spans.values()),
        },
        "operating_points": operating_points,
        "diagnostics": _diagnostic_classification(spans, score_point_ms=policy_grid.score_point_ms),
        "pareto_frontier": _pareto_frontier(points),
        "policy_sweep": points,
    }
    if policy_grid.bootstrap_resamples:
        result["confidence_intervals"] = _bootstrap_confidence_intervals(
            spans,
            policy_grid=policy_grid,
            operating_points=operating_points,
        )
    for point in points:
        point.pop("eot_latency_sum_ms", None)
    return result
