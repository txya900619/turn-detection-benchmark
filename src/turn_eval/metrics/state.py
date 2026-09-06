from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from turn_eval.core.types import StatePredictionRecord, TurnState


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_turn_states(
    records: Iterable[StatePredictionRecord],
    labels: Sequence[TurnState],
) -> dict[str, object]:
    """Compute deterministic row-gold/column-predicted turn-state metrics."""
    rows = tuple(records)
    ordered_labels = tuple(labels)
    if not rows:
        raise ValueError("cannot evaluate an empty state prediction set")
    if not ordered_labels or len(ordered_labels) != len(set(ordered_labels)):
        raise ValueError("state labels must be a non-empty unique sequence")

    indices = {label: index for index, label in enumerate(ordered_labels)}
    confusion = np.zeros((len(ordered_labels), len(ordered_labels) + 1), dtype=np.int64)
    abstain_index = len(ordered_labels)
    for row in rows:
        if row.label not in indices:
            raise ValueError(
                f"prediction {row.example_id} uses a gold label outside the vocabulary"
            )
        if row.predicted_label is not None and row.predicted_label not in indices:
            raise ValueError(f"prediction {row.example_id} uses a label outside the vocabulary")
        predicted_index = (
            abstain_index if row.predicted_label is None else indices[row.predicted_label]
        )
        confusion[indices[row.label], predicted_index] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    weighted_f1_sum = 0.0
    for label, index in indices.items():
        true_positive = int(confusion[index, index])
        support = int(confusion[index, :].sum())
        predicted = int(confusion[:, index].sum())
        precision = _safe_ratio(true_positive, predicted)
        recall = _safe_ratio(true_positive, support)
        f1 = _safe_ratio(2 * true_positive, support + predicted)
        false_positive = predicted - true_positive
        false_negative = support - true_positive
        true_negative = len(rows) - true_positive - false_positive - false_negative
        per_class[label.value] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": predicted,
            "one_vs_rest_accuracy": _safe_ratio(true_positive + true_negative, len(rows)),
            "miss_rate": _safe_ratio(false_negative, support),
            "false_alarm_rate": _safe_ratio(false_positive, len(rows) - support),
        }
        weighted_f1_sum += f1 * support

    total = len(rows)
    inference_ms = np.asarray([row.inference_latency_ms for row in rows], dtype=np.float64)
    critical_errors: dict[str, dict[str, float | int]] = {}
    for source, target, name in (
        (TurnState.INCOMPLETE, TurnState.COMPLETE, "incomplete_to_complete"),
        (TurnState.BACKCHANNEL, TurnState.COMPLETE, "backchannel_to_complete"),
    ):
        if source in indices and target in indices:
            count = int(confusion[indices[source], indices[target]])
            denominator = int(confusion[indices[source], :].sum())
            critical_errors[name] = {
                "count": count,
                "denominator": denominator,
                "rate": _safe_ratio(count, denominator),
            }

    return {
        "labels": [label.value for label in ordered_labels],
        "prediction_labels": [*[label.value for label in ordered_labels], "abstain"],
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_orientation": "rows=gold, columns=predicted",
        "per_class": per_class,
        "accuracy": float(sum(confusion[index, index] for index in range(len(indices))) / total),
        "macro_f1": sum(float(item["f1"]) for item in per_class.values()) / len(ordered_labels),
        "balanced_accuracy": sum(float(item["recall"]) for item in per_class.values())
        / len(ordered_labels),
        "weighted_f1": weighted_f1_sum / total,
        "critical_errors": critical_errors,
        "abstentions": {
            "count": int(confusion[:, abstain_index].sum()),
            "rate": float(confusion[:, abstain_index].sum() / total),
        },
        "counts": {"examples": total, "groups": len({row.group_id for row in rows})},
        "inference_latency_ms": {
            "p50": float(np.quantile(inference_ms, 0.50)),
            "p90": float(np.quantile(inference_ms, 0.90)),
            "p95": float(np.quantile(inference_ms, 0.95)),
        },
    }


def eot_subset(
    records: Iterable[StatePredictionRecord],
) -> tuple[StatePredictionRecord, ...]:
    """Select native Complete/Incomplete examples without remapping other states."""
    allowed = {TurnState.COMPLETE, TurnState.INCOMPLETE}
    rows = tuple(row for row in records if row.label in allowed)
    if any(row.predicted_label is not None and row.predicted_label not in allowed for row in rows):
        raise ValueError("two-state predictions must emit only complete or incomplete")
    return rows
