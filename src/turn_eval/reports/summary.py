from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from turn_eval.artifacts import validate_run_artifacts
from turn_eval.core.types import DecisionLabel, PredictionRecord, StatePredictionRecord, TurnState
from turn_eval.metrics.eot import PolicyGrid, evaluate_eot_sweep
from turn_eval.metrics.state import evaluate_turn_states


def _load_policy(path: Path) -> PolicyGrid:
    data = json.loads(path.read_text())
    return PolicyGrid(
        thresholds=tuple(data["thresholds"]),
        action_delays_ms=tuple(data["action_delays_ms"]),
        timeouts_ms=tuple(data["timeouts_ms"]),
        calibration_source=data["calibration_source"],
        split_role=data["split_role"],
        min_hold_span_ms=float(data.get("min_hold_span_ms", 0.0)),
        max_hold_span_ms=(
            None if data.get("max_hold_span_ms") is None else float(data["max_hold_span_ms"])
        ),
        score_point_ms=(
            None if data.get("score_point_ms") is None else float(data["score_point_ms"])
        ),
        bootstrap_resamples=int(data.get("bootstrap_resamples", 0)),
        bootstrap_seed=int(data.get("bootstrap_seed", 0)),
    )


def _load_predictions(path: Path) -> tuple[PredictionRecord, ...]:
    records: list[PredictionRecord] = []
    for row in pq.read_table(path).to_pylist():
        records.append(
            PredictionRecord(
                example_id=row["example_id"],
                conversation_id=row["conversation_id"],
                group_id=row["group_id"],
                language=row["language"],
                decision_point_id=row["decision_point_id"],
                span_id=row["span_id"],
                label=DecisionLabel(row["label"]),
                silence_start_s=row["silence_start_s"],
                silence_end_s=row["silence_end_s"],
                target_at_s=row["target_at_s"],
                observed_until_s=row["observed_until_s"],
                available_at_s=row["available_at_s"],
                p_eot=row["p_eot"],
                inference_latency_ms=row["inference_latency_ms"],
                metadata=json.loads(row["metadata_json"]),
            )
        )
    return tuple(records)


def recompute_run_metrics(run_dir: Path) -> dict[str, object]:
    """Recompute metrics from persisted predictions without invoking a model."""

    validate_run_artifacts(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["benchmark"]["protocol"] == "clip_classification":
        stored = json.loads((run_dir / "metrics.json").read_text())
        mapping = json.loads((run_dir / "label_mapping.json").read_text())
        labels = tuple(TurnState(value) for value in mapping["output_labels"])
        records = tuple(
            StatePredictionRecord(
                example_id=row["example_id"],
                group_id=row["group_id"],
                language=row["language"],
                label=TurnState(row["label"]),
                predicted_label=(
                    None if row["predicted_label"] is None else TurnState(row["predicted_label"])
                ),
                scores=json.loads(row["scores_json"]),
                inference_latency_ms=row["inference_latency_ms"],
                metadata=json.loads(row["metadata_json"]),
                model_output=json.loads(row.get("model_output_json", "{}")),
            )
            for row in pq.read_table(run_dir / "predictions.parquet").to_pylist()
        )
        computed = evaluate_turn_states(records, labels)
        if manifest["schema_version"] == 2 and "prediction_labels" not in stored:
            computed.pop("prediction_labels")
            computed.pop("abstentions")
            computed["confusion_matrix"] = [row[:-1] for row in computed["confusion_matrix"]]
        return computed
    policy = _load_policy(run_dir / "policy.json")
    predictions = _load_predictions(run_dir / "predictions.parquet")
    computed = evaluate_eot_sweep(predictions, policy)
    stored = json.loads((run_dir / "metrics.json").read_text())
    if stored.get("schema_version") == 1:
        computed["schema_version"] = 1
        computed.pop("diagnostics", None)
    return computed


def _format_operating_point(point: object, value_key: str, scale: float = 1.0) -> str:
    if not isinstance(point, dict):
        return "n/a"
    value = float(point[value_key]) * scale
    if value_key == "false_cutoff_rate":
        return f"{value:.1f}%"
    return f"{value:.0f} ms"


def _format_result(
    metrics: dict[str, object],
    *,
    point_name: str,
    value_key: str,
    scale: float = 1.0,
) -> str:
    points = metrics["operating_points"]
    assert isinstance(points, dict)
    rendered = _format_operating_point(points[point_name], value_key, scale)
    confidence = metrics.get("confidence_intervals")
    if not isinstance(confidence, dict) or not isinstance(confidence.get("intervals"), dict):
        return rendered
    interval = confidence["intervals"].get(point_name)
    if not isinstance(interval, dict):
        return rendered
    lower = float(interval["lower"]) * scale
    upper = float(interval["upper"]) * scale
    if value_key == "false_cutoff_rate":
        return f"{rendered} (95% CI {lower:.1f}–{upper:.1f}%)"
    return f"{rendered} (95% CI {lower:.0f}–{upper:.0f} ms)"


def _render_state_markdown(
    manifest: dict[str, object], runtime: dict[str, object], metrics: dict[str, object]
) -> str:
    labels = metrics["labels"]
    per_class = metrics["per_class"]
    lines = [
        f"# Turn evaluation: {manifest['run_id']}",
        "",
        f"Model: `{manifest['model']['name']}` (`{manifest['model']['revision']}`)",
        "",
        f"Benchmark: `{manifest['benchmark']['name']}`",
        "",
        "| State | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in labels:
        row = per_class[label]
        lines.append(
            f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['support']} |"
        )
    latency = runtime["inference_latency_ms"]
    lines.extend(
        (
            "",
            f"Accuracy: {metrics['accuracy']:.4f}",
            "",
            f"Macro-F1: {metrics['macro_f1']:.4f}",
            "",
            f"Weighted-F1: {metrics['weighted_f1']:.4f}",
        )
    )
    if "abstentions" in metrics:
        lines.extend(("", f"Abstain rate: {metrics['abstentions']['rate']:.4f}"))
    lines.extend(
        (
            "",
            f"Inference p50/p90/p95: {latency['p50']:.3f} / "
            f"{latency['p90']:.3f} / {latency['p95']:.3f} ms",
            "",
        )
    )
    return "\n".join(lines)


def render_markdown(run_dir: Path, metrics: dict[str, object]) -> str:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    runtime = json.loads((run_dir / "runtime.json").read_text())
    if manifest["benchmark"]["protocol"] == "clip_classification":
        return _render_state_markdown(manifest, runtime, metrics)
    points = metrics["operating_points"]
    assert isinstance(points, dict)
    latency = runtime["inference_latency_ms"]
    return "\n".join(
        (
            f"# Turn evaluation: {manifest['run_id']}",
            "",
            f"Model: `{manifest['model']['name']}` (`{manifest['model']['revision']}`)",
            "",
            f"Benchmark: `{manifest['benchmark']['name']}`",
            "",
            "| EOT operating point | Result |",
            "|---|---:|",
            "| False cutoff @ 300 ms | "
            + _format_result(
                metrics,
                point_name="false_cutoff_at_300ms",
                value_key="false_cutoff_rate",
                scale=100,
            )
            + " |",
            "| False cutoff @ 600 ms | "
            + _format_result(
                metrics,
                point_name="false_cutoff_at_600ms",
                value_key="false_cutoff_rate",
                scale=100,
            )
            + " |",
            "| Latency @ 5% cutoff | "
            + _format_result(
                metrics,
                point_name="latency_at_5pct_cutoff",
                value_key="mean_latency_ms",
            )
            + " |",
            "| Latency @ 10% cutoff | "
            + _format_result(
                metrics,
                point_name="latency_at_10pct_cutoff",
                value_key="mean_latency_ms",
            )
            + " |",
            "",
            "| Runtime | Result |",
            "|---|---:|",
            f"| Inference p50 | {latency['p50']:.3f} ms |",
            f"| Inference p90 | {latency['p90']:.3f} ms |",
            f"| RTF | {runtime['rtf']:.6f} |",
            "",
        )
    )
