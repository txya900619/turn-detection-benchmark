from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

DURATION_BINS_MS = (
    (200, 300, "200--<300 ms"),
    (300, 500, "300--<500 ms"),
    (500, 1_000, "500--<1000 ms"),
    (1_000, 2_000, "1--<2 s"),
    (2_000, 5_001, "2--5 s"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reconstruct_hold_outcomes(
    rows: list[dict[str, Any]],
    operating_point: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    threshold = float(operating_point["threshold"])
    action_delay_s = float(operating_point["action_delay_ms"]) / 1_000
    timeout_s = float(operating_point["timeout_ms"]) / 1_000
    minimum_ms = float(policy.get("min_hold_span_ms", 0.0))
    maximum = policy.get("max_hold_span_ms")
    maximum_ms = None if maximum is None else float(maximum)
    score_point = policy.get("score_point_ms")
    score_point_ms = None if score_point is None else float(score_point)
    spans: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["label"] != "hold":
            continue
        duration_ms = round((float(row["silence_end_s"]) - float(row["silence_start_s"])) * 1_000)
        if duration_ms < minimum_ms - 1e-6:
            continue
        if maximum_ms is not None and duration_ms > maximum_ms + 1e-6:
            continue
        spans[str(row["span_id"])].append(row)
    if not spans:
        raise ValueError("EOT analysis found no eligible HOLD spans")

    outcomes: dict[str, dict[str, Any]] = {}
    for span_id, span_rows in spans.items():
        span_rows.sort(key=lambda row: float(row["available_at_s"]))
        first = span_rows[0]
        eligible = span_rows
        if score_point_ms is not None:
            eligible = [
                row
                for row in span_rows
                if abs(
                    (float(row["target_at_s"]) - float(first["silence_start_s"])) * 1_000
                    - score_point_ms
                )
                <= 1e-3
            ]
            if not eligible:
                raise ValueError(f"HOLD span {span_id} has no score at {score_point_ms} ms")
        fires = [
            float(row["available_at_s"]) for row in eligible if float(row["p_eot"]) > threshold
        ]
        silence_start = float(first["silence_start_s"])
        silence_end = float(first["silence_end_s"])
        model_commit = max(min(fires), silence_start + action_delay_s) if fires else None
        timeout_commit = silence_start + timeout_s
        commit = timeout_commit if model_commit is None else min(model_commit, timeout_commit)
        false_cutoff = commit < silence_end - 1e-9
        cause = "safe"
        if false_cutoff:
            cause = (
                "model"
                if model_commit is not None and model_commit <= timeout_commit
                else "timeout"
            )
        outcomes[span_id] = {
            "span_id": span_id,
            "group_id": str(first["group_id"]),
            "duration_ms": round((silence_end - silence_start) * 1_000),
            "false_cutoff": false_cutoff,
            "cause": cause,
            "commit_latency_ms": (commit - silence_start) * 1_000,
        }
    return outcomes


def _cluster_paired_ci(
    candidate: dict[str, dict[str, Any]],
    reference: dict[str, dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    if set(candidate) != set(reference):
        raise ValueError("paired EOT analysis requires identical HOLD span coverage")
    groups: dict[str, list[str]] = defaultdict(list)
    for span_id, row in candidate.items():
        if row["group_id"] != reference[span_id]["group_id"]:
            raise ValueError(f"paired EOT group mismatch for {span_id}")
        groups[row["group_id"]].append(span_id)
    contributions = [
        (
            sum(int(candidate[item]["false_cutoff"]) for item in span_ids),
            sum(int(reference[item]["false_cutoff"]) for item in span_ids),
            len(span_ids),
        )
        for span_ids in groups.values()
    ]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(resamples):
        selected = rng.integers(0, len(contributions), size=len(contributions))
        candidate_false = sum(contributions[index][0] for index in selected)
        reference_false = sum(contributions[index][1] for index in selected)
        denominator = sum(contributions[index][2] for index in selected)
        samples.append((candidate_false - reference_false) / denominator)
    values = np.asarray(samples, dtype=np.float64)
    denominator = len(candidate)
    estimate = (
        sum(int(row["false_cutoff"]) for row in candidate.values())
        - sum(int(row["false_cutoff"]) for row in reference.values())
    ) / denominator
    return {
        "estimate": estimate,
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
        "groups": len(groups),
        "resamples": resamples,
    }


def _summarize_outcomes(outcomes: dict[str, dict[str, Any]]) -> dict[str, object]:
    rows = list(outcomes.values())
    false_cutoffs = sum(int(row["false_cutoff"]) for row in rows)
    duration_slices = []
    for lower, upper, label in DURATION_BINS_MS:
        selected = [row for row in rows if lower <= row["duration_ms"] < upper]
        if not selected:
            continue
        errors = sum(int(row["false_cutoff"]) for row in selected)
        duration_slices.append(
            {
                "label": label,
                "lower_ms": lower,
                "upper_exclusive_ms": upper,
                "spans": len(selected),
                "false_cutoffs": errors,
                "false_cutoff_rate": errors / len(selected),
                "model_triggered": sum(row["cause"] == "model" for row in selected),
                "timeout_triggered": sum(row["cause"] == "timeout" for row in selected),
            }
        )
    if sum(int(row["spans"]) for row in duration_slices) != len(rows):
        raise ValueError("EOT duration bins do not cover every HOLD span")
    return {
        "hold_spans": len(rows),
        "false_cutoffs": false_cutoffs,
        "false_cutoff_rate": false_cutoffs / len(rows),
        "model_triggered": sum(row["cause"] == "model" for row in rows),
        "timeout_triggered": sum(row["cause"] == "timeout" for row in rows),
        "safe": sum(row["cause"] == "safe" for row in rows),
        "duration_slices": duration_slices,
    }


def _render_report(analyses: list[dict[str, Any]], operating_point_name: str) -> str:
    lines = [
        "# EOT-Bench HOLD Error Analysis",
        "",
        f"Operating point: `{operating_point_name}`. Every threshold, action delay, and",
        "timeout is replayed from its immutable input metrics. No threshold or policy is",
        "selected by this analysis. Paired intervals use conversation-cluster bootstrap.",
    ]
    for language_analysis in analyses:
        lines.extend(
            [
                "",
                f"## {language_analysis['language'].upper()}",
                "",
                "| Model | Threshold | Delay | Timeout | False cutoffs | Model | Timeout | Rate |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for model in language_analysis["models"]:
            point, summary = model["operating_point"], model["summary"]
            lines.append(
                f"| {model['model']} | {point['threshold']:.6g} | "
                f"{point['action_delay_ms']:.0f} ms | {point['timeout_ms']:.0f} ms | "
                f"{summary['false_cutoffs']} / {summary['hold_spans']} | "
                f"{summary['model_triggered']} | {summary['timeout_triggered']} | "
                f"{100 * summary['false_cutoff_rate']:.2f}% |"
            )
        lines.extend(
            [
                "",
                f"Candidate: `{language_analysis['candidate_model']}`.",
                "",
                "| Reference | Candidate-only | Reference-only | Both false | "
                "Both safe | Delta [95% CI] |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for pair in language_analysis["paired_comparisons"]:
            ci = pair["false_cutoff_rate_delta"]
            counts = pair["paired_counts"]
            lines.append(
                f"| {pair['reference_model']} | {counts['candidate_only']} | "
                f"{counts['reference_only']} | {counts['both_false']} | "
                f"{counts['both_safe']} | {100 * ci['estimate']:+.2f} pp "
                f"[{100 * ci['lower']:+.2f}, {100 * ci['upper']:+.2f}] |"
            )
        labels = [
            row["label"] for row in language_analysis["models"][0]["summary"]["duration_slices"]
        ]
        lines.extend(
            [
                "",
                "| HOLD duration | N | "
                + " | ".join(model["model"] for model in language_analysis["models"])
                + " |",
                "|---|---:|" + "---:|" * len(language_analysis["models"]),
            ]
        )
        for index, label in enumerate(labels):
            slices = [
                model["summary"]["duration_slices"][index] for model in language_analysis["models"]
            ]
            if any(
                item["label"] != label or item["spans"] != slices[0]["spans"] for item in slices
            ):
                raise ValueError("EOT duration slice alignment mismatch")
            lines.append(
                f"| {label} | {slices[0]['spans']} | "
                + " | ".join(f"{100 * item['false_cutoff_rate']:.1f}%" for item in slices)
                + " |"
            )
    return "\n".join(lines) + "\n"


def write_eot_hold_analysis(
    run_dirs: list[Path],
    output_dir: Path,
    *,
    candidate_model: str,
    operating_point_name: str = "false_cutoff_at_300ms",
    bootstrap_resamples: int = 10_000,
    seed: int = 0,
) -> Path:
    if not run_dirs:
        raise ValueError("EOT analysis requires at least one run")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    if output_dir.exists():
        raise FileExistsError(f"EOT analysis output already exists: {output_dir}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    inputs: list[dict[str, str]] = []
    for run_dir in run_dirs:
        required = [
            run_dir / "manifest.json",
            run_dir / "metrics.json",
            run_dir / "policy.json",
            run_dir / "predictions.parquet",
            run_dir / "SUCCESS",
        ]
        if any(not path.is_file() for path in required):
            raise ValueError(f"EOT analysis requires a completed run: {run_dir}")
        manifest = json.loads(required[0].read_text())
        metrics = json.loads(required[1].read_text())
        policy = json.loads(required[2].read_text())
        if manifest["benchmark"]["name"] != "livekit-eot-bench":
            raise ValueError("EOT analysis accepts only LiveKit eot-bench runs")
        if manifest["benchmark"].get("partial_run") is not False:
            raise ValueError("EOT analysis excludes partial runs")
        point = metrics.get("operating_points", {}).get(operating_point_name)
        if not isinstance(point, dict):
            raise ValueError(f"run has no operating point {operating_point_name}: {run_dir}")
        rows = pq.read_table(required[3]).to_pylist()
        outcomes = reconstruct_hold_outcomes(rows, point, policy)
        summary = _summarize_outcomes(outcomes)
        if (
            summary["hold_spans"] != point["hold_spans"]
            or summary["false_cutoffs"] != point["false_cutoffs"]
        ):
            raise ValueError(f"replayed EOT counts do not match metrics: {run_dir}")
        language = str(manifest["benchmark"]["language"])
        grouped[language].append(
            {
                "model": manifest["model"]["name"],
                "run": str(run_dir.resolve()),
                "dataset_revision": manifest["dataset"]["revision"],
                "operating_point": point,
                "summary": summary,
                "outcomes": outcomes,
            }
        )
        for path in required[:-1]:
            inputs.append({"path": str(path.resolve()), "sha256": _sha256(path)})

    analyses: list[dict[str, Any]] = []
    for language, models in sorted(grouped.items()):
        if len({model["model"] for model in models}) != len(models):
            raise ValueError(f"EOT analysis has duplicate model identities for {language}")
        revisions = {model["dataset_revision"] for model in models}
        if len(revisions) != 1:
            raise ValueError(f"EOT analysis dataset revisions differ for {language}")
        candidates = [model for model in models if model["model"] == candidate_model]
        if len(candidates) != 1:
            raise ValueError(
                f"candidate model {candidate_model} is missing or duplicated for {language}"
            )
        candidate = candidates[0]
        references = sorted(
            (model for model in models if model is not candidate),
            key=lambda row: row["model"],
        )
        ordered = [candidate, *references]
        comparisons = []
        for reference in ordered[1:]:
            if set(candidate["outcomes"]) != set(reference["outcomes"]):
                raise ValueError(f"EOT HOLD coverage differs for {language}")
            pairs = [
                (
                    bool(candidate["outcomes"][span_id]["false_cutoff"]),
                    bool(reference["outcomes"][span_id]["false_cutoff"]),
                )
                for span_id in candidate["outcomes"]
            ]
            comparisons.append(
                {
                    "reference_model": reference["model"],
                    "paired_counts": {
                        "both_false": sum(left and right for left, right in pairs),
                        "candidate_only": sum(left and not right for left, right in pairs),
                        "reference_only": sum(not left and right for left, right in pairs),
                        "both_safe": sum(not left and not right for left, right in pairs),
                    },
                    "false_cutoff_rate_delta": _cluster_paired_ci(
                        candidate["outcomes"],
                        reference["outcomes"],
                        resamples=bootstrap_resamples,
                        seed=seed,
                    ),
                }
            )
        for model in ordered:
            del model["outcomes"]
        analyses.append(
            {
                "language": language,
                "candidate_model": candidate_model,
                "models": ordered,
                "paired_comparisons": comparisons,
            }
        )

    created_at = datetime.now(UTC).isoformat()
    result = {
        "schema_version": 1,
        "created_at": created_at,
        "operating_point": operating_point_name,
        "calibration": "none; stored operating points replayed without modification",
        "bootstrap": {
            "method": "paired conversation-cluster bootstrap",
            "resamples": bootstrap_resamples,
            "seed": seed,
            "confidence": 0.95,
        },
        "analyses": analyses,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    (output_dir / "report.md").write_text(_render_report(analyses, operating_point_name))
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": created_at,
                "candidate_model": candidate_model,
                "operating_point": operating_point_name,
                "bootstrap_resamples": bootstrap_resamples,
                "seed": seed,
                "inputs": inputs,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    (output_dir / "SUCCESS").write_text("ok\n")
    return output_dir / "report.md"
