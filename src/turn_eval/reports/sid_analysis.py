from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int,
) -> dict[str, float]:
    if not len(values):
        raise ValueError("cannot bootstrap an empty metric")
    estimates: list[np.ndarray] = []
    completed = 0
    while completed < resamples:
        count = min(250, resamples - completed)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates.append(values[indices].mean(axis=1))
        completed += count
    bootstrap = np.concatenate(estimates)
    return {
        "estimate": float(values.mean()),
        "lower": float(np.quantile(bootstrap, 0.025)),
        "upper": float(np.quantile(bootstrap, 0.975)),
    }


def _row_values(row: dict[str, Any], tolerance_s: float) -> dict[str, float | bool]:
    gold = row["gold_break_time_s"]
    predicted = row["predicted_break_time_s"]
    duration = float(row["annotated_duration_s"])
    positive = gold is not None
    trigger = predicted is not None
    if positive:
        gold = float(gold)
        correct = trigger and float(predicted) >= gold - tolerance_s - 1e-12
        early = trigger and float(predicted) < gold - tolerance_s - 1e-12
        if not trigger:
            penalty = duration - gold
        elif abs(float(predicted) - gold) <= tolerance_s + 1e-12:
            penalty = 0.0
        elif float(predicted) < gold:
            penalty = duration
        else:
            penalty = float(predicted) - gold
    else:
        correct = False
        early = False
        penalty = (
            0.0
            if not trigger or abs(float(predicted) - duration) <= tolerance_s + 1e-12
            else duration
        )
    return {
        "positive": positive,
        "correct": bool(correct),
        "early": bool(early),
        "trigger": trigger,
        "penalty": penalty,
    }


def analyze_sid_pair(
    candidate_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    *,
    tolerance_s: float = 0.05,
    bootstrap_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, object]:
    if tolerance_s < 0:
        raise ValueError("tolerance must be non-negative")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    candidate = {str(row["example_id"]): row for row in candidate_rows}
    reference = {str(row["example_id"]): row for row in reference_rows}
    if len(candidate) != len(candidate_rows) or len(reference) != len(reference_rows):
        raise ValueError("SID analysis inputs contain duplicate example IDs")
    if set(candidate) != set(reference) or not candidate:
        raise ValueError("SID paired analysis requires identical non-empty example coverage")
    ids = sorted(candidate)
    for example_id in ids:
        left, right = candidate[example_id], reference[example_id]
        for field in ("category", "annotated_duration_s", "gold_break_time_s"):
            if left[field] != right[field]:
                raise ValueError(f"SID paired input mismatch for {example_id}: {field}")

    rng = np.random.default_rng(seed)
    values = {
        example_id: (
            _row_values(candidate[example_id], tolerance_s),
            _row_values(reference[example_id], tolerance_s),
        )
        for example_id in ids
    }

    def comparison(selected: list[str], field: str) -> dict[str, object]:
        left = np.asarray([float(values[item][0][field]) for item in selected])
        right = np.asarray([float(values[item][1][field]) for item in selected])
        paired = _mean_ci(left - right, rng=rng, resamples=bootstrap_resamples)
        return {
            "candidate": float(left.mean()),
            "reference": float(right.mean()),
            "delta": paired,
        }

    positive_ids = [item for item in ids if values[item][0]["positive"]]
    strict_fir_candidate = np.asarray(
        [
            float(pair[0]["early"] if pair[0]["positive"] else pair[0]["trigger"])
            for pair in values.values()
        ]
    )
    strict_fir_reference = np.asarray(
        [
            float(pair[1]["early"] if pair[1]["positive"] else pair[1]["trigger"])
            for pair in values.values()
        ]
    )
    strict_delta = _mean_ci(
        strict_fir_candidate - strict_fir_reference,
        rng=rng,
        resamples=bootstrap_resamples,
    )
    overall: dict[str, object] = {
        "strict_fir": {
            "candidate": float(strict_fir_candidate.mean()),
            "reference": float(strict_fir_reference.mean()),
            "delta": strict_delta,
        },
        "average_penalty_time_s": comparison(ids, "penalty"),
    }
    if positive_ids:
        overall["interruption_recall"] = comparison(positive_ids, "correct")

    categories: dict[str, object] = {}
    for category in sorted({str(candidate[item]["category"]) for item in ids}):
        selected = [item for item in ids if candidate[item]["category"] == category]
        positive = bool(values[selected[0]][0]["positive"])
        primary_field = "correct" if positive else "trigger"
        left = np.asarray([bool(values[item][0][primary_field]) for item in selected])
        right = np.asarray([bool(values[item][1][primary_field]) for item in selected])
        category_result: dict[str, object] = {
            "examples": len(selected),
            "primary_metric": "interruption_recall" if positive else "strict_trigger_rate",
            "primary": comparison(selected, primary_field),
            "average_penalty_time_s": comparison(selected, "penalty"),
            "paired_counts": {
                "both_positive": int(np.sum(left & right)),
                "candidate_only": int(np.sum(left & ~right)),
                "reference_only": int(np.sum(~left & right)),
                "neither": int(np.sum(~left & ~right)),
            },
        }
        if positive:
            category_result["early_interruption_rate"] = comparison(selected, "early")
        categories[category] = category_result

    return {
        "schema_version": 1,
        "method": "fixed-policy paired clip bootstrap",
        "confidence": 0.95,
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
        "tolerance_s": tolerance_s,
        "examples": len(ids),
        "overall": overall,
        "by_category": categories,
    }


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _delta(metric: dict[str, Any], *, percentage: bool) -> str:
    result = metric["delta"]
    scale = 100 if percentage else 1
    suffix = " pp" if percentage else " s"
    return (
        f"{scale * result['estimate']:+.2f}{suffix} "
        f"[{scale * result['lower']:+.2f}, {scale * result['upper']:+.2f}]"
    )


def _render_report(pairs: list[dict[str, Any]]) -> str:
    lines = [
        "# SID-Bench Paired Fixed-Policy Analysis",
        "",
        "Thresholds remain frozen at the values recorded in each input run. The 95%",
        "intervals use paired clip bootstrap resampling; no SID-Bench label is used",
        "for calibration. Negative strict-FIR and APT deltas favor the candidate.",
    ]
    for pair in pairs:
        analysis = pair["analysis"]
        overall = analysis["overall"]
        lines.extend(
            [
                "",
                f"## {pair['split'].upper()}",
                "",
                f"Candidate: `{pair['candidate_model']}`; reference: `{pair['reference_model']}`.",
                "",
                "| Metric | Candidate | Reference | Paired delta [95% CI] |",
                "|---|---:|---:|---:|",
            ]
        )
        if "interruption_recall" in overall:
            metric = overall["interruption_recall"]
            lines.append(
                f"| Recall | {_pct(metric['candidate'])} | {_pct(metric['reference'])} | "
                f"{_delta(metric, percentage=True)} |"
            )
        metric = overall["strict_fir"]
        lines.append(
            f"| Strict FIR | {_pct(metric['candidate'])} | {_pct(metric['reference'])} | "
            f"{_delta(metric, percentage=True)} |"
        )
        metric = overall["average_penalty_time_s"]
        lines.append(
            f"| APT | {metric['candidate']:.3f} s | {metric['reference']:.3f} s | "
            f"{_delta(metric, percentage=False)} |"
        )
        lines.extend(
            [
                "",
                "| Category | Metric | Candidate | Reference | Paired delta [95% CI] |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for category, result in analysis["by_category"].items():
            metric = result["primary"]
            lines.append(
                f"| {category} | {result['primary_metric']} | {_pct(metric['candidate'])} | "
                f"{_pct(metric['reference'])} | {_delta(metric, percentage=True)} |"
            )
            if "early_interruption_rate" in result:
                metric = result["early_interruption_rate"]
                lines.append(
                    f"| {category} | early interruption | {_pct(metric['candidate'])} | "
                    f"{_pct(metric['reference'])} | {_delta(metric, percentage=True)} |"
                )
    return "\n".join(lines) + "\n"


def write_sid_paired_analysis(
    candidate_runs: list[Path],
    reference_runs: list[Path],
    output_dir: Path,
    *,
    bootstrap_resamples: int = 10_000,
    seed: int = 0,
) -> Path:
    if len(candidate_runs) != len(reference_runs) or not candidate_runs:
        raise ValueError("candidate and reference run lists must have equal non-zero length")
    if output_dir.exists():
        raise FileExistsError(f"SID analysis output already exists: {output_dir}")
    pairs: list[dict[str, Any]] = []
    inputs: list[dict[str, str]] = []
    seen_splits: set[str] = set()
    for candidate_run, reference_run in zip(candidate_runs, reference_runs, strict=True):
        if not (candidate_run / "SUCCESS").is_file() or not (reference_run / "SUCCESS").is_file():
            raise ValueError("paired SID analysis requires completed runs")
        manifests = [
            json.loads((run / "manifest.json").read_text())
            for run in (candidate_run, reference_run)
        ]
        split = str(manifests[0]["dataset"]["split"])
        if (
            manifests[0]["benchmark"]["name"] != "sid-bench"
            or manifests[1]["benchmark"]["name"] != "sid-bench"
        ):
            raise ValueError("paired SID analysis requires SID-Bench runs")
        if any(manifest["benchmark"].get("partial_run") is not False for manifest in manifests):
            raise ValueError("paired SID analysis excludes partial runs")
        if any(
            manifest["dataset"].get("evaluated_rows") != manifest["dataset"].get("source_rows")
            for manifest in manifests
        ):
            raise ValueError("paired SID analysis requires full dataset coverage")
        if split != manifests[1]["dataset"]["split"] or split in seen_splits:
            raise ValueError("paired SID analysis requires unique matching dataset splits")
        if manifests[0]["dataset"]["revision"] != manifests[1]["dataset"]["revision"]:
            raise ValueError("paired SID analysis dataset revisions do not match")
        seen_splits.add(split)
        prediction_paths = [run / "predictions.parquet" for run in (candidate_run, reference_run)]
        rows = [pq.read_table(path).to_pylist() for path in prediction_paths]
        analysis = analyze_sid_pair(
            rows[0],
            rows[1],
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        )
        pairs.append(
            {
                "split": split,
                "candidate_model": manifests[0]["model"]["name"],
                "reference_model": manifests[1]["model"]["name"],
                "candidate_run": str(candidate_run.resolve()),
                "reference_run": str(reference_run.resolve()),
                "analysis": analysis,
            }
        )
        for run, prediction_path in zip(
            (candidate_run, reference_run), prediction_paths, strict=True
        ):
            for path in (run / "manifest.json", run / "policy.json", prediction_path):
                inputs.append({"path": str(path.resolve()), "sha256": _sha256(path)})

    output_dir.mkdir(parents=True)
    result = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "calibration": "none; all input policies remain frozen",
        "pairs": pairs,
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    (output_dir / "report.md").write_text(_render_report(pairs))
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": result["created_at"],
                "analysis_method": "fixed-policy paired clip bootstrap",
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
