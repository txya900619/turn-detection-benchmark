from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SUMMARY_FIELDS = (
    "benchmark",
    "language",
    "task",
    "model",
    "row_role",
    "channel_mode",
    "execution_mode",
    "signal_provenance",
    "model_license",
    "threshold",
    "false_cutoff_at_300ms",
    "false_cutoff_at_600ms",
    "latency_at_5pct_cutoff_ms",
    "latency_at_10pct_cutoff_ms",
    "recall",
    "false_positive_rate",
    "strict_fir",
    "latency_p50_ms",
    "latency_p90_ms",
    "latency_p95_ms",
    "average_penalty_time_s",
    "nonbreak_false_positive_rate",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "complete_precision",
    "complete_recall",
    "complete_f1",
    "incomplete_precision",
    "incomplete_recall",
    "incomplete_f1",
    "backchannel_precision",
    "backchannel_recall",
    "backchannel_f1",
    "wait_precision",
    "wait_recall",
    "wait_f1",
    "incomplete_to_complete_error_rate",
    "backchannel_to_complete_error_rate",
    "abstention_rate",
    "rtf",
    "process_cpu_percent",
    "ram_peak_bytes",
    "gpu_vram_peak_bytes",
    "model_artifact_size_bytes",
    "run_path",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _empty_row(**values: object) -> dict[str, object]:
    row = dict.fromkeys(SUMMARY_FIELDS)
    unknown = set(values) - set(SUMMARY_FIELDS)
    if unknown:
        raise ValueError(f"unknown comparison fields: {sorted(unknown)}")
    row.update(values)
    return row


def _model_license(model: dict[str, Any]) -> str:
    value = model.get("model_license", model.get("license"))
    return "N/A" if value in (None, "") else str(value)


def _select_latest(
    candidates: list[tuple[Path, dict[str, Any]]],
    *,
    identity,
    canonical_bonus=lambda path: 0,
) -> tuple[list[tuple[Path, dict[str, Any]]], list[dict[str, str]]]:
    grouped: dict[object, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for candidate in candidates:
        grouped[identity(candidate[1])].append(candidate)
    selected: list[tuple[Path, dict[str, Any]]] = []
    rejected: list[dict[str, str]] = []
    for group in grouped.values():
        ordered = sorted(
            group,
            key=lambda item: (
                str(item[1].get("created_at", "")),
                canonical_bonus(item[0]),
                str(item[0]),
            ),
        )
        winner = ordered[-1]
        selected.append(winner)
        rejected.extend(
            {"path": str(path.resolve()), "reason": "superseded canonical run"}
            for path, _ in ordered[:-1]
        )
    return sorted(selected, key=lambda item: str(item[0])), rejected


def _discover_eot(
    results_root: Path, languages: set[str]
) -> tuple[list[dict[str, object]], list[dict[str, str]], list[Path]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    rejected: list[dict[str, str]] = []
    for path in sorted(results_root.glob("*/eotbench/**/manifest.json")):
        run = path.parent
        manifest = _load_json(path)
        benchmark = manifest.get("benchmark", {})
        language = benchmark.get("language")
        if language not in languages:
            continue
        reason = None
        if benchmark.get("name") != "livekit-eot-bench":
            reason = "unexpected benchmark"
        elif benchmark.get("partial_run") is not False:
            reason = "partial run"
        elif benchmark.get("primary_leaderboard_eligible") is not True:
            reason = "not primary-leaderboard eligible"
        elif not (run / "SUCCESS").is_file():
            reason = "missing SUCCESS"
        elif not (run / "metrics.json").is_file():
            reason = "missing metrics"
        elif manifest.get("dataset", {}).get("evaluated_rows") != manifest.get("dataset", {}).get(
            "source_rows"
        ):
            reason = "incomplete dataset coverage"
        if reason:
            rejected.append({"path": str(run.resolve()), "reason": reason})
        else:
            candidates.append((run, manifest))
    selected, superseded = _select_latest(
        candidates,
        identity=lambda manifest: (
            manifest["benchmark"]["language"],
            manifest["model"]["name"],
            json.dumps(manifest["evaluation_mode"], sort_keys=True),
        ),
    )
    rejected.extend(superseded)
    rows: list[dict[str, object]] = []
    inputs: list[Path] = []
    for run, manifest in selected:
        metrics_path = run / "metrics.json"
        metrics = _load_json(metrics_path)
        points = metrics["operating_points"]
        mode = manifest["evaluation_mode"]
        model = manifest["model"]
        diagnostic = str(model["name"]).startswith("fixed-timeout")
        rows.append(
            _empty_row(
                benchmark="livekit-eot-bench",
                language=manifest["benchmark"]["language"],
                task="eot_hold",
                model=model["name"],
                row_role="diagnostic" if diagnostic else "baseline",
                channel_mode=mode["channel_mode"],
                execution_mode=mode["execution_mode"],
                signal_provenance=mode["signal_provenance"],
                model_license=_model_license(model),
                false_cutoff_at_300ms=points["false_cutoff_at_300ms"]["false_cutoff_rate"],
                false_cutoff_at_600ms=points["false_cutoff_at_600ms"]["false_cutoff_rate"],
                latency_at_5pct_cutoff_ms=points["latency_at_5pct_cutoff"]["mean_latency_ms"],
                latency_at_10pct_cutoff_ms=points["latency_at_10pct_cutoff"]["mean_latency_ms"],
                run_path=str(run.resolve()),
            )
        )
        inputs.extend((run / "manifest.json", metrics_path))
    return rows, rejected, inputs


def _discover_sid(
    results_root: Path, languages: set[str], include_noise: bool
) -> tuple[list[dict[str, object]], list[dict[str, str]], list[Path]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    rejected: list[dict[str, str]] = []
    for path in sorted(results_root.glob("*/sidbench/**/manifest.json")):
        run = path.parent
        manifest = _load_json(path)
        benchmark = manifest.get("benchmark", {})
        split = manifest.get("dataset", {}).get("split")
        language = "noise" if split == "noise" else benchmark.get("language")
        if language not in languages and not (language == "noise" and include_noise):
            continue
        reason = None
        if benchmark.get("name") != "sid-bench":
            reason = "unexpected benchmark"
        elif benchmark.get("partial_run") is not False:
            reason = "partial run"
        elif benchmark.get("primary_leaderboard_eligible") is not True:
            reason = "not primary-leaderboard eligible"
        elif not (run / "SUCCESS").is_file():
            reason = "missing SUCCESS"
        elif not (run / "metrics.json").is_file():
            reason = "missing metrics"
        elif manifest.get("dataset", {}).get("evaluated_rows") != manifest.get("dataset", {}).get(
            "source_rows"
        ):
            reason = "incomplete dataset coverage"
        if reason:
            rejected.append({"path": str(run.resolve()), "reason": reason})
        else:
            manifest = {**manifest, "report_language": language}
            candidates.append((run, manifest))
    selected, superseded = _select_latest(
        candidates,
        identity=lambda manifest: (
            manifest["report_language"],
            manifest["model"]["name"],
            json.dumps(manifest["evaluation_mode"], sort_keys=True),
        ),
    )
    rejected.extend(superseded)
    rows: list[dict[str, object]] = []
    inputs: list[Path] = []
    for run, manifest in selected:
        metrics_path = run / "metrics.json"
        metrics = _load_json(metrics_path)
        mode = manifest["evaluation_mode"]
        model = manifest["model"]
        latency = metrics["response_latency_s"]
        rows.append(
            _empty_row(
                benchmark="sid-bench",
                language=manifest["report_language"],
                task="interruption",
                model=model["name"],
                row_role="baseline",
                channel_mode=mode["channel_mode"],
                execution_mode=mode["execution_mode"],
                signal_provenance=mode["signal_provenance"],
                model_license=_model_license(model),
                threshold=_load_json(run / "policy.json")["threshold"],
                recall=metrics["interruption_recall"],
                false_positive_rate=metrics["false_interruption_rate"],
                strict_fir=metrics["diagnostics"]["strict_paper_fir"],
                latency_p50_ms=(None if latency["p50"] is None else latency["p50"] * 1_000),
                latency_p90_ms=(None if latency["p90"] is None else latency["p90"] * 1_000),
                latency_p95_ms=(None if latency["p95"] is None else latency["p95"] * 1_000),
                average_penalty_time_s=metrics["average_penalty_time_s"],
                nonbreak_false_positive_rate=metrics["diagnostics"]["nonbreak_false_positive_rate"],
                run_path=str(run.resolve()),
            )
        )
        inputs.extend((run / "manifest.json", run / "policy.json", metrics_path))
    return rows, rejected, inputs


def _discover_fastturn(
    results_root: Path, enabled: bool
) -> tuple[list[dict[str, object]], list[dict[str, str]], list[Path]]:
    if not enabled:
        return [], [], []
    candidates: list[tuple[Path, dict[str, Any]]] = []
    rejected: list[dict[str, str]] = []
    suites = {
        "eot": ("complete", "incomplete"),
        "full": ("complete", "incomplete", "backchannel", "wait"),
    }
    for suite, labels in suites.items():
        root = results_root / "zh" / "fastturn" / suite
        for path in sorted(root.glob("**/manifest.json")):
            run = path.parent
            manifest = _load_json(path)
            benchmark = manifest.get("benchmark", {})
            model = manifest.get("model", {})
            counts = manifest.get("dataset", {}).get("class_counts", {})
            expected_rows = sum(int(counts.get(label, 0)) for label in labels)
            metrics_path = run / "metrics.json"
            runtime_path = run / "runtime.json"
            reason = None
            if benchmark.get("name") != f"fastturn-{suite}":
                reason = "unexpected benchmark"
            elif benchmark.get("partial_run") is not False:
                reason = "partial run"
            elif str(model.get("name", "")).startswith("fixed-state"):
                reason = "diagnostic model"
            elif not (run / "SUCCESS").is_file():
                reason = "missing SUCCESS"
            elif not metrics_path.is_file():
                reason = "missing metrics"
            elif not runtime_path.is_file():
                reason = "missing runtime"
            elif (
                not expected_rows
                or manifest.get("dataset", {}).get("evaluated_rows") != expected_rows
            ):
                reason = f"incomplete {suite} subset coverage"
            elif _load_json(metrics_path).get("labels") != list(labels):
                reason = f"unexpected {suite} label set"
            if reason:
                rejected.append({"path": str(run.resolve()), "reason": reason})
            else:
                candidates.append((run, {**manifest, "report_suite": suite}))
    selected, superseded = _select_latest(
        candidates,
        identity=lambda manifest: (
            manifest["report_suite"],
            manifest["model"]["name"],
            json.dumps(manifest["evaluation_mode"], sort_keys=True),
        ),
    )
    rejected.extend(superseded)
    rows: list[dict[str, object]] = []
    inputs: list[Path] = []
    for run, manifest in selected:
        metrics_path = run / "metrics.json"
        runtime_path = run / "runtime.json"
        metrics = _load_json(metrics_path)
        runtime = _load_json(runtime_path)
        complete = metrics["per_class"]["complete"]
        incomplete = metrics["per_class"]["incomplete"]
        backchannel = metrics["per_class"].get("backchannel")
        wait = metrics["per_class"].get("wait")
        mode = manifest["evaluation_mode"]
        model = manifest["model"]
        suite = manifest["report_suite"]
        mapping_path = run / "label_mapping.json"
        mapping = _load_json(mapping_path) if mapping_path.is_file() else {}
        rows.append(
            _empty_row(
                benchmark=f"fastturn-{suite}",
                language="zh",
                task="turn_state_2state" if suite == "eot" else "turn_state_4state",
                model=model["name"],
                row_role="baseline",
                channel_mode=mode["channel_mode"],
                execution_mode=mode["execution_mode"],
                signal_provenance=mode["signal_provenance"],
                model_license=_model_license(model),
                threshold=mapping.get("threshold"),
                accuracy=metrics["accuracy"],
                macro_f1=metrics["macro_f1"],
                weighted_f1=metrics["weighted_f1"],
                balanced_accuracy=metrics["balanced_accuracy"],
                complete_precision=complete["precision"],
                complete_recall=complete["recall"],
                complete_f1=complete["f1"],
                incomplete_precision=incomplete["precision"],
                incomplete_recall=incomplete["recall"],
                incomplete_f1=incomplete["f1"],
                backchannel_precision=(None if backchannel is None else backchannel["precision"]),
                backchannel_recall=(None if backchannel is None else backchannel["recall"]),
                backchannel_f1=(None if backchannel is None else backchannel["f1"]),
                wait_precision=None if wait is None else wait["precision"],
                wait_recall=None if wait is None else wait["recall"],
                wait_f1=None if wait is None else wait["f1"],
                incomplete_to_complete_error_rate=metrics["critical_errors"][
                    "incomplete_to_complete"
                ]["rate"],
                backchannel_to_complete_error_rate=metrics["critical_errors"]
                .get("backchannel_to_complete", {})
                .get("rate"),
                abstention_rate=metrics["abstentions"]["rate"],
                latency_p50_ms=metrics["inference_latency_ms"]["p50"],
                latency_p90_ms=metrics["inference_latency_ms"]["p90"],
                latency_p95_ms=metrics["inference_latency_ms"]["p95"],
                rtf=runtime["rtf"],
                ram_peak_bytes=runtime["ram_peak_bytes"],
                gpu_vram_peak_bytes=runtime["gpu_vram_peak_bytes"],
                model_artifact_size_bytes=runtime["model_artifact_size_bytes"],
                run_path=str(run.resolve()),
            )
        )
        inputs.extend((run / "manifest.json", metrics_path, runtime_path))
        if mapping_path.is_file():
            inputs.append(mapping_path)
    return rows, rejected, inputs


def _turnbench_bonus(path: Path) -> int:
    return 2 if path.name == "final" else 1 if path.name == "merged" else 0


def _discover_turnbench(
    results_root: Path, enabled: bool
) -> tuple[list[dict[str, object]], list[dict[str, str]], list[Path]]:
    if not enabled:
        return [], [], []
    candidates: list[tuple[Path, dict[str, Any]]] = []
    rejected: list[dict[str, str]] = []
    for path in sorted((results_root / "en" / "turnbench").glob("**/metadata.json")):
        run = path.parent
        metadata = _load_json(path)
        score_path = run / "scored" / "scores.json"
        if not score_path.is_file():
            score_path = run / "scores.json"
        benchmark = metadata.get("benchmark", {})
        reason = None
        if benchmark.get("name") != "turnbench":
            reason = "unexpected benchmark"
        elif benchmark.get("leaderboard_eligible") is not True:
            reason = "not leaderboard eligible"
        elif benchmark.get("conversation_count") != 38:
            reason = "incomplete conversation coverage"
        elif not score_path.is_file():
            reason = "missing official scores"
        if reason:
            rejected.append({"path": str(run.resolve()), "reason": reason})
        else:
            metadata = {**metadata, "score_path": str(score_path)}
            candidates.append((run, metadata))
    selected, superseded = _select_latest(
        candidates,
        identity=lambda metadata: (
            metadata["model"]["name"],
            metadata["evaluation"]["mode"],
        ),
        canonical_bonus=_turnbench_bonus,
    )
    rejected.extend(superseded)
    rows: list[dict[str, object]] = []
    inputs: list[Path] = []
    for run, metadata in selected:
        scores_path = Path(metadata["score_path"])
        scores = _load_json(scores_path)
        if scores.get("dataset_revision") != metadata["benchmark"].get("dataset_revision"):
            raise ValueError(f"TurnBench score/metadata revision mismatch: {run}")
        evaluation = metadata["evaluation"]
        model = metadata["model"]
        role = "diagnostic" if model["name"] == "turnbench-rms-vad" else "baseline"
        provenance = evaluation.get("signal_provenance", {})
        for task in ("eot", "interruption"):
            if "scores" in scores and "selection" in scores:
                score = scores["scores"][task]
                threshold = scores["selection"][task]["theta"]
            else:
                score = scores[task]
                threshold = evaluation.get("threshold")
            rows.append(
                _empty_row(
                    benchmark="turnbench",
                    language="en",
                    task=task,
                    model=model["name"],
                    row_role=role,
                    channel_mode=evaluation["mode"],
                    execution_mode=evaluation["execution"],
                    signal_provenance=(
                        provenance.get(task) if isinstance(provenance, dict) else provenance
                    ),
                    model_license=_model_license(model),
                    threshold=threshold,
                    recall=score["recall"],
                    false_positive_rate=score["fp_rate"],
                    latency_p50_ms=score["latency_ms"]["p50"],
                    latency_p90_ms=score["latency_ms"]["p90"],
                    run_path=str(run.resolve()),
                )
            )
        inputs.extend((run / "metadata.json", scores_path))
    return rows, rejected, inputs


def _runtime_rows(path: Path | None) -> tuple[list[dict[str, object]], list[Path]]:
    if path is None:
        return [], []
    summary = _load_json(path)
    rows = []
    for profile in summary.get("rows", []):
        rows.append(
            _empty_row(
                benchmark="runtime-profile",
                language="shared",
                task="runtime",
                model=profile["model"],
                row_role="deployment-profile",
                channel_mode="user-plus-silent-agent-shape",
                execution_mode=profile["backend"],
                signal_provenance="n/a",
                model_license="see benchmark rows",
                latency_p50_ms=profile["warm_p50_ms"],
                latency_p90_ms=profile["warm_p90_ms"],
                latency_p95_ms=profile["warm_p95_ms"],
                rtf=profile["rtf"],
                process_cpu_percent=profile["process_cpu_percent"],
                ram_peak_bytes=profile["ram_peak_bytes"],
                gpu_vram_peak_bytes=profile["gpu_vram_peak_bytes"],
                model_artifact_size_bytes=profile["model_artifact_size_bytes"],
                run_path=profile["profile"],
            )
        )
    if not rows:
        raise ValueError(f"runtime summary contains no rows: {path}")
    return rows, [path]


def _fastturn_coverage(results_root: Path, include_zh: bool) -> dict[str, object]:
    if not include_zh:
        return {"status": "not requested", "eligible_runs": 0, "excluded_runs": []}
    eligible: dict[str, list[str]] = {"eot": [], "full": []}
    excluded = []
    for path in sorted((results_root / "zh" / "fastturn").glob("**/manifest.json")):
        manifest = _load_json(path)
        benchmark = manifest.get("benchmark", {})
        model = manifest.get("model", {})
        benchmark_name = benchmark.get("name")
        suite = {"fastturn-eot": "eot", "fastturn-full": "full"}.get(benchmark_name)
        if (
            suite is not None
            and benchmark.get("partial_run") is False
            and not str(model.get("name", "")).startswith("fixed-state")
            and (path.parent / "SUCCESS").is_file()
        ):
            eligible[suite].append(str(path.parent.resolve()))
        else:
            if str(model.get("name", "")).startswith("fixed-state"):
                reason = "diagnostic model"
            elif suite is None:
                reason = "unsupported FastTurn suite"
            elif benchmark.get("partial_run") is not False:
                reason = "partial run"
            else:
                reason = "missing SUCCESS"
            excluded.append({"path": str(path.parent.resolve()), "reason": reason})
    has_eot = bool(eligible["eot"])
    has_full = bool(eligible["full"])
    if has_eot and has_full:
        status = "binary EOT and native four-state coverage complete in separate tables"
    elif has_eot:
        status = "binary EOT complete; native four-state coverage gap"
    elif has_full:
        status = "native four-state complete; binary EOT comparison gap"
    else:
        status = "coverage gap: no full non-diagnostic model run"
    selected = [*eligible["eot"], *eligible["full"]]
    return {
        "status": status,
        "eligible_runs": len(selected),
        "eligible_by_suite": {suite: len(paths) for suite, paths in eligible.items()},
        "selected_runs": selected,
        "excluded_runs": excluded,
    }


def _percent(value: object) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.2f}%"


def _milliseconds(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.0f} ms"


def _seconds(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f} s"


def _decimal(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4g}"


def _mebibytes(value: object) -> str:
    return "n/a" if value is None else f"{float(value) / 1_048_576:.0f} MiB"


def _mode(row: dict[str, object]) -> str:
    return f"{row['channel_mode']} / {row['execution_mode']} / {row['signal_provenance']}"


def _render_markdown(
    rows: list[dict[str, object]], fastturn: dict[str, object], languages: set[str]
) -> str:
    lines = [
        "# Turn Detection Baseline Comparison",
        "",
        "No overall score is computed. Rows with different channel modes or signal provenance",
        "are not directly interchangeable. TurnBench thresholds are selected and scored on dev;",
        "SID-Bench policies are frozen externally; eot-bench reports its prescribed policy sweep.",
    ]
    eot = [row for row in rows if row["benchmark"] == "livekit-eot-bench"]
    for language in sorted(languages):
        subset = [row for row in eot if row["language"] == language]
        if not subset:
            continue
        lines.extend(
            (
                "",
                f"## EOT / HOLD — {language.upper()}",
                "",
                "| Model | Role | Mode | Cutoff @300 | Cutoff @600 | "
                "Latency @5% | Latency @10% | License |",
                "|---|---|---|---:|---:|---:|---:|---|",
            )
        )
        for row in sorted(subset, key=lambda item: str(item["model"])):
            lines.append(
                f"| {row['model']} | {row['row_role']} | {_mode(row)} | "
                f"{_percent(row['false_cutoff_at_300ms'])} | "
                f"{_percent(row['false_cutoff_at_600ms'])} | "
                f"{_milliseconds(row['latency_at_5pct_cutoff_ms'])} | "
                f"{_milliseconds(row['latency_at_10pct_cutoff_ms'])} | "
                f"{row['model_license']} |"
            )

    turnbench = [row for row in rows if row["benchmark"] == "turnbench"]
    if turnbench:
        lines.extend(
            (
                "",
                "## TurnBench EN dev",
                "",
                "| Model | Task | Role | Mode / signal | Recall | FPR | "
                "Latency p50 | Latency p90 | License |",
                "|---|---|---|---|---:|---:|---:|---:|---|",
            )
        )
        for row in sorted(turnbench, key=lambda item: (str(item["model"]), str(item["task"]))):
            lines.append(
                f"| {row['model']} | {row['task']} | {row['row_role']} | {_mode(row)} | "
                f"{_percent(row['recall'])} | {_percent(row['false_positive_rate'])} | "
                f"{_milliseconds(row['latency_p50_ms'])} | "
                f"{_milliseconds(row['latency_p90_ms'])} | {row['model_license']} |"
            )

    sid = [row for row in rows if row["benchmark"] == "sid-bench"]
    if sid:
        lines.extend(
            (
                "",
                "## SID-Bench interruption",
                "",
                "| Split | Model | Mode / signal | Recall | Official FIR | Strict FIR | "
                "IRL p50 | IRL p90 | APT | Non-break FPR | License |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
            )
        )
        for row in sorted(sid, key=lambda item: (str(item["language"]), str(item["model"]))):
            lines.append(
                f"| {str(row['language']).upper()} | {row['model']} | {_mode(row)} | "
                f"{_percent(row['recall'])} | {_percent(row['false_positive_rate'])} | "
                f"{_percent(row['strict_fir'])} | {_milliseconds(row['latency_p50_ms'])} | "
                f"{_milliseconds(row['latency_p90_ms'])} | "
                f"{_seconds(row['average_penalty_time_s'])} | "
                f"{_percent(row['nonbreak_false_positive_rate'])} | {row['model_license']} |"
            )

    fastturn_rows = [row for row in rows if row["benchmark"] == "fastturn-eot"]
    if fastturn_rows:
        lines.extend(
            (
                "",
                "## FastTurn ZH — binary EOT subset",
                "",
                "| Model | Mode | Threshold | Complete P/R/F1 | Incomplete P/R/F1 | Macro-F1 | "
                "Inc→Com | Abstain | p50 | RTF | License |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            )
        )
        for row in sorted(fastturn_rows, key=lambda item: str(item["model"])):
            lines.append(
                f"| {row['model']} | {_mode(row)} | {_decimal(row['threshold'])} | "
                f"{_percent(row['complete_precision'])} / "
                f"{_percent(row['complete_recall'])} / {_percent(row['complete_f1'])} | "
                f"{_percent(row['incomplete_precision'])} / "
                f"{_percent(row['incomplete_recall'])} / {_percent(row['incomplete_f1'])} | "
                f"{_percent(row['macro_f1'])} | "
                f"{_percent(row['incomplete_to_complete_error_rate'])} | "
                f"{_percent(row['abstention_rate'])} | "
                f"{_milliseconds(row['latency_p50_ms'])} | {float(row['rtf']):.3f} | "
                f"{row['model_license']} |"
            )

    fastturn_full = [row for row in rows if row["benchmark"] == "fastturn-full"]
    if fastturn_full:
        lines.extend(
            (
                "",
                "## FastTurn ZH — native four-state",
                "",
                "This table is not comparable to the binary EOT subset above.",
                "",
                "| Model | Mode | Complete F1 | Incomplete F1 | Backchannel F1 | Wait F1 | "
                "Macro-F1 | Inc→Com | BC→Com | Abstain | p50 | RTF | License |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            )
        )
        for row in sorted(fastturn_full, key=lambda item: str(item["model"])):
            lines.append(
                f"| {row['model']} | {_mode(row)} | {_percent(row['complete_f1'])} | "
                f"{_percent(row['incomplete_f1'])} | {_percent(row['backchannel_f1'])} | "
                f"{_percent(row['wait_f1'])} | {_percent(row['macro_f1'])} | "
                f"{_percent(row['incomplete_to_complete_error_rate'])} | "
                f"{_percent(row['backchannel_to_complete_error_rate'])} | "
                f"{_percent(row['abstention_rate'])} | "
                f"{_milliseconds(row['latency_p50_ms'])} | {float(row['rtf']):.3f} | "
                f"{row['model_license']} |"
            )

    runtime = [row for row in rows if row["benchmark"] == "runtime-profile"]
    if runtime:
        lines.extend(
            (
                "",
                "## Isolated runtime",
                "",
                "| Model | Backend | Warm p50 | Warm p90 | RTF | CPU | RAM | VRAM | Artifacts |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
        for row in runtime:
            lines.append(
                f"| {row['model']} | {row['execution_mode']} | "
                f"{_milliseconds(row['latency_p50_ms'])} | "
                f"{_milliseconds(row['latency_p90_ms'])} | {float(row['rtf']):.3f} | "
                f"{float(row['process_cpu_percent']):.1f}% | "
                f"{_mebibytes(row['ram_peak_bytes'])} | "
                f"{_mebibytes(row['gpu_vram_peak_bytes'])} | "
                f"{_mebibytes(row['model_artifact_size_bytes'])} |"
            )

    lines.extend(
        (
            "",
            "## Coverage gaps",
            "",
            f"FastTurn ZH: {fastturn['status']}. X2-Turn remains deferred; partial smoke runs and",
            "fixed-state diagnostics are excluded from baseline tables.",
            "",
        )
    )
    return "\n".join(lines)


def write_comparison_report(
    *,
    results_root: Path,
    output_dir: Path,
    languages: set[str],
    runtime_summary: Path | None,
) -> Path:
    if not languages or not languages <= {"en", "zh"}:
        raise ValueError("comparison languages must be a non-empty subset of en/zh")
    results_root = results_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"comparison output already exists: {output_dir}")
    eot, eot_rejected, eot_inputs = _discover_eot(results_root, languages)
    sid, sid_rejected, sid_inputs = _discover_sid(
        results_root, languages, include_noise=languages == {"en", "zh"}
    )
    turnbench, turnbench_rejected, turnbench_inputs = _discover_turnbench(
        results_root, enabled="en" in languages
    )
    runtime, runtime_inputs = _runtime_rows(runtime_summary)
    fastturn_rows, fastturn_rejected, fastturn_inputs = _discover_fastturn(
        results_root, enabled="zh" in languages
    )
    fastturn = _fastturn_coverage(results_root, include_zh="zh" in languages)
    fastturn_eot = [row for row in fastturn_rows if row["benchmark"] == "fastturn-eot"]
    fastturn_full = [row for row in fastturn_rows if row["benchmark"] == "fastturn-full"]
    rows = [*eot, *turnbench, *sid, *fastturn_rows, *runtime]
    if not rows:
        raise ValueError("comparison discovery selected no rows")
    inputs = sorted(
        {
            path.resolve()
            for path in [
                *eot_inputs,
                *sid_inputs,
                *turnbench_inputs,
                *fastturn_inputs,
                *runtime_inputs,
            ]
        }
    )
    rejected = {
        "eot_bench": eot_rejected,
        "turnbench": turnbench_rejected,
        "sid_bench": sid_rejected,
        "fastturn": fastturn_rejected,
    }
    temporary = output_dir.with_name(f"{output_dir.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    (temporary / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "no_overall_score": True,
                "languages": sorted(languages),
                "sections": {
                    "eot_bench": eot,
                    "turnbench": turnbench,
                    "sid_bench": sid,
                    "fastturn": fastturn_rows,
                    "fastturn_eot": fastturn_eot,
                    "fastturn_full": fastturn_full,
                    "runtime": runtime,
                    "fastturn_coverage": fastturn,
                },
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with (temporary / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (temporary / "report.md").write_text(
        _render_markdown(rows, fastturn, languages), encoding="utf-8"
    )
    (temporary / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "created_at": datetime.now(UTC).isoformat(),
                "results_root": str(results_root),
                "languages": sorted(languages),
                "selection_policy": {
                    "complete_primary_runs_only": True,
                    "turnbench_requires_official_scores_and_38_conversations": True,
                    "duplicate_identity": "benchmark + language + model + evaluation mode",
                    "duplicate_resolution": (
                        "latest created_at; canonical final directory wins ties"
                    ),
                    "single_overall_score": False,
                },
                "inputs": [{"path": str(path), "sha256": _sha256(path)} for path in inputs],
                "selected_row_count": len(rows),
                "rejected": rejected,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_dir)
    return output_dir / "report.md"
