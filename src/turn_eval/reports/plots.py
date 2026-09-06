from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_MODEL_LABELS = {
    "dualturn-endpointing": "DualTurn",
    "fixed-timeout-oracle-vad": "Oracle-VAD timeout",
    "maai-realtime-vap-trilingual-cpc-10hz-5s": "Realtime-VAP",
    "smart-turn-v3.2-cpu": "SmartTurn v3.2",
}
_COLORS = {
    "dualturn-endpointing": "#0072B2",
    "fixed-timeout-oracle-vad": "#777777",
    "maai-realtime-vap-trilingual-cpc-10hz-5s": "#D55E00",
    "smart-turn-v3.2-cpu": "#009E73",
}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_label(model: str) -> str:
    return _MODEL_LABELS.get(model, model)


def extract_plot_data(summary_path: Path) -> tuple[dict[str, Any], list[Path]]:
    """Load canonical report rows without recalibrating any model policy."""
    summary_path = summary_path.resolve()
    summary = _load_object(summary_path)
    if summary.get("no_overall_score") is not True:
        raise ValueError("comparison summary must explicitly prohibit an overall score")
    sections = summary.get("sections")
    if not isinstance(sections, dict):
        raise ValueError("comparison summary is missing sections")

    inputs = [summary_path]
    eot: list[dict[str, Any]] = []
    for row in sections.get("eot_bench", []):
        run_path = Path(row["run_path"])
        metrics_path = run_path / "metrics.json"
        metrics = _load_object(metrics_path)
        frontier = metrics.get("pareto_frontier")
        if not isinstance(frontier, list) or not frontier:
            raise ValueError(f"missing eot-bench Pareto frontier: {metrics_path}")
        points = []
        for point in frontier:
            latency = float(point["mean_latency_ms"])
            cutoff = float(point["false_cutoff_rate"])
            if latency < 0 or not 0 <= cutoff <= 1:
                raise ValueError(f"invalid eot-bench Pareto point: {metrics_path}")
            points.append(
                {
                    "mean_latency_ms": latency,
                    "false_cutoff_rate": cutoff,
                    "threshold": float(point["threshold"]),
                    "action_delay_ms": float(point["action_delay_ms"]),
                    "timeout_ms": float(point["timeout_ms"]),
                }
            )
        eot.append(
            {
                "language": row["language"],
                "model": row["model"],
                "label": _model_label(str(row["model"])),
                "row_role": row["row_role"],
                "channel_mode": row["channel_mode"],
                "signal_provenance": row["signal_provenance"],
                "frontier": sorted(points, key=lambda point: point["mean_latency_ms"]),
                "operating_points": metrics["operating_points"],
                "source": str(metrics_path.resolve()),
            }
        )
        inputs.append(metrics_path.resolve())

    # These are one frozen operating point per model/split. Deliberately do not
    # load predictions or run a threshold sweep on SID-Bench test data.
    sid = []
    for row in sections.get("sid_bench", []):
        sid.append(
            {
                key: row.get(key)
                for key in (
                    "language",
                    "model",
                    "channel_mode",
                    "signal_provenance",
                    "threshold",
                    "recall",
                    "false_positive_rate",
                    "latency_p50_ms",
                    "latency_p90_ms",
                    "average_penalty_time_s",
                    "nonbreak_false_positive_rate",
                    "run_path",
                )
            }
            | {"label": _model_label(str(row["model"]))}
        )

    if not eot and not sid:
        raise ValueError("comparison summary has no plottable EOT or SID rows")
    return (
        {
            "schema_version": 1,
            "source_summary": str(summary_path),
            "policy_notes": {
                "eot_bench": "official threshold/action-delay/timeout Pareto sweep",
                "sid_bench": (
                    "one externally frozen policy per model; no SID test-set threshold sweep"
                ),
            },
            "eot_bench": eot,
            "sid_bench": sid,
        },
        sorted(set(inputs)),
    )


def _style_axes(axis: Any) -> None:
    axis.grid(True, color="#d8dde3", linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _render_eot(data: dict[str, Any], language: str, destination: Path) -> bool:
    rows = [row for row in data["eot_bench"] if row["language"] == language]
    if not rows:
        return False
    figure, axis = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    for row in sorted(rows, key=lambda item: (item["row_role"] == "diagnostic", item["label"])):
        model = str(row["model"])
        points = row["frontier"]
        mode = f"{row['channel_mode']}, {row['signal_provenance']}"
        diagnostic = row["row_role"] == "diagnostic"
        axis.plot(
            [point["mean_latency_ms"] for point in points],
            [point["false_cutoff_rate"] * 100 for point in points],
            color=_COLORS.get(model),
            linewidth=2.2 if not diagnostic else 1.7,
            linestyle="--" if diagnostic else "-",
            alpha=0.9,
            label=f"{row['label']} ({mode})",
        )
        for marker, key in (("o", "false_cutoff_at_300ms"), ("s", "false_cutoff_at_600ms")):
            point = row["operating_points"][key]
            axis.scatter(
                [point["mean_latency_ms"]],
                [point["false_cutoff_rate"] * 100],
                marker=marker,
                s=42,
                color=_COLORS.get(model),
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
    _style_axes(axis)
    axis.set_title(f"EOT / HOLD trade-off — {language.upper()}", fontsize=15, loc="left")
    axis.set_xlabel("Mean conversational endpointing latency (ms)  ↓")
    axis.set_ylabel("False cutoff rate (%)  ↓")
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False, fontsize=8.5, loc="upper right")
    axis.text(
        0,
        -0.17,
        "Lower-left is better. ○ prescribed 300 ms point; □ prescribed 600 ms point. "
        "Dashed oracle-VAD timeout is diagnostic.",
        transform=axis.transAxes,
        fontsize=8.5,
        color="#4b5563",
    )
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)
    return True


def _render_sid(data: dict[str, Any], destination: Path) -> bool:
    rows = [
        row
        for row in data["sid_bench"]
        if row["language"] in {"en", "zh"}
        and row["recall"] is not None
        and row["false_positive_rate"] is not None
    ]
    if not rows:
        return False
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 5.4), sharey=True)
    for axis, language in zip(axes, ("en", "zh"), strict=True):
        subset = [row for row in rows if row["language"] == language]
        for row in subset:
            model = str(row["model"])
            latency = row["latency_p50_ms"]
            size = 70 if latency is None else max(65, min(230, 55 + float(latency) * 0.45))
            axis.scatter(
                float(row["false_positive_rate"]) * 100,
                float(row["recall"]) * 100,
                s=size,
                color=_COLORS.get(model),
                edgecolor="white",
                linewidth=1.0,
                zorder=3,
            )
            annotation = row["label"]
            if latency is not None:
                annotation += f"\np50 {float(latency):.0f} ms"
            axis.annotate(
                annotation,
                (float(row["false_positive_rate"]) * 100, float(row["recall"]) * 100),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=8.5,
            )
        _style_axes(axis)
        axis.set_title(language.upper(), fontsize=13)
        axis.set_xlabel("Official false interruption rate (%)  ↓")
        axis.set_xlim(0, 100)
        axis.set_ylim(0, 100)
    axes[0].set_ylabel("Interruption recall (%)  ↑")
    figure.suptitle("SID-Bench frozen-policy trade-off", fontsize=15, x=0.01, y=0.97, ha="left")
    figure.subplots_adjust(left=0.06, right=0.99, bottom=0.18, top=0.87, wspace=0.06)
    figure.text(
        0.01,
        0.025,
        "Upper-left is better. Bubble size encodes IRL p50. Policies were frozen externally; "
        "this is not a test-tuned Pareto frontier.",
        fontsize=8.5,
        color="#4b5563",
    )
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)
    return True


def write_suite_plots(summary_path: Path, output_dir: Path) -> Path:
    data, inputs = extract_plot_data(summary_path)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"plot output already exists: {output_dir}")
    temporary = output_dir.with_name(f"{output_dir.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    images = []
    for language in ("en", "zh"):
        filename = f"eot_pareto_{language}.png"
        if _render_eot(data, language, temporary / filename):
            images.append(filename)
    if _render_sid(data, temporary / "sid_interruption_tradeoff.png"):
        images.append("sid_interruption_tradeoff.png")
    (temporary / "plot_data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (temporary / "README.md").write_text(
        "# Benchmark figures\n\n"
        "EOT images plot each selected run's complete official policy Pareto frontier. "
        "The SID image compares one externally frozen operating point per model and split; "
        "it does not tune or sweep thresholds on SID-Bench test data. Bubble area encodes "
        "interruption response latency p50. No overall score is computed.\n",
        encoding="utf-8",
    )
    (temporary / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": datetime.now(UTC).isoformat(),
                "source_summary": str(summary_path.resolve()),
                "inputs": [{"path": str(path), "sha256": _sha256(path)} for path in inputs],
                "images": images,
                "sid_threshold_policy": "external-frozen-only; no test-set sweep",
                "single_overall_score": False,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_dir)
    return output_dir
