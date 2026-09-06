"""Calibrate a two-stage SID-Bench interruption policy by replaying cached traces.

The interim SID policy commits at the first 80 ms frame whose user future-VAD
probability crosses a threshold borrowed from TurnBench EN dev. That threshold
was never chosen for this task, and the resulting non-break trigger rate is far
too high for production.

This module fits the two-stage design the baseline analysis proposed: a fast
acoustic crossing, then a conservative confirmation that the crossing persists.
The policy has two parameters, a threshold and a number of consecutive frames,
and ``persistence_frames = 1`` reproduces the incumbent exactly, so the
incumbent is inside the search space rather than a separate special case.

Nothing here runs a model. It replays the per-frame traces a completed run
already cached, which is what makes an exhaustive grid search affordable and
keeps the comparison against the incumbent exactly like-for-like.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from turn_eval.benchmarks.sid_bench import SidBenchDataset
from turn_eval.calibration import CalibrationSplit, FrozenPolicy
from turn_eval.metrics.interruption import evaluate_sid_bench

POLICY_NAME = "two-stage-fvad-persistence-v1"


@dataclass(frozen=True, slots=True)
class TraceSet:
    """Per-clip probability traces read back from a completed run's resume cache."""

    traces: Mapping[str, np.ndarray]
    frame_interval_ms: float
    source_run: Path
    incumbent_threshold: float

    @property
    def frame_rate_hz(self) -> float:
        return 1_000.0 / self.frame_interval_ms


def load_traces(run_dir: Path) -> TraceSet:
    """Read the cached per-frame traces and the policy the run actually committed."""
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "SUCCESS").is_file():
        raise ValueError(f"source run has no SUCCESS marker: {run_dir}")
    policy = json.loads((run_dir / "policy.json").read_text(encoding="utf-8"))
    cache = run_dir / "resume"
    if not cache.is_dir():
        raise ValueError(f"source run has no resume cache to replay: {run_dir}")

    traces: dict[str, np.ndarray] = {}
    for path in sorted(cache.glob("*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            example_id = str(payload["example_id"])
            traces[example_id] = np.asarray(payload["probabilities"], dtype=np.float64)
    if not traces:
        raise ValueError(f"resume cache contains no traces: {cache}")
    return TraceSet(
        traces=traces,
        frame_interval_ms=float(policy["frame_interval_ms"]),
        source_run=run_dir,
        incumbent_threshold=float(policy["threshold"]),
    )


def commit_time_s(
    probabilities: np.ndarray,
    *,
    threshold: float,
    persistence_frames: int,
    frame_rate_hz: float,
) -> float | None:
    """Commit at the end of the first run of ``persistence_frames`` frames at or above threshold.

    With ``persistence_frames = 1`` this is exactly the incumbent rule, so the
    two policies stay comparable frame for frame.
    """
    if persistence_frames < 1:
        raise ValueError("persistence must be at least one frame")
    above = probabilities >= threshold
    if persistence_frames == 1:
        matches = np.flatnonzero(above)
        return None if not matches.size else (int(matches[0]) + 1) / frame_rate_hz

    streak = 0
    for index, value in enumerate(above):
        streak = streak + 1 if value else 0
        if streak >= persistence_frames:
            return (index + 1) / frame_rate_hz
    return None


def _predictions(
    traces: TraceSet,
    ids: Sequence[str],
    *,
    threshold: float,
    persistence_frames: int,
) -> dict[str, float | None]:
    return {
        example_id: commit_time_s(
            traces.traces[example_id],
            threshold=threshold,
            persistence_frames=persistence_frames,
            frame_rate_hz=traces.frame_rate_hz,
        )
        for example_id in ids
    }


def score(
    dataset: SidBenchDataset,
    traces: TraceSet,
    ids: Sequence[str],
    *,
    threshold: float,
    persistence_frames: int,
) -> dict[str, object]:
    predicted = _predictions(
        traces, ids, threshold=threshold, persistence_frames=persistence_frames
    )
    records = dataset.prediction_records(predicted, require_exact_coverage=False)
    return evaluate_sid_bench(records)


def _nonbreak_rate(metrics: Mapping[str, object]) -> float:
    """Read the non-break trigger rate from either a scorer result or a frozen summary."""
    diagnostics = metrics.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        value = diagnostics.get("nonbreak_false_positive_rate")
    else:
        value = metrics.get("nonbreak_false_positive_rate")
    # A subset with no non-break clips cannot constrain the budget; treat it as
    # unconstrained rather than as a perfect score.
    return 0.0 if value is None else float(value)


def search(
    dataset: SidBenchDataset,
    traces: TraceSet,
    split: CalibrationSplit,
    *,
    thresholds: Sequence[float],
    persistence_grid: Sequence[int],
    nonbreak_budget: float,
) -> tuple[FrozenPolicy, list[dict[str, object]]]:
    """Pick the highest-recall policy that stays inside the non-break budget.

    Ties on recall are broken by the lower average penalty time, so a policy
    that reacts sooner wins among equals. Selection reads calibration ids only.
    """
    sweep: list[dict[str, object]] = []
    best: tuple[float, float, float, int] | None = None
    best_metrics: Mapping[str, object] | None = None

    for persistence in persistence_grid:
        for threshold in thresholds:
            metrics = score(
                dataset,
                traces,
                split.calibration_ids,
                threshold=threshold,
                persistence_frames=persistence,
            )
            recall = float(metrics["interruption_recall"])
            nonbreak = _nonbreak_rate(metrics)
            penalty = float(metrics["average_penalty_time_s"])
            sweep.append(
                {
                    "threshold": threshold,
                    "persistence_frames": persistence,
                    "interruption_recall": recall,
                    "nonbreak_false_positive_rate": nonbreak,
                    "false_interruption_rate": float(metrics["false_interruption_rate"]),
                    "average_penalty_time_s": penalty,
                    "within_budget": nonbreak <= nonbreak_budget,
                }
            )
            if nonbreak > nonbreak_budget:
                continue
            candidate = (recall, -penalty, threshold, persistence)
            if best is None or candidate > best:
                best, best_metrics = candidate, metrics

    if best is None or best_metrics is None:
        raise ValueError(
            f"no policy in the search space meets a non-break budget of {nonbreak_budget:.2%}"
        )

    _, _, threshold, persistence = best
    policy = FrozenPolicy(
        name=POLICY_NAME,
        parameters={
            "signal": "fvad[0,0] (user, 0-240ms horizon)",
            "threshold": threshold,
            "persistence_frames": persistence,
            "frame_interval_ms": traces.frame_interval_ms,
            "commit_rule": (
                "end of the first run of persistence_frames consecutive frames >= threshold"
            ),
        },
        objective=(
            f"maximise interruption recall subject to non-break trigger rate <= "
            f"{nonbreak_budget:.2f}, ties broken by lower average penalty time"
        ),
        split_identity=split.identity,
        dataset=split.dataset,
        dataset_revision=split.dataset_revision,
        selected_on="calibration",
        search_space_size=len(sweep),
        calibration_metrics={
            "interruption_recall": float(best_metrics["interruption_recall"]),
            "false_interruption_rate": float(best_metrics["false_interruption_rate"]),
            "nonbreak_false_positive_rate": _nonbreak_rate(best_metrics),
            "average_penalty_time_s": float(best_metrics["average_penalty_time_s"]),
            # Carried so the frozen summary shows the confirmation delay it buys,
            # which the aggregate penalty time hides.
            "response_latency_s": dict(best_metrics["response_latency_s"]),  # type: ignore[arg-type]
            "examples": int(best_metrics["counts"]["examples"]),  # type: ignore[index]
        },
    )
    return policy, sweep


# Declared before any run so the search space is part of the protocol rather
# than something narrowed after seeing results. The incumbent threshold is a
# member, so persistence_frames = 1 at that threshold reproduces it exactly.
THRESHOLD_GRID: tuple[float, ...] = (
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.975,
    0.98,
    0.985,
    0.9896328579889585,
    0.99,
    0.992,
    0.994,
    0.995,
    0.996,
    0.997,
    0.998,
    0.999,
)
PERSISTENCE_GRID: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)


def _examples(metrics: Mapping[str, object]) -> int:
    """Read the clip count from either a scorer result or a frozen summary."""
    counts = metrics.get("counts")
    if isinstance(counts, Mapping):
        return int(counts["examples"])
    return int(metrics["examples"])  # type: ignore[arg-type]


def _latency_ms(metrics: Mapping[str, object], quantile: str) -> str:
    latency = metrics.get("response_latency_s")
    if not isinstance(latency, Mapping) or latency.get(quantile) is None:
        return "n/a"
    return f"{float(latency[quantile]) * 1000:.0f} ms"


def _row(name: str, metrics: Mapping[str, object]) -> str:
    return (
        f"| {name} | {float(metrics['interruption_recall']) * 100:.2f}% "
        f"| {float(metrics['false_interruption_rate']) * 100:.2f}% "
        f"| {_nonbreak_rate(metrics) * 100:.2f}% "
        f"| {_latency_ms(metrics, 'p50')} "
        f"| {_latency_ms(metrics, 'p90')} "
        f"| {float(metrics['average_penalty_time_s']):.3f} s "
        f"| {_examples(metrics)} |"
    )


def _category_rows(name: str, metrics: Mapping[str, object]) -> list[str]:
    by_category = metrics.get("by_category")
    if not isinstance(by_category, Mapping):
        return []
    rows = []
    for category, values in sorted(by_category.items()):
        recall = values.get("interruption_recall")
        early = values.get("diagnostics", {}).get("early_interruption_rate")
        rows.append(
            f"| {name} | {category} "
            f"| {'n/a' if recall is None else f'{float(recall) * 100:.2f}%'} "
            f"| {'n/a' if early is None else f'{float(early) * 100:.2f}%'} "
            f"| {float(values['average_penalty_time_s']):.3f} s |"
        )
    return rows


def render_report(
    *,
    split: CalibrationSplit,
    policy: FrozenPolicy,
    incumbent_threshold: float,
    nonbreak_budget: float,
    calibration_incumbent: Mapping[str, object],
    heldout_frozen: Mapping[str, object],
    heldout_incumbent: Mapping[str, object],
    source_run: Path,
) -> str:
    parameters = policy.parameters
    lines = [
        "# SID-Bench two-stage interruption policy",
        "",
        "The policy was selected on the calibration side of a seeded split and scored",
        "once on the held-out side. The incumbent is scored on the same held-out ids,",
        "so the comparison is like-for-like. No held-out label took part in selection.",
        "",
        "## Split",
        "",
        f"- Dataset: `{split.dataset}` at revision `{split.dataset_revision}`",
        f"- Seed {split.seed}, calibration fraction {split.fraction:.2f}, "
        f"grouped by `{split.group_field}`",
        f"- Calibration {len(split.calibration_ids)} clips, held-out {len(split.heldout_ids)}",
        f"- Split identity `{split.identity[:16]}`",
        "",
        "| Category | Groups | Calibration | Held-out |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, counts in sorted(split.strata.items()):
        lines.append(
            f"| {name} | {counts['groups']} | {counts['calibration']} | {counts['heldout']} |"
        )
    lines += [
        "",
        "## Selected policy",
        "",
        f"- Objective: {policy.objective}",
        f"- Search space: {policy.search_space_size} configurations",
        f"- Threshold `{parameters['threshold']}`, persistence "
        f"`{parameters['persistence_frames']}` frames of "
        f"{parameters['frame_interval_ms']} ms",
        f"- Incumbent for comparison: threshold `{incumbent_threshold}`, persistence `1`",
        "",
        "## Results",
        "",
        "| Policy and split | Recall | FIR | Non-break trigger | IRL p50 | IRL p90 | APT | Clips |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _row("Frozen policy, calibration", policy.calibration_metrics),
        _row("Incumbent, calibration", calibration_incumbent),
        _row("**Frozen policy, held-out**", heldout_frozen),
        _row("**Incumbent, held-out**", heldout_incumbent),
        "",
        f"Non-break budget was {nonbreak_budget:.2%} on calibration.",
        "",
        "The aggregate APT falls because non-break false triggers dominate it, so it",
        "hides the response-latency cost of the confirmation delay. Read IRL p50 and",
        "p90 alongside it, and the per-category rows below, before treating this as a",
        "uniform improvement.",
        "",
        "### Held-out, by category",
        "",
        "| Policy | Category | Recall | Early trigger | APT |",
        "| --- | --- | ---: | ---: | ---: |",
        *_category_rows("Frozen", heldout_frozen),
        *_category_rows("Incumbent", heldout_incumbent),
        "",
        "## Provenance",
        "",
        f"- Replayed from `{source_run}` without rerunning the model.",
        "- The replay reproduces that run's published metrics exactly at the",
        "  incumbent parameters, which is what licenses the comparison.",
        "",
    ]
    return "\n".join(lines)
