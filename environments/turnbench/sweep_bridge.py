from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from turnbench.data import DEV_DATASET, DEV_REVISION, resolve_dataset
from turnbench.durations import load_durations
from turnbench.score import score_submission
from turnbench.submission import Submission
from turnbench.sweep import commit_events, load_probs, operating_point, sweep, validate_probs

SCORER_REVISION = "e02d547803fd0e45221afcaeb1cd24b61ae7568e"


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def task_score_payload(score):
    return {
        "tp": score.tp,
        "fn": score.fn,
        "fp": score.fp,
        "tn": score.tn,
        "recall": score.recall,
        "fp_rate": score.fp_rate,
        "latency_ms": asdict(score.latency()),
    }


def committed_by_id(probs, theta: float):
    return {
        entry.conversation_id: {
            "speaker_1": commit_events(entry.speaker_1.prob, probs.frame_rate_hz, theta),
            "speaker_2": commit_events(entry.speaker_2.prob, probs.frame_rate_hz, theta),
        }
        for entry in probs.probs
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep EOT/INT probabilities with the pinned official TurnBench scorer"
    )
    parser.add_argument("--eot-probs", type=Path, required=True)
    parser.add_argument("--int-probs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default=DEV_DATASET)
    parser.add_argument("--revision", default=DEV_REVISION)
    parser.add_argument("--fp-budget", type=float, default=0.10)
    args = parser.parse_args()

    if not 0 <= args.fp_budget <= 1:
        raise ValueError("--fp-budget must be in [0, 1]")

    eot_probs = load_probs(args.eot_probs)
    int_probs = load_probs(args.int_probs)
    if eot_probs.task != "eot" or int_probs.task != "int":
        raise ValueError("--eot-probs must have task=eot and --int-probs task=int")

    dataset = resolve_dataset(source=args.dataset, revision=args.revision, skip_audio=True)
    durations = load_durations("dev")
    validate_probs(eot_probs, durations)
    validate_probs(int_probs, durations)

    eot_ids = [entry.conversation_id for entry in eot_probs.probs]
    int_ids = [entry.conversation_id for entry in int_probs.probs]
    if set(eot_ids) != set(int_ids):
        raise ValueError("EOT and INT probability files cover different conversations")

    rows_eot = sweep(eot_probs, dataset)
    rows_int = sweep(int_probs, dataset)
    selected_eot = operating_point(rows_eot, fp_budget=args.fp_budget)
    selected_int = operating_point(rows_int, fp_budget=args.fp_budget)
    if selected_eot is None or selected_int is None:
        raise RuntimeError("no threshold satisfies the requested false-positive budget")

    eot_events = committed_by_id(eot_probs, selected_eot.theta)
    int_events = committed_by_id(int_probs, selected_int.theta)
    predictions = {
        "schema_version": 1,
        "predictions": [
            {
                "conversation_id": conversation_id,
                "speaker_1": {
                    "eot": eot_events[conversation_id]["speaker_1"],
                    "interruption": int_events[conversation_id]["speaker_1"],
                },
                "speaker_2": {
                    "eot": eot_events[conversation_id]["speaker_2"],
                    "interruption": int_events[conversation_id]["speaker_2"],
                },
            }
            for conversation_id in eot_ids
        ],
    }
    submission = Submission.model_validate(predictions)
    scores = score_submission(submission, dataset)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions-dev.json").write_text(
        json.dumps(predictions, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for filename, rows in (("sweep-eot.json", rows_eot), ("sweep-int.json", rows_int)):
        (output / filename).write_text(
            json.dumps(json_safe([asdict(row) for row in rows]), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    summary = {
        "scorer_revision": SCORER_REVISION,
        "dataset": args.dataset,
        "dataset_revision": args.revision,
        "selection": {
            "split": "dev",
            "objective": "highest_recall_with_fp_rate_at_or_below_budget",
            "fp_budget": args.fp_budget,
            "eot": asdict(selected_eot),
            "interruption": asdict(selected_int),
        },
        "scores": {
            "eot": task_score_payload(scores.task_eot),
            "interruption": task_score_payload(scores.task_int),
        },
    }
    (output / "scores.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe(summary), allow_nan=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
