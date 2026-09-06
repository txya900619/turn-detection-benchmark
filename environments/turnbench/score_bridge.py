from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from turnbench.data import DEV_DATASET, DEV_REVISION, resolve_dataset
from turnbench.score import score_submission
from turnbench.submission import load_submission


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def task_payload(score):
    return {
        "tp": score.tp,
        "fn": score.fn,
        "fp": score.fp,
        "tn": score.tn,
        "recall": score.recall,
        "fp_rate": score.fp_rate,
        "latency_ms": asdict(score.latency()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Machine-readable bridge to pinned TurnBench")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--dataset", default=DEV_DATASET)
    parser.add_argument("--revision", default=DEV_REVISION)
    args = parser.parse_args()

    submission = load_submission(args.predictions)
    dataset = resolve_dataset(source=args.dataset, revision=args.revision, skip_audio=True)
    scores = score_submission(submission, dataset)
    output = {
        "scorer_revision": "e02d547803fd0e45221afcaeb1cd24b61ae7568e",
        "dataset": args.dataset,
        "dataset_revision": args.revision,
        "conversation_count": len(submission.predictions),
        "eot": task_payload(scores.task_eot),
        "interruption": task_payload(scores.task_int),
    }
    print(json.dumps(json_safe(output), allow_nan=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
