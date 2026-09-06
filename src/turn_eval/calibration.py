"""Seeded calibration splits and frozen policy artifacts.

None of the benchmarks in this repository publishes a development split, so a
policy with any free parameter has nowhere legitimate to be tuned. This module
supplies the missing piece: a deterministic, persisted split of a test set into
a calibration part and a held-out part, and an immutable artifact recording the
parameters chosen on the calibration part.

Two rules make the result defensible, and both are enforced here rather than
left to the caller.

*Grouping.* The split is drawn over groups, not examples, so no conversation or
clip can contribute to both sides. For SID-Bench each clip is its own group; for
eot-bench the group is the conversation.

*Direction.* A frozen policy records the split it was selected on, and scoring
code refuses to score a held-out set with a policy frozen on anything else. A
held-out number is only meaningful if the policy provably predates it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from turn_eval.core.hashing import canonical_json, content_hash

SPLIT_SCHEMA_VERSION = 1
POLICY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CalibrationSplit:
    """A deterministic partition of one dataset split into calibration and held-out."""

    dataset: str
    dataset_revision: str
    seed: int
    fraction: float
    group_field: str
    calibration_ids: tuple[str, ...]
    heldout_ids: tuple[str, ...]
    strata: Mapping[str, Mapping[str, int]]

    def __post_init__(self) -> None:
        if not 0 < self.fraction < 1:
            raise ValueError("calibration fraction must lie strictly between zero and one")
        overlap = set(self.calibration_ids) & set(self.heldout_ids)
        if overlap:
            raise ValueError(f"calibration and held-out sets overlap on {len(overlap)} ids")
        if not self.calibration_ids or not self.heldout_ids:
            raise ValueError("both sides of a calibration split must be non-empty")

    @property
    def identity(self) -> str:
        """Hash of the exact membership, so a policy can be pinned to this split."""
        return content_hash(
            {
                "dataset": self.dataset,
                "dataset_revision": self.dataset_revision,
                "calibration_ids": sorted(self.calibration_ids),
                "heldout_ids": sorted(self.heldout_ids),
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "dataset": self.dataset,
            "dataset_revision": self.dataset_revision,
            "seed": self.seed,
            "fraction": self.fraction,
            "group_field": self.group_field,
            "counts": {
                "calibration": len(self.calibration_ids),
                "heldout": len(self.heldout_ids),
            },
            "strata": {name: dict(counts) for name, counts in self.strata.items()},
            "split_identity": self.identity,
            "calibration_ids": sorted(self.calibration_ids),
            "heldout_ids": sorted(self.heldout_ids),
        }


def _stable_rank(seed: int, dataset: str, group_id: str) -> str:
    """Rank a group by a hash of its identity, so membership never depends on order.

    Using the id rather than a sequential draw means the split is reproducible
    from the seed and the id alone: re-deriving it does not require replaying the
    dataset in the same order, and adding an unrelated group elsewhere in the
    dataset cannot reshuffle the rest.
    """
    payload = f"{seed}:{dataset}:{group_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def make_calibration_split(
    *,
    dataset: str,
    dataset_revision: str,
    groups: Mapping[str, str],
    seed: int,
    fraction: float,
    group_field: str,
) -> CalibrationSplit:
    """Partition groups into calibration and held-out, stratified by stratum label.

    ``groups`` maps a group id to its stratum label. Each stratum is split
    independently at the requested fraction, so a rare category cannot land
    entirely on one side.
    """
    if not groups:
        raise ValueError("cannot split an empty group set")

    by_stratum: dict[str, list[str]] = {}
    for group_id, stratum in groups.items():
        by_stratum.setdefault(stratum, []).append(group_id)

    calibration: list[str] = []
    heldout: list[str] = []
    strata: dict[str, dict[str, int]] = {}
    for stratum, members in sorted(by_stratum.items()):
        ordered = sorted(members, key=lambda gid: _stable_rank(seed, dataset, gid))
        take = round(len(ordered) * fraction)
        # Never let a stratum contribute to only one side while it has the room
        # to contribute to both; a stratum of one is reported rather than split.
        if len(ordered) >= 2:
            take = min(max(take, 1), len(ordered) - 1)
        chosen, rest = ordered[:take], ordered[take:]
        calibration.extend(chosen)
        heldout.extend(rest)
        strata[stratum] = {
            "groups": len(ordered),
            "calibration": len(chosen),
            "heldout": len(rest),
        }

    return CalibrationSplit(
        dataset=dataset,
        dataset_revision=dataset_revision,
        seed=seed,
        fraction=fraction,
        group_field=group_field,
        calibration_ids=tuple(sorted(calibration)),
        heldout_ids=tuple(sorted(heldout)),
        strata=strata,
    )


def write_calibration_split(path: Path, split: CalibrationSplit) -> Path:
    """Persist a split immutably; an existing file is never silently replaced."""
    if path.exists():
        raise FileExistsError(f"calibration split already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = split.to_dict()
    payload["created_at"] = datetime.now(UTC).isoformat()
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return path


def load_calibration_split(path: Path) -> CalibrationSplit:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError(f"unsupported calibration split schema: {payload.get('schema_version')}")
    split = CalibrationSplit(
        dataset=payload["dataset"],
        dataset_revision=payload["dataset_revision"],
        seed=payload["seed"],
        fraction=payload["fraction"],
        group_field=payload["group_field"],
        calibration_ids=tuple(payload["calibration_ids"]),
        heldout_ids=tuple(payload["heldout_ids"]),
        strata=payload["strata"],
    )
    if split.identity != payload["split_identity"]:
        raise ValueError(f"calibration split membership does not match its identity: {path}")
    return split


@dataclass(frozen=True, slots=True)
class FrozenPolicy:
    """Parameters selected on a calibration split, pinned to that split."""

    name: str
    parameters: Mapping[str, object]
    objective: str
    split_identity: str
    dataset: str
    dataset_revision: str
    selected_on: str
    search_space_size: int
    calibration_metrics: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": POLICY_SCHEMA_VERSION,
            "name": self.name,
            "parameters": dict(self.parameters),
            "objective": self.objective,
            "split_identity": self.split_identity,
            "dataset": self.dataset,
            "dataset_revision": self.dataset_revision,
            "selected_on": self.selected_on,
            "search_space_size": self.search_space_size,
            "calibration_metrics": dict(self.calibration_metrics),
            "heldout_labels_used_for_selection": False,
        }


def write_frozen_policy(path: Path, policy: FrozenPolicy) -> Path:
    if path.exists():
        raise FileExistsError(f"frozen policy already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = policy.to_dict()
    payload["created_at"] = datetime.now(UTC).isoformat()
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return path


def load_frozen_policy(path: Path) -> FrozenPolicy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError(f"unsupported frozen policy schema: {payload.get('schema_version')}")
    return FrozenPolicy(
        name=payload["name"],
        parameters=payload["parameters"],
        objective=payload["objective"],
        split_identity=payload["split_identity"],
        dataset=payload["dataset"],
        dataset_revision=payload["dataset_revision"],
        selected_on=payload["selected_on"],
        search_space_size=payload["search_space_size"],
        calibration_metrics=payload["calibration_metrics"],
    )


def require_heldout_evaluation(
    policy: FrozenPolicy,
    split: CalibrationSplit,
    evaluated_ids: Sequence[str],
) -> None:
    """Refuse to report a held-out score whose policy was not frozen beforehand.

    This is the check that makes a held-out number mean anything. It fails
    closed on three separate ways of accidentally leaking: a policy frozen on a
    different split, a policy frozen on the held-out side, and an evaluation set
    that reaches into calibration ids.
    """
    if policy.split_identity != split.identity:
        raise ValueError(
            "frozen policy was selected on a different calibration split; "
            f"policy {policy.split_identity[:12]} against split {split.identity[:12]}"
        )
    if policy.selected_on != "calibration":
        raise ValueError(f"policy was selected on {policy.selected_on!r}, not on calibration")
    leaked = set(evaluated_ids) & set(split.calibration_ids)
    if leaked:
        raise ValueError(
            f"held-out evaluation includes {len(leaked)} calibration ids; "
            "the reported score would be contaminated"
        )
