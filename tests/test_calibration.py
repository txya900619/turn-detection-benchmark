from __future__ import annotations

from pathlib import Path

import pytest

from turn_eval.calibration import (
    CalibrationSplit,
    FrozenPolicy,
    load_calibration_split,
    load_frozen_policy,
    make_calibration_split,
    require_heldout_evaluation,
    write_calibration_split,
    write_frozen_policy,
)


def _groups(count: int, strata: int = 2) -> dict[str, str]:
    return {f"clip-{index:04d}": f"stratum-{index % strata}" for index in range(count)}


def _split(**overrides: object) -> CalibrationSplit:
    kwargs: dict[str, object] = {
        "dataset": "sid-bench-zh",
        "dataset_revision": "rev-1",
        "groups": _groups(100),
        "seed": 0,
        "fraction": 0.3,
        "group_field": "example_id",
    }
    kwargs.update(overrides)
    return make_calibration_split(**kwargs)  # type: ignore[arg-type]


def test_split_is_deterministic_for_a_seed() -> None:
    assert _split().calibration_ids == _split().calibration_ids


def test_a_different_seed_gives_a_different_partition() -> None:
    assert _split(seed=0).calibration_ids != _split(seed=1).calibration_ids


def test_membership_does_not_depend_on_input_order() -> None:
    forward = _groups(60)
    reversed_order = dict(reversed(list(forward.items())))
    assert _split(groups=forward).identity == _split(groups=reversed_order).identity


def test_sides_are_disjoint_and_cover_every_group() -> None:
    split = _split()
    assert not set(split.calibration_ids) & set(split.heldout_ids)
    assert set(split.calibration_ids) | set(split.heldout_ids) == set(_groups(100))


def test_each_stratum_is_split_independently() -> None:
    # A rare stratum must reach both sides rather than landing wholly on one.
    groups = {**_groups(90, strata=1), "rare-a": "rare", "rare-b": "rare"}
    split = _split(groups=groups)
    calibration = set(split.calibration_ids)
    assert {"rare-a", "rare-b"} & calibration
    assert {"rare-a", "rare-b"} - calibration
    assert split.strata["rare"] == {"groups": 2, "calibration": 1, "heldout": 1}


def test_a_singleton_stratum_is_reported_rather_than_split() -> None:
    groups = {**_groups(40, strata=1), "only-one": "singleton"}
    split = _split(groups=groups)
    assert split.strata["singleton"]["groups"] == 1
    assert split.strata["singleton"]["calibration"] + split.strata["singleton"]["heldout"] == 1


def test_an_empty_side_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        CalibrationSplit(
            dataset="d",
            dataset_revision="r",
            seed=0,
            fraction=0.5,
            group_field="example_id",
            calibration_ids=(),
            heldout_ids=("a",),
            strata={},
        )


def test_overlapping_sides_are_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        CalibrationSplit(
            dataset="d",
            dataset_revision="r",
            seed=0,
            fraction=0.5,
            group_field="example_id",
            calibration_ids=("a", "b"),
            heldout_ids=("b", "c"),
            strata={},
        )


def test_split_round_trips_and_is_immutable(tmp_path: Path) -> None:
    split = _split()
    path = write_calibration_split(tmp_path / "split.json", split)
    assert load_calibration_split(path).identity == split.identity
    with pytest.raises(FileExistsError):
        write_calibration_split(path, split)


def test_tampered_membership_fails_its_identity_check(tmp_path: Path) -> None:
    import json

    path = write_calibration_split(tmp_path / "split.json", _split())
    payload = json.loads(path.read_text())
    payload["heldout_ids"].append(payload["calibration_ids"].pop())
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not match its identity"):
        load_calibration_split(path)


def _policy(split: CalibrationSplit, **overrides: object) -> FrozenPolicy:
    kwargs: dict[str, object] = {
        "name": "two-stage-interruption",
        "parameters": {"threshold": 0.9, "persistence_frames": 3},
        "objective": "maximise recall subject to non-break trigger rate <= 0.10",
        "split_identity": split.identity,
        "dataset": split.dataset,
        "dataset_revision": split.dataset_revision,
        "selected_on": "calibration",
        "search_space_size": 400,
        "calibration_metrics": {"recall": 0.71},
    }
    kwargs.update(overrides)
    return FrozenPolicy(**kwargs)  # type: ignore[arg-type]


def test_frozen_policy_round_trips(tmp_path: Path) -> None:
    split = _split()
    path = write_frozen_policy(tmp_path / "policy.json", _policy(split))
    loaded = load_frozen_policy(path)
    assert loaded.parameters["persistence_frames"] == 3
    assert loaded.to_dict()["heldout_labels_used_for_selection"] is False


def test_heldout_evaluation_accepts_a_matching_policy() -> None:
    split = _split()
    require_heldout_evaluation(_policy(split), split, split.heldout_ids)


def test_policy_from_another_split_is_refused() -> None:
    split, other = _split(seed=0), _split(seed=1)
    with pytest.raises(ValueError, match="different calibration split"):
        require_heldout_evaluation(_policy(other), split, split.heldout_ids)


def test_policy_selected_on_heldout_is_refused() -> None:
    split = _split()
    with pytest.raises(ValueError, match="not on calibration"):
        require_heldout_evaluation(_policy(split, selected_on="heldout"), split, split.heldout_ids)


def test_evaluating_calibration_ids_is_refused() -> None:
    split = _split()
    contaminated = [*split.heldout_ids, split.calibration_ids[0]]
    with pytest.raises(ValueError, match="contaminated"):
        require_heldout_evaluation(_policy(split), split, contaminated)


def test_persistence_commit_rule_matches_the_incumbent_at_one_frame() -> None:
    import numpy as np

    from turn_eval.reports.sid_calibration import commit_time_s

    probabilities = np.array([0.5, 0.99, 0.5, 0.99, 0.99, 0.99])
    # persistence 1 is exactly the incumbent "first frame at or above" rule.
    assert commit_time_s(
        probabilities, threshold=0.9, persistence_frames=1, frame_rate_hz=12.5
    ) == pytest.approx(2 / 12.5)
    # persistence 3 needs the run at indices 3, 4, 5 and commits at its end.
    assert commit_time_s(
        probabilities, threshold=0.9, persistence_frames=3, frame_rate_hz=12.5
    ) == pytest.approx(6 / 12.5)
    # A run that never reaches the required length never commits.
    assert (
        commit_time_s(probabilities, threshold=0.9, persistence_frames=4, frame_rate_hz=12.5)
        is None
    )


def test_a_broken_streak_restarts_the_confirmation() -> None:
    import numpy as np

    from turn_eval.reports.sid_calibration import commit_time_s

    probabilities = np.array([0.99, 0.99, 0.1, 0.99, 0.99])
    assert (
        commit_time_s(probabilities, threshold=0.9, persistence_frames=3, frame_rate_hz=12.5)
        is None
    )


def test_persistence_below_one_frame_is_rejected() -> None:
    import numpy as np

    from turn_eval.reports.sid_calibration import commit_time_s

    with pytest.raises(ValueError, match="at least one frame"):
        commit_time_s(np.array([1.0]), threshold=0.5, persistence_frames=0, frame_rate_hz=12.5)
