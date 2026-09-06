from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from turn_eval.benchmarks.turnbench import (
    TurnBenchProbabilityFile,
    commit_probability_events,
    sweep_with_official_turnbench,
    turnbench_candidate_thresholds,
)


def payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "task": "eot",
        "frame_rate_hz": 25.0,
        "probs": [
            {
                "conversation_id": "20",
                "speaker_1": {"prob": [0.0, 0.8, 0.9]},
                "speaker_2": {"prob": [0.0, 0.0, 0.0]},
            }
        ],
    }


def test_probability_schema_frame_counts_and_round_trip(tmp_path) -> None:
    probabilities = TurnBenchProbabilityFile.from_dict(payload())
    probabilities.validate({"20": 0.14})
    path = tmp_path / "probs-eot.json"
    probabilities.write(path)

    assert TurnBenchProbabilityFile.load(path) == probabilities
    with pytest.raises(ValueError, match="frames, expected"):
        probabilities.validate({"20": 1.0})


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_probability_schema_rejects_invalid_values(bad: float) -> None:
    value = payload()
    value["probs"][0]["speaker_1"]["prob"][1] = bad
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        TurnBenchProbabilityFile.from_dict(value)


def test_official_commit_rule_uses_strict_edge_frame_end_and_refractory() -> None:
    committed = commit_probability_events(
        [0.5, 0.6, 0.4, 0.7, 0.4, 0.8],
        frame_rate_hz=2.0,
        theta=0.5,
        refractory_s=2.0,
    )

    assert committed == (1.0, 3.0)


def test_probability_file_commits_to_official_submission() -> None:
    probabilities = TurnBenchProbabilityFile.from_dict(payload())
    submission = probabilities.commit(theta=0.5)

    assert submission.predictions[0].speaker_1.eot == (0.08,)
    assert submission.predictions[0].speaker_1.interruption == ()


def test_candidate_thresholds_are_quantiles_plus_uniform_grid() -> None:
    probabilities = TurnBenchProbabilityFile.from_dict(payload())
    candidates = turnbench_candidate_thresholds(probabilities, count=16)

    assert candidates == tuple(sorted(set(candidates)))
    assert 0.01 in candidates
    assert 0.8 in candidates
    assert 0.9 in candidates


def test_official_sweep_bridge_is_invoked_with_independent_task_files(
    tmp_path, monkeypatch
) -> None:
    eot_path = tmp_path / "eot.json"
    TurnBenchProbabilityFile.from_dict(payload()).write(eot_path)
    int_payload = payload()
    int_payload["task"] = "int"
    int_path = tmp_path / "int.json"
    TurnBenchProbabilityFile.from_dict(int_payload).write(int_path)
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout=json.dumps({"scores": {}}), stderr="")

    monkeypatch.setattr("turn_eval.benchmarks.turnbench.subprocess.run", fake_run)
    result = sweep_with_official_turnbench(
        eot_path,
        int_path,
        tmp_path / "scored",
        scorer_project=tmp_path / "official",
    )

    assert result == {"scores": {}}
    assert captured["command"][0:4] == ["uv", "run", "--frozen", "--project"]
    assert "--eot-probs" in captured["command"]
    assert "--int-probs" in captured["command"]
    assert captured["kwargs"]["check"] is False
