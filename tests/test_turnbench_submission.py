from __future__ import annotations

import json

import pytest

from turn_eval.benchmarks.turnbench import (
    TurnBenchCommittedEvent,
    TurnBenchSubmission,
    TurnBenchTask,
)


def valid_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "predictions": [
            {
                "conversation_id": "20",
                "speaker_1": {"eot": [1.0, 2.0], "interruption": [3.0]},
                "speaker_2": {"eot": [], "interruption": []},
            }
        ],
    }


def test_official_submission_round_trip(tmp_path) -> None:
    submission = TurnBenchSubmission.from_dict(valid_payload())
    path = tmp_path / "predictions.json"
    submission.write(path)

    assert TurnBenchSubmission.load(path) == submission
    assert json.loads(path.read_text()) == valid_payload()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update(schema_version=2),
        lambda payload: payload.update(model="hidden-extra"),
        lambda payload: payload["predictions"][0]["speaker_1"].update(eot=[2.0, 1.0]),
        lambda payload: payload["predictions"][0]["speaker_1"].update(eot=[1.0, 1.0]),
        lambda payload: payload["predictions"][0]["speaker_1"].update(eot=[float("nan")]),
    ],
)
def test_invalid_official_submissions_are_rejected(mutate) -> None:
    payload = valid_payload()
    mutate(payload)
    with pytest.raises(ValueError):
        TurnBenchSubmission.from_dict(payload)


def test_coverage_and_duration_validation() -> None:
    submission = TurnBenchSubmission.from_dict(valid_payload())
    submission.validate_coverage(["20"])
    submission.validate_durations({"20": 3.0})

    with pytest.raises(ValueError, match="missing"):
        submission.validate_coverage(["20", "21"])
    with pytest.raises(ValueError, match="past duration"):
        submission.validate_durations({"20": 2.5})


def test_committed_events_charge_lookahead_and_fill_empty_channels() -> None:
    submission = TurnBenchSubmission.from_committed_events(
        [
            TurnBenchCommittedEvent(
                conversation_id="20",
                speaker=2,
                task=TurnBenchTask.INTERRUPTION,
                observed_until_s=1.2,
                commit_time_s=1.5,
            )
        ],
        conversation_ids=["20", "21"],
    )

    assert submission.predictions[0].speaker_2.interruption == (1.5,)
    assert submission.predictions[0].speaker_1.eot == ()
    assert submission.predictions[1].speaker_2.interruption == ()


def test_committed_event_cannot_use_future_audio() -> None:
    with pytest.raises(ValueError, match="future audio"):
        TurnBenchCommittedEvent(
            conversation_id="20",
            speaker=1,
            task=TurnBenchTask.EOT,
            observed_until_s=1.1,
            commit_time_s=1.0,
        )
