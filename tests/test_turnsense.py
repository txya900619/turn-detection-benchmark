from __future__ import annotations

import numpy as np
import pytest

from turn_eval.core.types import AudioData, ClipRequest, TurnState
from turn_eval.models.turnsense import TurnSenseFastTurnAdapter, map_turnsense_probabilities


def test_turnsense_native_complete_and_incomplete_mapping() -> None:
    predicted, scores, native = map_turnsense_probabilities(np.array([0.8, 0.1, 0.1]))

    assert predicted is TurnState.COMPLETE
    assert native == "complete"
    assert scores == {"complete": pytest.approx(0.8), "incomplete": pytest.approx(0.1)}


def test_turnsense_invalid_is_closed_set_abstention() -> None:
    predicted, scores, native = map_turnsense_probabilities(np.array([0.05, 0.15, 0.8]))

    assert predicted is None
    assert native == "invalid"
    assert scores == {"complete": pytest.approx(0.05), "incomplete": pytest.approx(0.15)}


def test_turnsense_logits_are_softmaxed_before_artifact_output() -> None:
    adapter = TurnSenseFastTurnAdapter(
        (TurnState.COMPLETE, TurnState.INCOMPLETE),
        infer=lambda _: np.array([-1.0, 2.0, 0.0]),
    )

    prediction = adapter.predict_clip(
        ClipRequest("example", "zh", AudioData(np.zeros(1600, dtype=np.float32), 16_000))
    )

    probabilities = prediction.model_output["native_probabilities"]
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert prediction.predicted_label is TurnState.INCOMPLETE


def test_turnsense_adapter_uses_last_eight_seconds_and_preserves_native_output() -> None:
    seen = []
    adapter = TurnSenseFastTurnAdapter(
        (TurnState.COMPLETE, TurnState.INCOMPLETE),
        infer=lambda samples: seen.append(samples) or np.array([0.1, 0.7, 0.2]),
    )
    audio = AudioData(np.arange(10 * 16_000, dtype=np.float32), 16_000)

    prediction = adapter.predict_clip(ClipRequest("example", "zh", audio))

    assert len(seen[0]) == 8 * 16_000
    assert seen[0][0] == 2 * 16_000
    assert prediction.predicted_label is TurnState.INCOMPLETE
    assert prediction.model_output["native_probabilities"]["invalid"] == pytest.approx(0.2)


def test_turnsense_adapter_resamples_to_native_rate() -> None:
    pytest.importorskip("soxr")
    adapter = TurnSenseFastTurnAdapter(
        (TurnState.COMPLETE, TurnState.INCOMPLETE),
        infer=lambda _: np.array([0.8, 0.1, 0.1]),
    )

    result = adapter.resample_audio(AudioData(np.zeros(4800, dtype=np.float32), 48_000))

    assert result.sample_rate == 16_000
    assert len(result.samples) == 1600


def test_turnsense_rejects_non_binary_fastturn_suite() -> None:
    with pytest.raises(ValueError, match="only FastTurn eot"):
        TurnSenseFastTurnAdapter(tuple(TurnState), infer=lambda _: np.ones(3) / 3)
