from __future__ import annotations

import numpy as np
import pytest

from turn_eval.core.types import (
    AudioData,
    ClipRequest,
    DecisionRequest,
    ExecutionMode,
    TurnState,
)
from turn_eval.models.turnsense import (
    TurnSenseFastTurnAdapter,
    TurnSensePointwiseAdapter,
    map_turnsense_probabilities,
)


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


def _decision_request(samples: np.ndarray, *, sample_rate: int = 16_000) -> DecisionRequest:
    observed = len(samples) / sample_rate
    return DecisionRequest(
        example_id="clip",
        decision_point_id="dp-0",
        target_at_s=observed,
        observed_until_s=observed,
        silence_duration_s=0.0,
        user_audio=AudioData(samples, sample_rate),
        transcript=None,
    )


def test_pointwise_eot_is_raw_complete_probability() -> None:
    adapter = TurnSensePointwiseAdapter(infer=lambda _: np.array([0.7, 0.2, 0.1]))
    prediction = adapter.predict(_decision_request(np.zeros(16_000, dtype=np.float32)))

    assert prediction.scores["eot"] == pytest.approx(0.7)
    assert prediction.scores["turnsense_incomplete"] == pytest.approx(0.2)
    assert prediction.scores["turnsense_invalid"] == pytest.approx(0.1)


def test_invalid_mass_lowers_eot_rather_than_being_renormalised() -> None:
    """A low-confidence frame must become less likely to cut off, not equally likely."""
    adapter = TurnSensePointwiseAdapter(infer=lambda _: np.array([0.05, 0.05, 0.90]))
    prediction = adapter.predict(_decision_request(np.zeros(16_000, dtype=np.float32)))

    # Renormalising over complete/incomplete alone would give 0.5 here.
    assert prediction.scores["eot"] == pytest.approx(0.05)
    assert prediction.scores["turnsense_invalid"] == pytest.approx(0.90)


def test_pointwise_adapter_uses_the_upstream_eight_second_tail() -> None:
    seen: list[int] = []

    def infer(samples: np.ndarray) -> np.ndarray:
        seen.append(len(samples))
        return np.array([0.5, 0.3, 0.2])

    adapter = TurnSensePointwiseAdapter(infer=infer)
    adapter.predict(_decision_request(np.zeros(16_000 * 12, dtype=np.float32)))

    assert seen == [16_000 * 8]


def test_pointwise_adapter_declares_a_streaming_eot_signal() -> None:
    adapter = TurnSensePointwiseAdapter(infer=lambda _: np.array([0.5, 0.3, 0.2]))
    descriptor = adapter.descriptor

    assert descriptor.name == "turnsense-int8-cpu"
    assert descriptor.output_signals == frozenset({"eot"})
    assert ExecutionMode.POINTWISE in descriptor.execution_modes
    # The clip adapter's identity must stay distinct so runs cannot collide.
    assert (
        descriptor.name
        != TurnSenseFastTurnAdapter(
            (TurnState.COMPLETE, TurnState.INCOMPLETE), infer=lambda _: np.array([0.5, 0.3, 0.2])
        ).descriptor.name
    )


def test_pointwise_adapter_rejects_the_wrong_sample_rate() -> None:
    adapter = TurnSensePointwiseAdapter(infer=lambda _: np.array([0.5, 0.3, 0.2]))
    with pytest.raises(ValueError, match="16 kHz"):
        adapter.predict(_decision_request(np.zeros(8_000, dtype=np.float32), sample_rate=8_000))
