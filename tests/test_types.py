from __future__ import annotations

import numpy as np
import pytest

from turn_eval.core.types import (
    AudioData,
    ClipClassificationExample,
    DecisionLabel,
    DecisionPoint,
    DecisionPointExample,
    StatePredictionRecord,
    TurnPrediction,
    TurnState,
)


def test_audio_is_mono_finite_and_immutable() -> None:
    source = np.array([0.0, 0.5], dtype=np.float32)
    audio = AudioData(source, 1_000)
    source[0] = 1.0
    assert audio.samples[0] == 0.0
    with pytest.raises(ValueError):
        audio.samples[0] = 1.0
    with pytest.raises(ValueError, match="mono"):
        AudioData(np.zeros((2, 2), dtype=np.float32), 1_000)


def test_audio_prefix_allows_subsample_timestamp_rounding() -> None:
    audio = AudioData(np.zeros(16_000, dtype=np.float32), 16_000)

    prefix = audio.prefix(1.0000005)

    assert len(prefix.samples) == 16_000


def test_decision_points_must_be_chronological_and_inside_audio() -> None:
    audio = AudioData(np.zeros(1_000, dtype=np.float32), 1_000)
    later = DecisionPoint("later", "span", 0.8, 0.5, 0.9, DecisionLabel.EOT)
    earlier = DecisionPoint("earlier", "span", 0.6, 0.5, 0.9, DecisionLabel.EOT)
    with pytest.raises(ValueError, match="chronological"):
        DecisionPointExample("id", "conversation", "en", audio, (later, earlier))
    with pytest.raises(ValueError, match="inside"):
        DecisionPoint("bad", "span", 0.4, 0.5, 0.9, DecisionLabel.HOLD)


def test_prediction_enforces_causal_timing_and_probabilities() -> None:
    with pytest.raises(ValueError, match="before its inputs"):
        TurnPrediction("id", "point", 0.5, 0.6, {"eot": 0.5})
    with pytest.raises(ValueError, match="invalid probability"):
        TurnPrediction("id", "point", 0.6, 0.6, {"eot": 1.1})


def test_clip_example_requires_aligned_dual_channel_audio() -> None:
    user = AudioData(np.zeros(1_000, dtype=np.float32), 1_000)
    agent = AudioData(np.zeros(2_000, dtype=np.float32), 1_000)

    with pytest.raises(ValueError, match="equal duration"):
        ClipClassificationExample("clip", "zh", user, TurnState.COMPLETE, agent_audio=agent)


def test_state_prediction_validates_latency_and_scores() -> None:
    with pytest.raises(ValueError, match="latency"):
        StatePredictionRecord(
            "clip",
            "group",
            "zh",
            TurnState.COMPLETE,
            TurnState.COMPLETE,
            {"complete": 1.0},
            -1.0,
        )
