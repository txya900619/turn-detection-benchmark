from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from turn_eval.models.smart_turn_timeline import (
    SmartTurnTimelineConfig,
    score_smart_turn_channel,
)


@dataclass
class _Endpoint:
    segments: list[np.ndarray] = field(default_factory=list)

    def __call__(self, segment: np.ndarray) -> float:
        self.segments.append(segment.copy())
        return 0.75


def test_smart_turn_timeline_preserves_trigger_duplication_and_commit_grid() -> None:
    # One sample per chunk makes the reference state transitions easy to inspect:
    # silence, speech trigger, then two silent chunks to reach STOP.
    waveform = np.asarray([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vad_values = iter((0.0, 1.0, 0.0, 0.0, 0.0))
    endpoint = _Endpoint()
    config = SmartTurnTimelineConfig(
        sample_rate=10,
        chunk_samples=1,
        output_hz=10.0,
        pre_speech_ms=100.0,
        stop_ms=200.0,
        settle_ms=1_000.0,
        max_segment_ms=1_000.0,
    )

    scores = score_smart_turn_channel(
        waveform,
        vad_probability=lambda _: next(vad_values),
        endpoint_probability=endpoint,
        config=config,
    )

    assert scores.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.75, 0.75, 0.0])
    assert endpoint.segments[0].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_smart_turn_timeline_resets_completion_score_when_speech_resumes() -> None:
    waveform = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    vad_values = iter((1.0, 0.0, 0.0, 1.0))
    config = SmartTurnTimelineConfig(
        sample_rate=10,
        chunk_samples=1,
        output_hz=10.0,
        pre_speech_ms=100.0,
        stop_ms=200.0,
        settle_ms=1_000.0,
        max_segment_ms=1_000.0,
    )

    scores = score_smart_turn_channel(
        waveform,
        vad_probability=lambda _: next(vad_values),
        endpoint_probability=lambda _: 0.9,
        config=config,
    )

    assert scores.tolist() == pytest.approx([0.0, 0.0, 0.9, 0.0, 0.0])


def test_smart_turn_timeline_rejects_invalid_endpoint_probability() -> None:
    config = SmartTurnTimelineConfig(
        sample_rate=10,
        chunk_samples=1,
        output_hz=10.0,
        pre_speech_ms=100.0,
        stop_ms=100.0,
        settle_ms=100.0,
        max_segment_ms=100.0,
    )
    with pytest.raises(ValueError, match="probability"):
        score_smart_turn_channel(
            np.asarray([1.0, 0.0], dtype=np.float32),
            vad_probability=lambda chunk: float(chunk[0]),
            endpoint_probability=lambda _: 2.0,
            config=config,
        )
