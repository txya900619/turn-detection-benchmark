from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from turn_eval.core.types import AudioData, DecisionRequest, SupportStatus
from turn_eval.models.dualturn import (
    DUALTURN_FRAME_SAMPLES,
    DUALTURN_SAMPLE_RATE,
    DualTurnSingleChannelEOTAdapter,
)


@dataclass
class _Stream:
    chunks: list[np.ndarray] = field(default_factory=list)

    def push(self, chunk: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        self.chunks.append(chunk.copy())
        probability = len(self.chunks) / 10
        return probability, np.zeros(2, dtype=np.float32), np.zeros((2, 4), dtype=np.float32)


def _request(example_id: str, frames: int, point: str) -> DecisionRequest:
    samples = np.ones(frames * DUALTURN_FRAME_SAMPLES, dtype=np.float32)
    duration_s = len(samples) / DUALTURN_SAMPLE_RATE
    return DecisionRequest(
        example_id=example_id,
        decision_point_id=point,
        target_at_s=duration_s,
        observed_until_s=duration_s,
        silence_duration_s=0.0,
        user_audio=AudioData(samples, DUALTURN_SAMPLE_RATE),
        transcript=None,
    )


def test_dualturn_eot_streams_only_new_frames_with_silent_agent() -> None:
    streams: list[_Stream] = []

    def factory() -> _Stream:
        stream = _Stream()
        streams.append(stream)
        return stream

    adapter = DualTurnSingleChannelEOTAdapter(factory)

    first = adapter.predict(_request("example", 2, "p1"))
    second = adapter.predict(_request("example", 3, "p2"))

    assert first.scores == {"eot": 0.2}
    assert second.scores == {"eot": 0.3}
    assert len(streams) == 1
    assert len(streams[0].chunks) == 3
    assert all(chunk.shape == (2, DUALTURN_FRAME_SAMPLES) for chunk in streams[0].chunks)
    assert all(np.all(chunk[0] == 1) for chunk in streams[0].chunks)
    assert all(np.all(chunk[1] == 0) for chunk in streams[0].chunks)


def test_dualturn_eot_resets_between_examples_and_marks_zh_zero_shot() -> None:
    streams: list[_Stream] = []

    def factory() -> _Stream:
        stream = _Stream()
        streams.append(stream)
        return stream

    adapter = DualTurnSingleChannelEOTAdapter(factory)
    adapter.predict(_request("first", 1, "p1"))
    prediction = adapter.predict(_request("second", 1, "p1"))

    assert prediction.scores == {"eot": 0.1}
    assert len(streams) == 2
    assert adapter.descriptor.support_for_language("en") is SupportStatus.SUPPORTED
    assert adapter.descriptor.support_for_language("zh") is SupportStatus.UNVERIFIED_ZERO_SHOT
