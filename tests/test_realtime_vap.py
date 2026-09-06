from __future__ import annotations

import queue

import numpy as np
import pytest

from turn_eval.core.types import AudioData, DecisionRequest
from turn_eval.models.realtime_vap import RealtimeVAPSingleChannelEOTAdapter


class FakeVAP:
    def __init__(self) -> None:
        self.result_dict_queue: queue.Queue[dict[str, list[float]]] = queue.Queue()
        self.frames: list[tuple[np.ndarray, np.ndarray]] = []
        self.reset_count = 0

    def reset_runtime_state(self) -> None:
        self.reset_count += 1

    def process(self, user: np.ndarray, agent: np.ndarray) -> None:
        self.frames.append((user.copy(), agent.copy()))
        self.result_dict_queue.put({"p_now": [0.75, 0.25]})


def _request(example_id: str, frames: int) -> DecisionRequest:
    return DecisionRequest(
        example_id=example_id,
        decision_point_id=f"{example_id}-{frames}",
        target_at_s=frames / 10,
        observed_until_s=frames / 10,
        silence_duration_s=0.1,
        user_audio=AudioData(np.ones(frames * 1_600, dtype=np.float32), 16_000),
        transcript=None,
    )


def test_realtime_vap_incrementally_processes_prefix_with_silent_channel() -> None:
    runtime = FakeVAP()
    adapter = RealtimeVAPSingleChannelEOTAdapter(runtime)

    first = adapter.predict(_request("a", 2))
    second = adapter.predict(_request("a", 3))

    assert first.scores == {"eot": 0.25}
    assert second.scores == {"eot": 0.25}
    assert len(runtime.frames) == 3
    assert all(np.count_nonzero(agent) == 0 for _, agent in runtime.frames)
    assert runtime.reset_count == 1


def test_realtime_vap_resets_between_examples() -> None:
    runtime = FakeVAP()
    adapter = RealtimeVAPSingleChannelEOTAdapter(runtime)

    adapter.predict(_request("a", 1))
    adapter.predict(_request("b", 1))

    assert runtime.reset_count == 2
    assert len(runtime.frames) == 2


def test_realtime_vap_reuses_latest_causal_frame_for_subframe_decision() -> None:
    runtime = FakeVAP()
    adapter = RealtimeVAPSingleChannelEOTAdapter(runtime)
    adapter.predict(_request("a", 1))
    request = DecisionRequest(
        example_id="a",
        decision_point_id="unaligned",
        target_at_s=0.15,
        observed_until_s=0.15,
        silence_duration_s=0.05,
        user_audio=AudioData(np.ones(2_400, dtype=np.float32), 16_000),
        transcript=None,
    )

    prediction = adapter.predict(request)

    assert prediction.scores == {"eot": 0.25}
    assert len(runtime.frames) == 1


def test_realtime_vap_rejects_decision_before_first_frame() -> None:
    adapter = RealtimeVAPSingleChannelEOTAdapter(FakeVAP())
    request = DecisionRequest(
        example_id="a",
        decision_point_id="too-early",
        target_at_s=0.05,
        observed_until_s=0.05,
        silence_duration_s=0.05,
        user_audio=AudioData(np.ones(800, dtype=np.float32), 16_000),
        transcript=None,
    )

    with pytest.raises(ValueError, match="at least 100 ms"):
        adapter.predict(request)
