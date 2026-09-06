from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from turn_eval.core.types import AudioData, ClipRequest, TurnState
from turn_eval.models.x2_turn import X2TurnFastTurnAdapter, last_non_idle_frame, map_x2_frame


def _frame(index: int, label: str) -> SimpleNamespace:
    labels = ("idle", "noidle", "speaking", "turn_end", "backchannel", "uncertain")
    probabilities = {name: 0.0 for name in labels}
    probabilities[label] = 1.0
    return SimpleNamespace(
        index=index,
        start_ms=index * 80,
        end_ms=(index + 1) * 80,
        label=label,
        confidence=1.0,
        probabilities=probabilities,
    )


def test_x2_reduction_uses_last_non_idle_and_exact_mapping() -> None:
    frames = (_frame(0, "speaking"), _frame(1, "turn_end"), _frame(2, "idle"))

    selected = last_non_idle_frame(frames)
    predicted, scores = map_x2_frame(selected, (TurnState.COMPLETE, TurnState.INCOMPLETE))

    assert selected.index == 1
    assert predicted is TurnState.COMPLETE
    assert scores == {"complete": 1.0, "incomplete": 0.0}


@pytest.mark.parametrize("raw_label", ["idle", "noidle", "uncertain"])
def test_x2_unsupported_raw_states_abstain(raw_label: str) -> None:
    frame = None if raw_label == "idle" else _frame(0, raw_label)

    predicted, _ = map_x2_frame(
        frame, (TurnState.COMPLETE, TurnState.INCOMPLETE, TurnState.BACKCHANNEL)
    )

    assert predicted is None


def test_x2_adapter_preserves_raw_frames_and_generated_asr() -> None:
    result = SimpleNamespace(
        transcript="你好",
        frame_ms=80,
        generated_token_ids=[1, 2],
        turn_frames=[_frame(0, "speaking"), _frame(1, "idle")],
    )
    adapter = X2TurnFastTurnAdapter(
        (TurnState.COMPLETE, TurnState.INCOMPLETE),
        infer=lambda _: result,
    )
    request = ClipRequest("example", "zh", AudioData(np.zeros(1600), 16_000))

    prediction = adapter.predict_clip(request)

    assert prediction.predicted_label is TurnState.INCOMPLETE
    assert prediction.model_output["transcript"] == "你好"
    assert prediction.model_output["selected_frame_index"] == 0
    assert len(prediction.model_output["turn_frames"]) == 2


def test_x2_rejects_fastturn_full_wait_vocabulary() -> None:
    with pytest.raises(ValueError, match="Wait has no native class"):
        X2TurnFastTurnAdapter(tuple(TurnState), infer=lambda _: None)
