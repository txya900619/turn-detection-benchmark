from __future__ import annotations

import numpy as np
import pytest

from turn_eval.core.types import AudioData, ClipRequest, TurnState
from turn_eval.models.easy_turn import (
    EASY_TURN_MAX_AUDIO_SAMPLES,
    EasyTurnFastTurnAdapter,
    parse_easy_turn_output,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("今天天氣很好。<complete>", TurnState.COMPLETE),
        ("我想要……<INCOMPLETE>。", TurnState.INCOMPLETE),
        ("嗯嗯 <backchannel>\n", TurnState.BACKCHANNEL),
        ("等一下<wait>！", TurnState.WAIT),
        ("嗯<backchannel><|endoftext|>", TurnState.BACKCHANNEL),
    ],
)
def test_easy_turn_parses_one_terminal_native_tag(text: str, expected: TurnState) -> None:
    assert parse_easy_turn_output(text) is expected


@pytest.mark.parametrize(
    "text",
    [
        "沒有標籤",
        "<complete> 後面還有文字",
        "內容<complete><wait>",
        "先提到<complete>，最後仍是<complete>",
    ],
)
def test_easy_turn_ambiguous_or_non_terminal_output_abstains(text: str) -> None:
    assert parse_easy_turn_output(text) is None


def test_easy_turn_adapter_preserves_raw_text_and_emits_hard_label() -> None:
    adapter = EasyTurnFastTurnAdapter(
        tuple(TurnState),
        infer=lambda _: "你好。<complete>",
    )
    request = ClipRequest(
        "example",
        "zh",
        AudioData(np.zeros(1600, dtype=np.float32), 16_000),
    )

    prediction = adapter.predict_clip(request)

    assert prediction.predicted_label is TurnState.COMPLETE
    assert prediction.scores == {
        "complete": 1.0,
        "incomplete": 0.0,
        "backchannel": 0.0,
        "wait": 0.0,
    }
    assert prediction.model_output["raw_text"] == "你好。<complete>"
    assert prediction.model_output["score_semantics"] == ("hard-decoded-one-hot-not-probabilities")


def test_easy_turn_out_of_suite_native_label_is_closed_set_abstention() -> None:
    adapter = EasyTurnFastTurnAdapter(
        (TurnState.COMPLETE, TurnState.INCOMPLETE),
        infer=lambda _: "嗯<backchannel>",
    )
    request = ClipRequest(
        "example",
        "zh",
        AudioData(np.zeros(1600, dtype=np.float32), 16_000),
    )

    prediction = adapter.predict_clip(request)

    assert prediction.predicted_label is None
    assert prediction.scores == {"complete": 0.0, "incomplete": 0.0}
    assert prediction.model_output["native_label"] == "backchannel"


def test_easy_turn_batched_backend_preserves_request_order() -> None:
    seen = []

    def infer_batch(samples: tuple[np.ndarray, ...]) -> tuple[str, ...]:
        seen.append(tuple(len(sample) for sample in samples))
        return ("甲<complete>", "乙<wait><|endoftext|>")

    adapter = EasyTurnFastTurnAdapter(tuple(TurnState), infer_batch=infer_batch)
    requests = (
        ClipRequest("a", "zh", AudioData(np.zeros(800, dtype=np.float32), 16_000)),
        ClipRequest("b", "zh", AudioData(np.zeros(1600, dtype=np.float32), 16_000)),
    )

    predictions = adapter.predict_clips(requests)

    assert seen == [(800, 1600)]
    assert [prediction.example_id for prediction in predictions] == ["a", "b"]
    assert [prediction.predicted_label for prediction in predictions] == [
        TurnState.COMPLETE,
        TurnState.WAIT,
    ]


def test_easy_turn_abstains_on_audio_beyond_upstream_limit() -> None:
    called = False

    def infer(_: np.ndarray) -> str:
        nonlocal called
        called = True
        return "不應執行<complete>"

    adapter = EasyTurnFastTurnAdapter(tuple(TurnState), infer=infer)
    request = ClipRequest(
        "too-long",
        "zh",
        AudioData(
            np.zeros(EASY_TURN_MAX_AUDIO_SAMPLES + 1, dtype=np.float32),
            16_000,
        ),
    )

    prediction = adapter.predict_clip(request)

    assert called is False
    assert prediction.predicted_label is None
    assert all(score == 0.0 for score in prediction.scores.values())
    assert prediction.model_output["abstention_reason"] == "input-too-long"
    assert prediction.model_output["maximum_input_duration_s"] == 30.0


def test_easy_turn_rejects_empty_output_vocabulary() -> None:
    with pytest.raises(ValueError, match="non-empty and unique"):
        EasyTurnFastTurnAdapter((), infer=lambda _: "<complete>")
