from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from turn_eval.core.types import AudioData, ClipRequest, DecisionRequest, TurnState
from turn_eval.models.smart_turn import (
    SMART_TURN_SAMPLE_RATE,
    SmartTurnFastTurnAdapter,
    SmartTurnV32Adapter,
)


@dataclass
class _FeatureExtractor:
    audio_inputs: list[np.ndarray] = field(default_factory=list)

    def __call__(self, audio: np.ndarray, **_: object) -> dict[str, np.ndarray]:
        self.audio_inputs.append(audio)
        return {"input_features": np.zeros((1, 80, 800), dtype=np.float32)}


@dataclass
class _Session:
    inputs: list[np.ndarray] = field(default_factory=list)

    def run(self, _: None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.inputs.append(inputs["input_features"])
        return [np.array([[0.75]], dtype=np.float32)]


def test_smart_turn_uses_upstream_left_padding_and_probability() -> None:
    session = _Session()
    extractor = _FeatureExtractor()
    adapter = SmartTurnV32Adapter(
        revision="fixture",
        session=session,
        feature_extractor=extractor,
        artifact_checksum="sha256:fixture",
        artifact_size_bytes=123,
    )
    audio = AudioData(np.ones(3_200, dtype=np.float32), SMART_TURN_SAMPLE_RATE)
    request = DecisionRequest(
        example_id="example",
        decision_point_id="point",
        target_at_s=0.2,
        observed_until_s=0.2,
        silence_duration_s=0.2,
        user_audio=audio,
        transcript=None,
    )

    prediction = adapter.predict(request)

    assert prediction.scores["eot"] == 0.75
    assert extractor.audio_inputs[0].shape == (128_000,)
    assert np.all(extractor.audio_inputs[0][:-3_200] == 0)
    assert np.all(extractor.audio_inputs[0][-3_200:] == 1)
    assert session.inputs[0].shape == (1, 80, 800)
    assert adapter.descriptor.natural_score_point_ms == 200.0
    assert adapter.descriptor.artifact_size_bytes == 123


def test_smart_turn_fastturn_uses_predeclared_threshold() -> None:
    pointwise = SmartTurnV32Adapter(
        revision="fixture",
        session=_Session(),
        feature_extractor=_FeatureExtractor(),
        artifact_checksum="sha256:fixture",
        artifact_size_bytes=123,
    )
    adapter = SmartTurnFastTurnAdapter(
        (TurnState.COMPLETE, TurnState.INCOMPLETE),
        pointwise=pointwise,
    )
    request = ClipRequest(
        "example",
        "zh",
        AudioData(np.ones(3_200, dtype=np.float32), SMART_TURN_SAMPLE_RATE),
    )

    prediction = adapter.predict_clip(request)

    assert prediction.predicted_label is TurnState.COMPLETE
    assert prediction.scores == {"complete": 0.75, "incomplete": 0.25}
    assert prediction.model_output["threshold"] == 0.5
    assert prediction.model_output["threshold_source"] == "predeclared-upstream-default"
    assert adapter.descriptor.output_signals == frozenset({"turn_state"})
