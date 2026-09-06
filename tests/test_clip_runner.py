from __future__ import annotations

import numpy as np
import pytest

from turn_eval.core.types import (
    AudioData,
    ChannelMode,
    ClipClassificationExample,
    ClipRequest,
    EvaluationMode,
    ExecutionMode,
    ModelDescriptor,
    ServiceMode,
    SignalProvenance,
    StatePrediction,
    SupportStatus,
    TranscriptMode,
    TurnState,
)
from turn_eval.runner.clip_classification import run_clip_classification


class FixedClipAdapter:
    def __init__(self, predictions: dict[str, TurnState]) -> None:
        self.predictions = predictions
        self.requests: list[ClipRequest] = []
        self.descriptor = ModelDescriptor(
            name="fixed-clip",
            revision="fixture-v1",
            native_sample_rate=16_000,
            execution_modes=frozenset({ExecutionMode.OFFLINE}),
            output_signals=frozenset({"turn_state"}),
            supported_languages=frozenset({"zh"}),
        )

    def predict_clip(self, request: ClipRequest) -> StatePrediction:
        self.requests.append(request)
        label = self.predictions[request.example_id]
        return StatePrediction(request.example_id, label, {label.value: 1.0})


def _mode(transcript_mode: TranscriptMode = TranscriptMode.NONE) -> EvaluationMode:
    return EvaluationMode(
        channel_mode=ChannelMode.NATIVE_SINGLE,
        execution_mode=ExecutionMode.OFFLINE,
        signal_provenance=SignalProvenance.NATIVE,
        transcript_mode=transcript_mode,
        service_mode=ServiceMode.LOCAL,
        language_support=SupportStatus.SUPPORTED,
    )


def _examples(*, transcript: str | None = None) -> tuple[ClipClassificationExample, ...]:
    audio = AudioData(np.zeros(4_800, dtype=np.float32), 48_000)
    return (
        ClipClassificationExample(
            "a",
            "zh",
            audio,
            TurnState.COMPLETE,
            transcript=transcript,
            metadata={"source_kind": "real"},
        ),
        ClipClassificationExample(
            "b",
            "zh",
            audio,
            TurnState.INCOMPLETE,
            transcript=transcript,
            metadata={"source_kind": "real"},
        ),
    )


def test_clip_runner_hides_gold_and_tracks_runtime() -> None:
    model = FixedClipAdapter({"a": TurnState.COMPLETE, "b": TurnState.COMPLETE})

    run = run_clip_classification(_examples(), model, _mode(), initialization_time_ms=12.0)

    assert [row.label for row in run.predictions] == [
        TurnState.COMPLETE,
        TurnState.INCOMPLETE,
    ]
    assert [row.predicted_label for row in run.predictions] == [
        TurnState.COMPLETE,
        TurnState.COMPLETE,
    ]
    assert all(request.user_audio.sample_rate == 16_000 for request in model.requests)
    assert all(request.transcript is None for request in model.requests)
    assert not hasattr(model.requests[0], "label")
    assert not hasattr(model.requests[0], "metadata")
    assert run.initialization_time_ms == 12.0
    assert run.audio_s == pytest.approx(0.2)
    assert run.rtf >= 0


def test_clip_runner_rejects_reference_text_in_audio_only_mode() -> None:
    model = FixedClipAdapter({"a": TurnState.COMPLETE, "b": TurnState.INCOMPLETE})

    with pytest.raises(ValueError, match="cannot expose reference transcripts"):
        run_clip_classification(_examples(transcript="乾淨文字"), model, _mode())


def test_clip_runner_rejects_cross_example_prediction() -> None:
    class WrongIdAdapter(FixedClipAdapter):
        def predict_clip(self, request: ClipRequest) -> StatePrediction:
            return StatePrediction("other", TurnState.COMPLETE, {"complete": 1.0})

    with pytest.raises(ValueError, match="another request"):
        run_clip_classification(
            _examples(),
            WrongIdAdapter({"a": TurnState.COMPLETE, "b": TurnState.INCOMPLETE}),
            _mode(),
        )


def test_clip_runner_uses_adapter_resampler_when_available() -> None:
    model = FixedClipAdapter({"a": TurnState.COMPLETE, "b": TurnState.INCOMPLETE})
    calls = []

    def resample_audio(audio: AudioData) -> AudioData:
        calls.append(audio.sample_rate)
        return AudioData(np.zeros(1600, dtype=np.float32), 16_000)

    model.resample_audio = resample_audio

    run_clip_classification(_examples(), model, _mode())

    assert calls == [48_000, 48_000]


def test_clip_runner_streams_generator_with_declared_total() -> None:
    model = FixedClipAdapter({"a": TurnState.COMPLETE, "b": TurnState.INCOMPLETE})
    progress = []

    run = run_clip_classification(
        (example for example in _examples()),
        model,
        _mode(),
        total_examples=2,
        progress=lambda done, total: progress.append((done, total)),
    )

    assert len(run.predictions) == 2
    assert progress == [(1, 2), (2, 2)]


def test_clip_runner_rejects_generator_count_mismatch() -> None:
    model = FixedClipAdapter({"a": TurnState.COMPLETE, "b": TurnState.INCOMPLETE})

    with pytest.raises(ValueError, match="produced 2 examples; expected 3"):
        run_clip_classification(
            (example for example in _examples()),
            model,
            _mode(),
            total_examples=3,
        )


def test_clip_runner_uses_adapter_batch_and_counts_wall_time_once() -> None:
    class BatchedAdapter(FixedClipAdapter):
        def predict_clips(self, requests: tuple[ClipRequest, ...]):
            return tuple(self.predict_clip(request) for request in requests)

    model = BatchedAdapter({"a": TurnState.COMPLETE, "b": TurnState.INCOMPLETE})
    progress = []

    run = run_clip_classification(
        _examples(),
        model,
        _mode(),
        batch_size=2,
        progress=lambda done, total: progress.append((done, total)),
    )

    assert [row.example_id for row in run.predictions] == ["a", "b"]
    assert progress == [(2, 2)]
    assert run.inference_wall_s <= sum(row.inference_latency_ms / 1000 for row in run.predictions)


def test_clip_runner_rejects_batch_for_single_item_adapter() -> None:
    model = FixedClipAdapter({"a": TurnState.COMPLETE, "b": TurnState.INCOMPLETE})

    with pytest.raises(ValueError, match="does not implement batched"):
        run_clip_classification(_examples(), model, _mode(), batch_size=2)
