from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import monotonic_ns

import numpy as np
import pytest

from turn_eval.core.types import (
    AudioData,
    ChannelMode,
    DecisionLabel,
    DecisionPoint,
    DecisionPointExample,
    DecisionRequest,
    EvaluationMode,
    ExecutionMode,
    ModelDescriptor,
    ServiceMode,
    SignalProvenance,
    SupportStatus,
    TimedWord,
    TranscriptMode,
    TurnPrediction,
)
from turn_eval.runner.decision_point import run_decision_points


@dataclass
class RecordingAdapter:
    requests: list[DecisionRequest] = field(default_factory=list)
    descriptor: ModelDescriptor = field(
        default_factory=lambda: ModelDescriptor(
            name="recording-fake",
            revision="v1",
            native_sample_rate=1_000,
            execution_modes=frozenset({ExecutionMode.POINTWISE}),
            output_signals=frozenset({"eot"}),
            supported_languages=frozenset({"en"}),
            accepted_inputs=frozenset({"user_audio", "transcript"}),
            algorithmic_lookahead_ms=200,
        )
    )

    def predict(self, request: DecisionRequest) -> TurnPrediction:
        self.requests.append(request)
        return TurnPrediction(
            request.example_id,
            request.decision_point_id,
            request.observed_until_s,
            request.observed_until_s,
            {"eot": 0.25},
            emitted_at_ns=monotonic_ns(),
        )


def _mode_for_recording(
    transcript_mode: TranscriptMode = TranscriptMode.NONE,
) -> EvaluationMode:
    return EvaluationMode(
        ChannelMode.NATIVE_SINGLE,
        ExecutionMode.POINTWISE,
        SignalProvenance.NATIVE,
        transcript_mode,
        ServiceMode.LOCAL,
        SupportStatus.SUPPORTED,
    )


def test_runner_withholds_future_audio_labels_and_transcript() -> None:
    audio = AudioData(np.ones(1_000, dtype=np.float32), 1_000)
    point = DecisionPoint("p", "s", 0.5, 0.4, 0.9, DecisionLabel.HOLD)
    example = DecisionPointExample(
        "e",
        "c",
        "en",
        audio,
        (point,),
        words=(TimedWord("visible", 0.1, 0.4), TimedWord("future", 0.7, 0.8)),
    )
    adapter = RecordingAdapter()
    mode = _mode_for_recording(TranscriptMode.ORACLE_CAUSAL)

    result = run_decision_points((example,), adapter, mode, transcript_lag_s=0.1)

    request = adapter.requests[0]
    assert request.observed_until_s == 0.7
    assert request.silence_duration_s == pytest.approx(0.1)
    assert request.user_audio.duration_s == 0.7
    assert request.transcript == "visible"
    assert not hasattr(request, "label")
    assert result.predictions[0].available_at_s == 0.7
    assert result.predictions[0].available_at_s - point.timestamp_s == pytest.approx(0.2)


def test_runner_reports_protocol_level_progress() -> None:
    audio = AudioData(np.ones(1_000, dtype=np.float32), 1_000)
    points = (
        DecisionPoint("p1", "s", 0.5, 0.4, 0.9, DecisionLabel.HOLD),
        DecisionPoint("p2", "s", 0.6, 0.4, 0.9, DecisionLabel.HOLD),
    )
    example = DecisionPointExample("e", "c", "en", audio, points)
    adapter = RecordingAdapter()
    updates: list[tuple[int, int]] = []

    run_decision_points(
        (example,),
        adapter,
        _mode_for_recording(),
        progress=lambda *update: updates.append(update),
    )

    assert updates == [(1, 2), (2, 2)]


def test_runner_tolerates_one_sample_resampling_boundary() -> None:
    audio = AudioData(np.ones(106_274, dtype=np.float32), 16_000)
    duration_s = audio.duration_s
    point = DecisionPoint(
        "p",
        "s",
        duration_s,
        duration_s - 0.1,
        duration_s,
        DecisionLabel.EOT,
    )
    example = DecisionPointExample("e", "c", "en", audio, (point,))
    descriptor = replace(
        RecordingAdapter().descriptor,
        native_sample_rate=24_000,
        algorithmic_lookahead_ms=0,
    )
    adapter = RecordingAdapter(descriptor=descriptor)

    result = run_decision_points((example,), adapter, _mode_for_recording())

    assert len(result.predictions) == 1
    assert abs(len(adapter.requests[0].user_audio.samples) - round(duration_s * 24_000)) <= 1
