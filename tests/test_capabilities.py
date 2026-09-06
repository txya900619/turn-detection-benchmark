from __future__ import annotations

import pytest

from turn_eval.benchmarks.synthetic import synthetic_decision_point_examples
from turn_eval.core.types import (
    ChannelMode,
    EvaluationMode,
    ExecutionMode,
    ModelDescriptor,
    ServiceMode,
    SignalProvenance,
    SupportStatus,
    TranscriptMode,
)
from turn_eval.models.fixed_timeout import FixedTimeoutAdapter
from turn_eval.runner.decision_point import run_decision_points


def test_language_support_mismatch_is_rejected() -> None:
    mode = EvaluationMode(
        ChannelMode.NATIVE_SINGLE,
        ExecutionMode.POINTWISE,
        SignalProvenance.NATIVE,
        TranscriptMode.NONE,
        ServiceMode.LOCAL,
        SupportStatus.UNVERIFIED_ZERO_SHOT,
    )
    with pytest.raises(ValueError, match="language support mismatch"):
        run_decision_points(synthetic_decision_point_examples(), FixedTimeoutAdapter(), mode)


def test_audio_only_model_rejects_transcript_mode() -> None:
    mode = EvaluationMode(
        ChannelMode.NATIVE_SINGLE,
        ExecutionMode.POINTWISE,
        SignalProvenance.NATIVE,
        TranscriptMode.ORACLE_CAUSAL,
        ServiceMode.LOCAL,
        SupportStatus.SUPPORTED,
    )
    with pytest.raises(ValueError, match="audio-only"):
        run_decision_points(synthetic_decision_point_examples(), FixedTimeoutAdapter(), mode)


def test_unknown_language_is_unsupported_unless_zero_shot_is_declared() -> None:
    descriptor = ModelDescriptor(
        name="english-only",
        revision="v1",
        native_sample_rate=16_000,
        execution_modes=frozenset({ExecutionMode.POINTWISE}),
        output_signals=frozenset({"eot"}),
        supported_languages=frozenset({"en"}),
    )
    assert descriptor.support_for_language("zh") is SupportStatus.UNSUPPORTED
