from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite

import numpy as np
import numpy.typing as npt

FloatAudio = npt.NDArray[np.float32]


class EvaluationProtocol(StrEnum):
    TIMELINE = "timeline"
    DECISION_POINT = "decision_point"
    CLIP_CLASSIFICATION = "clip_classification"
    SINGLE_CHANNEL_INTERRUPTION = "single_channel_interruption_timeline"


class DecisionLabel(StrEnum):
    EOT = "eot"
    HOLD = "hold"


class TurnState(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    BACKCHANNEL = "backchannel"
    WAIT = "wait"


class ChannelMode(StrEnum):
    NATIVE_SINGLE = "native_single"
    NATIVE_DUAL = "native_dual"
    SINGLE_CHANNEL_FALLBACK = "single_channel_fallback"


class ExecutionMode(StrEnum):
    POINTWISE = "pointwise"
    STREAMING = "streaming"
    CAUSAL_OFFLINE = "causal_offline"
    OFFLINE = "offline"


class SignalProvenance(StrEnum):
    NATIVE = "native"
    DERIVED = "derived"


class TranscriptMode(StrEnum):
    NONE = "none"
    ORACLE_CAUSAL = "oracle_causal"
    ORACLE_REFERENCE = "oracle_reference"
    STREAMING_ASR = "streaming_asr"


class ServiceMode(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"


class SupportStatus(StrEnum):
    SUPPORTED = "supported"
    DERIVED = "derived"
    UNVERIFIED_ZERO_SHOT = "unverified_zero_shot"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class AudioData:
    samples: FloatAudio
    sample_rate: int

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("AudioData samples must be mono with shape [samples]")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not np.all(np.isfinite(samples)):
            raise ValueError("audio samples must be finite")
        samples = np.array(samples, dtype=np.float32, copy=True)
        samples.setflags(write=False)
        object.__setattr__(self, "samples", samples)

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate

    def prefix(self, observed_until_s: float) -> AudioData:
        requested_samples = round(observed_until_s * self.sample_rate)
        if observed_until_s < 0 or requested_samples - len(self.samples) > 1:
            raise ValueError("audio prefix timestamp is outside the waveform")
        end = min(len(self.samples), requested_samples)
        return AudioData(self.samples[:end], self.sample_rate)


@dataclass(frozen=True, slots=True)
class TimedWord:
    text: str
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("timed word text must not be empty")
        if self.start_s < 0 or self.end_s < self.start_s:
            raise ValueError("invalid timed word interval")


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if not self.role or not self.content:
            raise ValueError("chat message role and content must not be empty")


@dataclass(frozen=True, slots=True)
class DecisionPoint:
    id: str
    span_id: str
    timestamp_s: float
    silence_start_s: float
    silence_end_s: float
    label: DecisionLabel

    def __post_init__(self) -> None:
        if not self.id or not self.span_id:
            raise ValueError("decision point identifiers must not be empty")
        if self.silence_start_s < 0:
            raise ValueError("silence start must be non-negative")
        if self.silence_end_s <= self.silence_start_s:
            raise ValueError("silence end must be after silence start")
        if not self.silence_start_s <= self.timestamp_s <= self.silence_end_s:
            raise ValueError("decision timestamp must lie inside its silence span")

    @property
    def silence_duration_s(self) -> float:
        return self.timestamp_s - self.silence_start_s


@dataclass(frozen=True, slots=True)
class DecisionPointExample:
    id: str
    conversation_id: str
    language: str
    user_audio: AudioData
    decision_points: tuple[DecisionPoint, ...]
    locale: str | None = None
    words: tuple[TimedWord, ...] = ()
    messages: tuple[ChatMessage, ...] = ()
    group_id: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.conversation_id or not self.language:
            raise ValueError("example id, conversation id, and language are required")
        if not self.decision_points:
            raise ValueError("a decision-point example must contain decision points")
        point_ids = [point.id for point in self.decision_points]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("decision point ids must be unique within an example")
        timestamps = [point.timestamp_s for point in self.decision_points]
        if timestamps != sorted(timestamps):
            raise ValueError("decision points must be chronological")
        if timestamps[-1] > self.user_audio.duration_s + 1 / self.user_audio.sample_rate:
            raise ValueError("decision point exceeds audio duration")
        if tuple(sorted(self.words, key=lambda word: word.start_s)) != self.words:
            raise ValueError("timed words must be chronological")
        if self.words and self.words[-1].end_s > self.user_audio.duration_s + 1e-9:
            raise ValueError("timed word exceeds audio duration")


@dataclass(frozen=True, slots=True)
class ClipClassificationExample:
    id: str
    language: str
    user_audio: AudioData
    label: TurnState
    agent_audio: AudioData | None = None
    transcript: str | None = None
    group_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.language:
            raise ValueError("clip example id and language are required")
        if self.agent_audio is not None:
            tolerance_s = max(1 / self.user_audio.sample_rate, 1 / self.agent_audio.sample_rate)
            if abs(self.user_audio.duration_s - self.agent_audio.duration_s) > tolerance_s:
                raise ValueError("dual-channel clip audio must have equal duration")
        if self.transcript is not None and not self.transcript.strip():
            raise ValueError("clip transcript must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class ClipRequest:
    example_id: str
    language: str
    user_audio: AudioData
    agent_audio: AudioData | None = None
    transcript: str | None = None

    def __post_init__(self) -> None:
        if not self.example_id or not self.language:
            raise ValueError("clip request id and language are required")
        if self.transcript is not None and not self.transcript.strip():
            raise ValueError("clip request transcript must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class StatePrediction:
    example_id: str
    predicted_label: TurnState | None
    scores: Mapping[str, float]
    model_output: Mapping[str, object] = field(default_factory=dict)
    emitted_at_ns: int | None = None

    def __post_init__(self) -> None:
        if not self.example_id or not self.scores:
            raise ValueError("state prediction id and scores are required")
        if self.predicted_label is not None and self.predicted_label.value not in self.scores:
            raise ValueError("predicted state must be present in scores")
        for name, score in self.scores.items():
            if not name or not isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"invalid state score for {name!r}")


@dataclass(frozen=True, slots=True)
class StatePredictionRecord:
    example_id: str
    group_id: str
    language: str
    label: TurnState
    predicted_label: TurnState | None
    scores: Mapping[str, float]
    inference_latency_ms: float
    metadata: Mapping[str, str] = field(default_factory=dict)
    model_output: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id or not self.group_id or not self.language:
            raise ValueError("state prediction identifiers and language are required")
        if self.inference_latency_ms < 0 or not isfinite(self.inference_latency_ms):
            raise ValueError("inference latency must be finite and non-negative")
        if not self.scores:
            raise ValueError("state prediction scores must not be empty")
        if self.predicted_label is not None and self.predicted_label.value not in self.scores:
            raise ValueError("predicted state must be present in scores")
        for name, score in self.scores.items():
            if not name or not isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"invalid state score for {name!r}")


@dataclass(frozen=True, slots=True)
class EvaluationMode:
    channel_mode: ChannelMode
    execution_mode: ExecutionMode
    signal_provenance: SignalProvenance
    transcript_mode: TranscriptMode
    service_mode: ServiceMode
    language_support: SupportStatus

    def __post_init__(self) -> None:
        if self.language_support not in {
            SupportStatus.SUPPORTED,
            SupportStatus.UNVERIFIED_ZERO_SHOT,
        }:
            raise ValueError("language_support must describe native or zero-shot execution")


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    name: str
    revision: str
    native_sample_rate: int
    execution_modes: frozenset[ExecutionMode]
    output_signals: frozenset[str]
    supported_languages: frozenset[str]
    accepted_inputs: frozenset[str] = frozenset({"user_audio"})
    frame_interval_ms: float | None = None
    context_window_s: float | None = None
    algorithmic_lookahead_ms: float = 0.0
    natural_score_point_ms: float | None = None
    artifact_checksum: str | None = None
    artifact_size_bytes: int | None = None
    model_license: str | None = None
    code_license: str | None = None
    known_training_datasets: tuple[str, ...] = ()
    allow_unverified_zero_shot: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.revision:
            raise ValueError("model name and revision are required")
        if self.native_sample_rate <= 0:
            raise ValueError("native sample rate must be positive")
        if not self.execution_modes:
            raise ValueError("at least one execution mode is required")
        if not self.output_signals:
            raise ValueError("at least one output signal is required")
        if self.algorithmic_lookahead_ms < 0:
            raise ValueError("algorithmic lookahead cannot be negative")
        if self.natural_score_point_ms is not None and self.natural_score_point_ms < 0:
            raise ValueError("natural score point cannot be negative")
        if self.artifact_size_bytes is not None and self.artifact_size_bytes < 0:
            raise ValueError("artifact size cannot be negative")

    def support_for_language(self, language: str) -> SupportStatus:
        if language in self.supported_languages:
            return SupportStatus.SUPPORTED
        if self.allow_unverified_zero_shot:
            return SupportStatus.UNVERIFIED_ZERO_SHOT
        return SupportStatus.UNSUPPORTED


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    example_id: str
    decision_point_id: str
    target_at_s: float
    observed_until_s: float
    silence_duration_s: float
    user_audio: AudioData
    transcript: str | None

    def __post_init__(self) -> None:
        if self.target_at_s < 0 or self.observed_until_s < self.target_at_s:
            raise ValueError("invalid causal request timing")
        if self.silence_duration_s < 0 or self.silence_duration_s > self.target_at_s + 1e-9:
            raise ValueError("invalid causal silence duration")
        expected_samples = round(self.observed_until_s * self.user_audio.sample_rate)
        if abs(len(self.user_audio.samples) - expected_samples) > 1:
            raise ValueError("audio prefix does not match observed_until_s")


@dataclass(frozen=True, slots=True)
class TurnPrediction:
    example_id: str
    decision_point_id: str
    available_at_s: float
    observed_until_s: float
    scores: Mapping[str, float]
    emitted_at_ns: int | None = None

    def __post_init__(self) -> None:
        if self.available_at_s + 1e-9 < self.observed_until_s:
            raise ValueError("prediction cannot be available before its inputs")
        if not self.scores:
            raise ValueError("prediction must contain at least one score")
        for name, score in self.scores.items():
            if not name or not isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"invalid probability for signal {name!r}")


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    example_id: str
    conversation_id: str
    group_id: str
    language: str
    decision_point_id: str
    span_id: str
    label: DecisionLabel
    silence_start_s: float
    silence_end_s: float
    target_at_s: float
    observed_until_s: float
    available_at_s: float
    p_eot: float
    inference_latency_ms: float
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InterruptionPredictionRecord:
    """One SID-Bench gold/prediction pair on the clip's causal audio clock."""

    example_id: str
    group_id: str
    language: str
    category: str
    duration_s: float
    gold_break_time_s: float | None
    predicted_break_time_s: float | None

    def __post_init__(self) -> None:
        if not self.example_id or not self.group_id or not self.language or not self.category:
            raise ValueError("interruption prediction identifiers are required")
        if not isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("interruption example duration must be finite and positive")
        if self.gold_break_time_s is not None and (
            not isfinite(self.gold_break_time_s)
            or self.gold_break_time_s < 0
            or self.gold_break_time_s > self.duration_s + 1e-6
        ):
            raise ValueError("gold interruption time is outside the annotated clip")
        if self.predicted_break_time_s is not None and (
            not isfinite(self.predicted_break_time_s) or self.predicted_break_time_s < 0
        ):
            raise ValueError("predicted interruption time must be finite and non-negative")
