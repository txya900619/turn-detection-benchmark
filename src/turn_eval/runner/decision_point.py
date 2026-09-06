from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import perf_counter

from turn_eval.core.audio import StreamingLinearResampler
from turn_eval.core.types import (
    AudioData,
    DecisionPointExample,
    DecisionRequest,
    EvaluationMode,
    ExecutionMode,
    PredictionRecord,
    TranscriptMode,
)
from turn_eval.models.base import PointwiseModelAdapter


@dataclass(frozen=True, slots=True)
class DecisionPointRun:
    predictions: tuple[PredictionRecord, ...]
    initialization_time_ms: float
    inference_wall_s: float
    resampling_wall_s: float
    audio_s: float

    @property
    def rtf(self) -> float:
        return self.inference_wall_s / self.audio_s if self.audio_s else 0.0

    @property
    def realtime_multiplier(self) -> float | None:
        return self.audio_s / self.inference_wall_s if self.inference_wall_s else None


def _resample(audio: AudioData, output_rate: int) -> AudioData:
    if audio.sample_rate == output_rate:
        return audio
    resampler = StreamingLinearResampler(audio.sample_rate, output_rate)
    return AudioData(resampler.push(audio.samples), output_rate)


def _validate_compatibility(
    examples: tuple[DecisionPointExample, ...],
    model: PointwiseModelAdapter,
    mode: EvaluationMode,
) -> None:
    descriptor = model.descriptor
    if mode.execution_mode not in {ExecutionMode.POINTWISE, ExecutionMode.STREAMING}:
        raise ValueError("decision-point runner requires pointwise or streaming execution")
    if mode.execution_mode not in descriptor.execution_modes:
        raise ValueError("model does not support the selected decision-point execution mode")
    if "eot" not in descriptor.output_signals:
        raise ValueError("model does not produce a native EOT signal")
    if (
        mode.transcript_mode is not TranscriptMode.NONE
        and "transcript" not in descriptor.accepted_inputs
    ):
        raise ValueError("transcript mode selected for an audio-only model")
    if mode.transcript_mode is TranscriptMode.STREAMING_ASR:
        raise ValueError("streaming ASR replay is not implemented in M0")
    for example in examples:
        actual_support = descriptor.support_for_language(example.language)
        if actual_support is not mode.language_support:
            raise ValueError(
                f"language support mismatch for {example.language}: "
                f"descriptor resolves {actual_support.value}"
            )


def run_decision_points(
    examples: Iterable[DecisionPointExample],
    model: PointwiseModelAdapter,
    mode: EvaluationMode,
    *,
    transcript_lag_s: float = 0.0,
    initialization_time_ms: float = 0.0,
    score_point_s: float | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> DecisionPointRun:
    examples = tuple(examples)
    if transcript_lag_s < 0:
        raise ValueError("transcript lag cannot be negative")
    if score_point_s is not None and score_point_s < 0:
        raise ValueError("score point cannot be negative")
    _validate_compatibility(examples, model, mode)

    records: list[PredictionRecord] = []
    inference_wall_s = 0.0
    resampling_wall_s = 0.0
    audio_s = sum(example.user_audio.duration_s for example in examples)
    lookahead_s = model.descriptor.algorithmic_lookahead_ms / 1_000
    total_points = sum(
        score_point_s is None or abs(point.silence_duration_s - score_point_s) <= 1e-6
        for example in examples
        for point in example.decision_points
    )
    completed_points = 0

    for example in examples:
        resample_started = perf_counter()
        native_audio = _resample(example.user_audio, model.descriptor.native_sample_rate)
        resampling_wall_s += perf_counter() - resample_started

        for point in example.decision_points:
            if score_point_s is not None and abs(point.silence_duration_s - score_point_s) > 1e-6:
                continue
            observed_until_s = point.timestamp_s + lookahead_s
            required_samples = round(observed_until_s * native_audio.sample_rate)
            if required_samples - len(native_audio.samples) > 1:
                raise ValueError(f"lookahead for {point.id} extends beyond available fixture audio")
            prefix = native_audio.prefix(observed_until_s)
            transcript: str | None = None
            if mode.transcript_mode is TranscriptMode.ORACLE_CAUSAL:
                cutoff = observed_until_s - transcript_lag_s
                transcript = " ".join(word.text for word in example.words if word.end_s <= cutoff)

            request = DecisionRequest(
                example_id=example.id,
                decision_point_id=point.id,
                target_at_s=point.timestamp_s,
                observed_until_s=observed_until_s,
                silence_duration_s=point.silence_duration_s,
                user_audio=prefix,
                transcript=transcript,
            )
            started = perf_counter()
            prediction = model.predict(request)
            elapsed_s = perf_counter() - started
            inference_wall_s += elapsed_s
            if prediction.example_id != example.id or prediction.decision_point_id != point.id:
                raise ValueError("adapter returned prediction identifiers for another request")
            if abs(prediction.observed_until_s - observed_until_s) > 1e-9:
                raise ValueError("adapter reported an inconsistent observed_until_s")

            records.append(
                PredictionRecord(
                    example_id=example.id,
                    conversation_id=example.conversation_id,
                    group_id=example.group_id or example.conversation_id,
                    language=example.language,
                    decision_point_id=point.id,
                    span_id=point.span_id,
                    label=point.label,
                    silence_start_s=point.silence_start_s,
                    silence_end_s=point.silence_end_s,
                    target_at_s=point.timestamp_s,
                    observed_until_s=prediction.observed_until_s,
                    available_at_s=prediction.available_at_s,
                    p_eot=float(prediction.scores["eot"]),
                    inference_latency_ms=elapsed_s * 1_000,
                )
            )
            completed_points += 1
            if progress is not None:
                progress(completed_points, total_points)

    if score_point_s is not None and not records:
        raise ValueError(f"dataset has no decisions at score point {score_point_s}s")

    return DecisionPointRun(
        predictions=tuple(records),
        initialization_time_ms=initialization_time_ms,
        inference_wall_s=inference_wall_s,
        resampling_wall_s=resampling_wall_s,
        audio_s=audio_s,
    )
