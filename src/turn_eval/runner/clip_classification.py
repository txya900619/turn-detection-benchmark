from __future__ import annotations

from collections.abc import Callable, Iterable, Sized
from dataclasses import dataclass
from itertools import batched
from time import perf_counter

from turn_eval.core.audio import StreamingLinearResampler
from turn_eval.core.types import (
    AudioData,
    ChannelMode,
    ClipClassificationExample,
    ClipRequest,
    EvaluationMode,
    ExecutionMode,
    StatePredictionRecord,
    TranscriptMode,
)
from turn_eval.models.base import ClipClassificationModelAdapter


@dataclass(frozen=True, slots=True)
class ClipClassificationRun:
    predictions: tuple[StatePredictionRecord, ...]
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
    return AudioData(
        StreamingLinearResampler(audio.sample_rate, output_rate).push(audio.samples), output_rate
    )


def _prepare_audio(
    audio: AudioData,
    model: ClipClassificationModelAdapter,
) -> AudioData:
    model_resampler = getattr(model, "resample_audio", None)
    prepared = (
        model_resampler(audio)
        if callable(model_resampler)
        else _resample(audio, model.descriptor.native_sample_rate)
    )
    if prepared.sample_rate != model.descriptor.native_sample_rate:
        raise ValueError("adapter resampler did not produce the model native sample rate")
    return prepared


def _validate_model_compatibility(
    model: ClipClassificationModelAdapter, mode: EvaluationMode
) -> None:
    descriptor = model.descriptor
    if mode.execution_mode not in {ExecutionMode.OFFLINE, ExecutionMode.CAUSAL_OFFLINE}:
        raise ValueError("clip-classification runner requires offline execution")
    if mode.execution_mode not in descriptor.execution_modes:
        raise ValueError("model does not support the selected clip execution mode")
    if "turn_state" not in descriptor.output_signals:
        raise ValueError("model does not produce a turn_state signal")
    if (
        mode.channel_mode is ChannelMode.NATIVE_DUAL
        and "agent_audio" not in descriptor.accepted_inputs
    ):
        raise ValueError("dual-channel mode selected for a single-channel model")
    if mode.transcript_mode is TranscriptMode.ORACLE_REFERENCE:
        if "transcript" not in descriptor.accepted_inputs:
            raise ValueError("reference transcript selected for an audio-only model")
    elif mode.transcript_mode is not TranscriptMode.NONE:
        raise ValueError("clip runner supports only none or oracle_reference transcripts")


def _validate_example_compatibility(
    example: ClipClassificationExample,
    model: ClipClassificationModelAdapter,
    mode: EvaluationMode,
) -> None:
    if mode.channel_mode is ChannelMode.NATIVE_DUAL and example.agent_audio is None:
        raise ValueError("dual-channel mode requires agent audio on every clip")
    if mode.channel_mode is not ChannelMode.NATIVE_DUAL and example.agent_audio is not None:
        raise ValueError("single-channel mode cannot silently discard agent audio")
    if mode.transcript_mode is TranscriptMode.NONE and example.transcript is not None:
        raise ValueError("audio-only mode cannot expose reference transcripts")
    if mode.transcript_mode is TranscriptMode.ORACLE_REFERENCE and example.transcript is None:
        raise ValueError("reference transcript mode requires every clip transcript")
    actual_support = model.descriptor.support_for_language(example.language)
    if actual_support is not mode.language_support:
        raise ValueError(
            f"language support mismatch for {example.language}: "
            f"descriptor resolves {actual_support.value}"
        )


def run_clip_classification(
    examples: Iterable[ClipClassificationExample],
    model: ClipClassificationModelAdapter,
    mode: EvaluationMode,
    *,
    initialization_time_ms: float = 0.0,
    progress: Callable[[int, int], None] | None = None,
    total_examples: int | None = None,
    batch_size: int = 1,
) -> ClipClassificationRun:
    _validate_model_compatibility(model, mode)
    if batch_size <= 0:
        raise ValueError("clip batch size must be positive")
    predict_batch = getattr(model, "predict_clips", None)
    if batch_size > 1 and not callable(predict_batch):
        raise ValueError("clip adapter does not implement batched prediction")
    if total_examples is None and isinstance(examples, Sized):
        total_examples = len(examples)
    if total_examples is not None and total_examples <= 0:
        raise ValueError("total_examples must be positive")
    if progress is not None and total_examples is None:
        raise ValueError("streaming clip progress requires total_examples")

    records: list[StatePredictionRecord] = []
    inference_wall_s = 0.0
    resampling_wall_s = 0.0
    audio_s = 0.0
    completed = 0
    for example_batch in batched(examples, batch_size):
        for example in example_batch:
            _validate_example_compatibility(example, model, mode)
            audio_s += example.user_audio.duration_s
        resample_started = perf_counter()
        requests = tuple(
            ClipRequest(
                example_id=example.id,
                language=example.language,
                user_audio=_prepare_audio(example.user_audio, model),
                agent_audio=(
                    None
                    if example.agent_audio is None
                    else _prepare_audio(example.agent_audio, model)
                ),
                transcript=example.transcript,
            )
            for example in example_batch
        )
        resampling_wall_s += perf_counter() - resample_started
        started = perf_counter()
        predictions = (
            tuple(predict_batch(requests)) if batch_size > 1 else (model.predict_clip(requests[0]),)
        )
        elapsed_s = perf_counter() - started
        inference_wall_s += elapsed_s
        if len(predictions) != len(example_batch):
            raise ValueError("adapter returned the wrong number of batched predictions")
        for example, prediction in zip(example_batch, predictions, strict=True):
            if prediction.example_id != example.id:
                raise ValueError("adapter returned a prediction identifier for another request")
            records.append(
                StatePredictionRecord(
                    example_id=example.id,
                    group_id=example.group_id or example.id,
                    language=example.language,
                    label=example.label,
                    predicted_label=prediction.predicted_label,
                    scores=prediction.scores,
                    inference_latency_ms=elapsed_s * 1_000,
                    metadata=example.metadata,
                    model_output=prediction.model_output,
                )
            )
        completed += len(example_batch)
        if progress is not None:
            progress(completed, total_examples)

    if not records:
        raise ValueError("cannot run clip classification on an empty dataset")
    if total_examples is not None and len(records) != total_examples:
        raise ValueError(
            f"clip iterable produced {len(records)} examples; expected {total_examples}"
        )

    return ClipClassificationRun(
        predictions=tuple(records),
        initialization_time_ms=initialization_time_ms,
        inference_wall_s=inference_wall_s,
        resampling_wall_s=resampling_wall_s,
        audio_s=audio_s,
    )
