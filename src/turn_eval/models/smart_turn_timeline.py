from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

SMART_TURN_TIMELINE_SAMPLE_RATE = 16_000
SMART_TURN_TIMELINE_CHUNK_SAMPLES = 512
SMART_TURN_TIMELINE_OUTPUT_HZ = 12.5
SMART_TURN_TIMELINE_VAD_THRESHOLD = 0.5
SMART_TURN_TIMELINE_PRE_SPEECH_MS = 200.0
SMART_TURN_TIMELINE_STOP_MS = 1_000.0
SMART_TURN_TIMELINE_SETTLE_MS = 2_000.0
SMART_TURN_TIMELINE_MAX_SEGMENT_MS = 8_000.0
SMART_TURN_TIMELINE_SETTLE_STRIDE = 8

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class SmartTurnTimelineConfig:
    sample_rate: int = SMART_TURN_TIMELINE_SAMPLE_RATE
    chunk_samples: int = SMART_TURN_TIMELINE_CHUNK_SAMPLES
    output_hz: float = SMART_TURN_TIMELINE_OUTPUT_HZ
    vad_threshold: float = SMART_TURN_TIMELINE_VAD_THRESHOLD
    pre_speech_ms: float = SMART_TURN_TIMELINE_PRE_SPEECH_MS
    stop_ms: float = SMART_TURN_TIMELINE_STOP_MS
    settle_ms: float = SMART_TURN_TIMELINE_SETTLE_MS
    max_segment_ms: float = SMART_TURN_TIMELINE_MAX_SEGMENT_MS
    settle_stride: int = SMART_TURN_TIMELINE_SETTLE_STRIDE

    def __post_init__(self) -> None:
        positive = {
            "sample_rate": self.sample_rate,
            "chunk_samples": self.chunk_samples,
            "output_hz": self.output_hz,
            "pre_speech_ms": self.pre_speech_ms,
            "stop_ms": self.stop_ms,
            "settle_ms": self.settle_ms,
            "max_segment_ms": self.max_segment_ms,
            "settle_stride": self.settle_stride,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError("SmartTurn timeline durations, rates, and stride must be positive")
        if not 0 <= self.vad_threshold <= 1:
            raise ValueError("SmartTurn timeline VAD threshold must be in [0, 1]")


def score_smart_turn_channel(
    audio: npt.ArrayLike,
    *,
    vad_probability: Callable[[FloatArray], float],
    endpoint_probability: Callable[[FloatArray], float],
    config: SmartTurnTimelineConfig | None = None,
) -> FloatArray:
    """Reproduce the official TurnBench SmartTurn VAD/settling score trace.

    Both callbacks are invoked only after their respective audio chunk/prefix has
    been observed. Returned scores are placed on the official 12.5 Hz frame grid.
    """

    config = config or SmartTurnTimelineConfig()
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError("SmartTurn timeline audio must be mono")
    duration_s = len(waveform) / config.sample_rate
    score_count = int(duration_s * config.output_hz) + 1
    scores = np.zeros(score_count, dtype=np.float32)
    chunk_ms = config.chunk_samples / config.sample_rate * 1_000
    pre_chunks = math.ceil(config.pre_speech_ms / chunk_ms)
    stop_chunks = math.ceil(config.stop_ms / chunk_ms)
    settle_chunks = math.ceil(config.settle_ms / chunk_ms)
    max_chunks = math.ceil(config.max_segment_ms / chunk_ms)

    pre: deque[FloatArray] = deque(maxlen=pre_chunks)
    segment: list[FloatArray] = []
    active = False
    silent_chunks = 0
    extra_chunks = 0
    current_score = 0.0

    stop = len(waveform) - config.chunk_samples + 1
    for sample_index in range(0, max(0, stop), config.chunk_samples):
        chunk = waveform[sample_index : sample_index + config.chunk_samples]
        speaking = vad_probability(chunk) > config.vad_threshold
        output_index = int(sample_index / config.sample_rate * config.output_hz)

        if not active:
            pre.append(chunk)
            if speaking:
                current_score = 0.0
                # The upstream reference appends before copying the pre-roll and
                # then appends the trigger chunk once more. Preserve that exact
                # behavior for artifact parity.
                segment = list(pre) + [chunk]
                active = True
                silent_chunks = 0
                extra_chunks = 0
        elif extra_chunks == 0:
            segment.append(chunk)
            if len(segment) > max_chunks:
                segment.pop(0)
            silent_chunks = 0 if speaking else silent_chunks + 1
            if silent_chunks >= stop_chunks:
                extra_chunks = 1
                current_score = float(endpoint_probability(np.concatenate(segment)))
        elif speaking:
            current_score = 0.0
            segment = [chunk]
            active = True
            silent_chunks = 0
            extra_chunks = 0
        else:
            segment.append(chunk)
            extra_chunks += 1
            if extra_chunks % config.settle_stride == 1:
                current_score = float(endpoint_probability(np.concatenate(segment)))
            if extra_chunks >= settle_chunks:
                active = False
                silent_chunks = 0
                extra_chunks = 0
                pre.clear()

        if not 0 <= current_score <= 1:
            raise ValueError("SmartTurn endpoint probability must be in [0, 1]")
        scores[min(output_index, score_count - 1)] = current_score

    return scores
