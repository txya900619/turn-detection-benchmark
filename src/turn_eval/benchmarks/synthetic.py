from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from turn_eval.core.types import (
    AudioData,
    DecisionLabel,
    DecisionPoint,
    DecisionPointExample,
    TimedWord,
)


@dataclass(frozen=True, slots=True)
class _Span:
    start_s: float
    end_s: float
    label: DecisionLabel


def _speech_wave(sample_rate: int, duration_s: float) -> np.ndarray:
    time = np.arange(round(duration_s * sample_rate), dtype=np.float32) / sample_rate
    return (0.25 * np.sin(2 * np.pi * 37 * time)).astype(np.float32)


def _make_example(
    *,
    example_id: str,
    speech_ranges: tuple[tuple[float, float], ...],
    spans: tuple[_Span, ...],
    duration_s: float,
    sample_rate: int = 1_000,
) -> DecisionPointExample:
    samples = np.zeros(round(duration_s * sample_rate), dtype=np.float32)
    for start_s, end_s in speech_ranges:
        start = round(start_s * sample_rate)
        end = round(end_s * sample_rate)
        samples[start:end] = _speech_wave(sample_rate, end_s - start_s)

    points: list[DecisionPoint] = []
    for span_index, span in enumerate(spans):
        span_id = f"{example_id}-span-{span_index}"
        step = 1
        while True:
            timestamp_s = round(span.start_s + step * 0.1, 10)
            if span.label is DecisionLabel.HOLD and timestamp_s >= span.end_s - 1e-9:
                break
            if span.label is DecisionLabel.EOT and timestamp_s > span.end_s + 1e-9:
                break
            points.append(
                DecisionPoint(
                    id=f"{span_id}-at-{round((timestamp_s - span.start_s) * 1_000)}ms",
                    span_id=span_id,
                    timestamp_s=timestamp_s,
                    silence_start_s=span.start_s,
                    silence_end_s=span.end_s,
                    label=span.label,
                )
            )
            step += 1

    words = tuple(
        TimedWord(text=f"word-{index}", start_s=start, end_s=min(end, start + 0.2))
        for index, (start, end) in enumerate(speech_ranges)
    )
    return DecisionPointExample(
        id=example_id,
        conversation_id=f"conversation-{example_id}",
        group_id=f"speaker-pair-{example_id}",
        language="en",
        locale="en-US",
        user_audio=AudioData(samples, sample_rate),
        decision_points=tuple(points),
        words=words,
    )


def synthetic_decision_point_examples() -> tuple[DecisionPointExample, ...]:
    """Return deterministic turns with short and long mid-turn pauses."""

    return (
        _make_example(
            example_id="short-hold",
            speech_ranges=((0.0, 0.8), (1.2, 2.0)),
            spans=(
                _Span(0.8, 1.2, DecisionLabel.HOLD),
                _Span(2.0, 3.2, DecisionLabel.EOT),
            ),
            duration_s=4.2,
        ),
        _make_example(
            example_id="long-hold",
            speech_ranges=((0.0, 0.6), (1.4, 2.1)),
            spans=(
                _Span(0.6, 1.4, DecisionLabel.HOLD),
                _Span(2.1, 3.3, DecisionLabel.EOT),
            ),
            duration_s=4.3,
        ),
        _make_example(
            example_id="no-hold",
            speech_ranges=((0.0, 1.0),),
            spans=(_Span(1.0, 2.2, DecisionLabel.EOT),),
            duration_s=3.2,
        ),
    )
