from __future__ import annotations

import numpy as np

from turn_eval.core.audio import StreamingLinearResampler


def test_streaming_resampling_matches_single_chunk() -> None:
    samples = np.sin(np.linspace(0, 4 * np.pi, 101)).astype(np.float32)

    single = StreamingLinearResampler(16_000, 12_000).push(samples)
    streaming_resampler = StreamingLinearResampler(16_000, 12_000)
    chunked = np.concatenate(
        (
            streaming_resampler.push(samples[:7]),
            streaming_resampler.push(samples[7:38]),
            streaming_resampler.push(samples[38:]),
        )
    )

    np.testing.assert_allclose(chunked, single, rtol=0, atol=1e-7)


def test_resampler_reset_starts_a_new_stream() -> None:
    samples = np.linspace(-1, 1, 50, dtype=np.float32)
    resampler = StreamingLinearResampler(8_000, 16_000)
    expected = resampler.push(samples)
    resampler.reset()
    actual = resampler.push(samples)
    np.testing.assert_array_equal(actual, expected)


def test_integer_downsampling_matches_exact_source_samples() -> None:
    samples = np.linspace(-1, 1, 48_003, dtype=np.float32)

    actual = StreamingLinearResampler(48_000, 16_000).push(samples)

    np.testing.assert_array_equal(actual, samples[::3])
