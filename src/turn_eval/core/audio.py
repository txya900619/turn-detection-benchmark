from __future__ import annotations

import numpy as np
import numpy.typing as npt


class StreamingLinearResampler:
    """Small deterministic resampler with state preserved across input chunks.

    Linear interpolation is intentionally used for the M0 timing fixture. Production
    model adapters may replace the implementation while retaining the stateful seam.
    """

    _OUTPUT_BATCH_SIZE = 1_000_000

    def __init__(self, input_rate: int, output_rate: int) -> None:
        if input_rate <= 0 or output_rate <= 0:
            raise ValueError("sample rates must be positive")
        self.input_rate = input_rate
        self.output_rate = output_rate
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start = 0
        self._total_input = 0
        self._next_output = 0

    def push(self, chunk: npt.ArrayLike) -> npt.NDArray[np.float32]:
        values = np.asarray(chunk, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError("resampler input must be mono")
        if not np.all(np.isfinite(values)):
            raise ValueError("resampler input must be finite")
        if not len(values):
            return np.empty(0, dtype=np.float32)

        if not len(self._buffer):
            self._buffer_start = self._total_input
            self._buffer = values.copy()
        else:
            self._buffer = np.concatenate((self._buffer, values))
        self._total_input += len(values)
        last_input_index = self._total_input - 1

        ratio = self.input_rate / self.output_rate
        last_output_index = int(np.floor(last_input_index / ratio))
        batches: list[npt.NDArray[np.float32]] = []
        for start in range(
            self._next_output,
            last_output_index + 1,
            self._OUTPUT_BATCH_SIZE,
        ):
            stop = min(start + self._OUTPUT_BATCH_SIZE, last_output_index + 1)
            positions = np.arange(start, stop, dtype=np.float64) * ratio
            lower = np.floor(positions).astype(np.int64)
            upper = np.ceil(positions).astype(np.int64)
            local_lower = lower - self._buffer_start
            local_upper = upper - self._buffer_start
            fractions = positions - lower
            left = self._buffer[local_lower]
            right = self._buffer[local_upper]
            batches.append(np.asarray(left + (right - left) * fractions, dtype=np.float32))
        if last_output_index >= self._next_output:
            self._next_output = last_output_index + 1

        next_position = self._next_output * ratio
        keep_from = min(int(np.floor(next_position)), last_input_index)
        drop = keep_from - self._buffer_start
        if drop > 0:
            self._buffer = self._buffer[drop:].copy()
            self._buffer_start = keep_from

        if not batches:
            return np.empty(0, dtype=np.float32)
        if len(batches) == 1:
            return batches[0]
        return np.concatenate(batches)

    def reset(self) -> None:
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start = 0
        self._total_input = 0
        self._next_output = 0
