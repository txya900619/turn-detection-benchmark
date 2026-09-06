from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from turn_eval.core.types import AudioData, ExecutionMode, ModelDescriptor

RMS_VAD_WINDOW_MS = 20.0
RMS_VAD_THRESHOLD = 0.01
RMS_VAD_REVISION = "turnbench-e02d547803fd0e45221afcaeb1cd24b61ae7568e"


@dataclass(frozen=True, slots=True)
class RMSVADEvents:
    eot: tuple[float, ...]
    interruption: tuple[float, ...]


class RMSVADAdapter:
    """Exact local implementation of TurnBench's reference RMS energy VAD."""

    def __init__(
        self,
        *,
        window_ms: float = RMS_VAD_WINDOW_MS,
        threshold: float = RMS_VAD_THRESHOLD,
        native_sample_rate: int = 48_000,
    ) -> None:
        if window_ms <= 0:
            raise ValueError("RMS VAD window_ms must be positive")
        if threshold < 0:
            raise ValueError("RMS VAD threshold must be non-negative")
        if native_sample_rate <= 0:
            raise ValueError("RMS VAD native_sample_rate must be positive")
        self.window_ms = float(window_ms)
        self.threshold = float(threshold)
        self.descriptor = ModelDescriptor(
            name="turnbench-rms-vad",
            revision=RMS_VAD_REVISION,
            native_sample_rate=native_sample_rate,
            execution_modes=frozenset({ExecutionMode.STREAMING}),
            output_signals=frozenset({"eot", "interruption"}),
            supported_languages=frozenset({"en"}),
            frame_interval_ms=self.window_ms,
            algorithmic_lookahead_ms=0.0,
            artifact_size_bytes=0,
            model_license="N/A (rule-based)",
            code_license="MIT",
        )

    def speaking_windows(self, audio: AudioData) -> np.ndarray:
        samples_per_window = int(self.window_ms * audio.sample_rate / 1_000)
        if samples_per_window <= 0:
            raise ValueError("RMS VAD window is shorter than one audio sample")
        window_count = len(audio.samples) // samples_per_window
        if window_count == 0:
            return np.empty(0, dtype=np.bool_)
        windows = audio.samples[: window_count * samples_per_window].reshape(
            window_count, samples_per_window
        )
        rms = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1))
        return rms > self.threshold

    def predict(self, audio: AudioData) -> RMSVADEvents:
        speaking = self.speaking_windows(audio)
        eot: list[float] = []
        interruption: list[float] = []
        window_s = self.window_ms / 1_000
        for window_index in range(1, len(speaking)):
            if speaking[window_index] == speaking[window_index - 1]:
                continue
            commit_time_s = (window_index + 1) * window_s
            if speaking[window_index]:
                interruption.append(commit_time_s)
            else:
                eot.append(commit_time_s)
        return RMSVADEvents(eot=tuple(eot), interruption=tuple(interruption))
