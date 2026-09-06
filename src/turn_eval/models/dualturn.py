from __future__ import annotations

from collections.abc import Callable
from time import monotonic_ns
from typing import Protocol

import numpy as np

from turn_eval.core.types import DecisionRequest, ExecutionMode, ModelDescriptor, TurnPrediction

DUALTURN_REPOSITORY = "anyreach-ai/dualturn-endpointing"
DUALTURN_REVISION = "c3860ed71210fe0144af35af340fd7a4dec3d2d3"
DUALTURN_FILENAME = "stream_tick.onnx"
DUALTURN_SHA256 = "6700ea2f919b4c66355b3dafb4f91f2db9dae9e18a302b812730c04340bf819f"
DUALTURN_ARTIFACT_BYTES = 161_722_822
DUALTURN_SAMPLE_RATE = 24_000
DUALTURN_FRAME_SAMPLES = 1_920
DUALTURN_FRAME_INTERVAL_MS = 80.0


class DualTurnStream(Protocol):
    def push(self, chunk: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]: ...


class DualTurnSingleChannelEOTAdapter:
    """Causal DualTurn EOT with CH0=user and a silent CH1 fallback."""

    def __init__(
        self,
        stream_factory: Callable[[], DualTurnStream],
        *,
        artifact_checksum: str = f"sha256:{DUALTURN_SHA256}",
        artifact_size_bytes: int = DUALTURN_ARTIFACT_BYTES,
    ) -> None:
        self._stream_factory = stream_factory
        self._stream: DualTurnStream | None = None
        self._example_id: str | None = None
        self._processed_samples = 0
        self._last_eot: float | None = None
        self._descriptor = ModelDescriptor(
            name="dualturn-endpointing",
            revision=DUALTURN_REVISION,
            native_sample_rate=DUALTURN_SAMPLE_RATE,
            execution_modes=frozenset({ExecutionMode.STREAMING}),
            output_signals=frozenset({"eot"}),
            supported_languages=frozenset({"en"}),
            accepted_inputs=frozenset({"user_audio"}),
            frame_interval_ms=DUALTURN_FRAME_INTERVAL_MS,
            artifact_checksum=artifact_checksum,
            artifact_size_bytes=artifact_size_bytes,
            model_license="Apache-2.0",
            code_license="Apache-2.0",
            known_training_datasets=(
                "Switchboard",
                "OtoSpeech",
                "private dual-channel AI-agent phone calls",
            ),
            allow_unverified_zero_shot=True,
        )

    @property
    def descriptor(self) -> ModelDescriptor:
        return self._descriptor

    def _reset(self, example_id: str) -> None:
        self._stream = self._stream_factory()
        self._example_id = example_id
        self._processed_samples = 0
        self._last_eot = None

    def predict(self, request: DecisionRequest) -> TurnPrediction:
        if request.user_audio.sample_rate != DUALTURN_SAMPLE_RATE:
            raise ValueError("DualTurn requires 24 kHz audio")
        if request.example_id != self._example_id:
            self._reset(request.example_id)
        if self._stream is None:  # pragma: no cover - established by reset above
            raise RuntimeError("DualTurn stream was not initialized")

        samples = request.user_audio.samples
        if len(samples) < self._processed_samples:
            raise ValueError("streaming audio prefix cannot move backwards")
        complete_samples = len(samples) // DUALTURN_FRAME_SAMPLES * DUALTURN_FRAME_SAMPLES
        while self._processed_samples < complete_samples:
            end = self._processed_samples + DUALTURN_FRAME_SAMPLES
            user = samples[self._processed_samples : end]
            chunk = np.stack((user, np.zeros_like(user))).astype(np.float32, copy=False)
            eot, _, _ = self._stream.push(chunk)
            self._last_eot = float(eot)
            if not np.isfinite(self._last_eot) or not 0.0 <= self._last_eot <= 1.0:
                raise RuntimeError("DualTurn emitted an invalid EOT probability")
            self._processed_samples = end

        if self._last_eot is None:
            raise ValueError("DualTurn needs at least one complete 80 ms frame")
        return TurnPrediction(
            example_id=request.example_id,
            decision_point_id=request.decision_point_id,
            available_at_s=request.observed_until_s,
            observed_until_s=request.observed_until_s,
            scores={"eot": self._last_eot},
            emitted_at_ns=monotonic_ns(),
        )
