from __future__ import annotations

import queue
from time import monotonic_ns
from typing import Any

import numpy as np

from turn_eval.core.types import DecisionRequest, ExecutionMode, ModelDescriptor, TurnPrediction

REALTIME_VAP_MODEL_REPOSITORY = "maai-kyoto/vap_tri"
REALTIME_VAP_MODEL_REVISION = "1a7721ad12a26ac8cd6025bec5543f9c005b9f51"
REALTIME_VAP_MODEL_FILENAME = "vap_state_dict_tri_10hz_5000msec.pt"
REALTIME_VAP_MAAI_VERSION = "0.2.18"
REALTIME_VAP_SAMPLE_RATE = 16_000
REALTIME_VAP_FRAME_RATE_HZ = 10.0
REALTIME_VAP_FRAME_SAMPLES = 1_600
REALTIME_VAP_CONTEXT_S = 5.0


class RealtimeVAPSingleChannelEOTAdapter:
    """Run dual-channel multilingual VAP with a silent second-channel fallback."""

    def __init__(
        self,
        model: Any,
        *,
        artifact_checksum: str | None = None,
        artifact_size_bytes: int | None = None,
    ) -> None:
        self._model = model
        self._example_id: str | None = None
        self._processed_samples = 0
        self._last_p_user: float | None = None
        self._descriptor = ModelDescriptor(
            name="maai-realtime-vap-trilingual-cpc-10hz-5s",
            revision=REALTIME_VAP_MODEL_REVISION,
            native_sample_rate=REALTIME_VAP_SAMPLE_RATE,
            execution_modes=frozenset({ExecutionMode.STREAMING}),
            output_signals=frozenset({"eot"}),
            supported_languages=frozenset({"en", "zh", "ja"}),
            accepted_inputs=frozenset({"user_audio"}),
            frame_interval_ms=100.0,
            context_window_s=REALTIME_VAP_CONTEXT_S,
            artifact_checksum=artifact_checksum,
            artifact_size_bytes=artifact_size_bytes,
            model_license="CC-BY-NC-SA-4.0",
            code_license="MIT",
        )

    @property
    def descriptor(self) -> ModelDescriptor:
        return self._descriptor

    def _reset(self, example_id: str) -> None:
        self._model.reset_runtime_state()
        while True:
            try:
                self._model.result_dict_queue.get_nowait()
            except queue.Empty:
                break
        self._example_id = example_id
        self._processed_samples = 0
        self._last_p_user = None

    def predict(self, request: DecisionRequest) -> TurnPrediction:
        if request.user_audio.sample_rate != REALTIME_VAP_SAMPLE_RATE:
            raise ValueError("Realtime-VAP requires 16 kHz audio")
        if request.example_id != self._example_id:
            self._reset(request.example_id)

        samples = request.user_audio.samples
        if len(samples) < self._processed_samples:
            raise ValueError("streaming audio prefix cannot move backwards")
        complete_samples = len(samples) // REALTIME_VAP_FRAME_SAMPLES * REALTIME_VAP_FRAME_SAMPLES

        silence = np.zeros(REALTIME_VAP_FRAME_SAMPLES, dtype=np.float32)
        while self._processed_samples < complete_samples:
            end = self._processed_samples + REALTIME_VAP_FRAME_SAMPLES
            self._model.process(samples[self._processed_samples : end], silence)
            try:
                result = self._model.result_dict_queue.get_nowait()
            except queue.Empty as error:
                frame = self._processed_samples // REALTIME_VAP_FRAME_SAMPLES
                raise RuntimeError(f"Realtime-VAP emitted no result for frame {frame}") from error
            p_now = result.get("p_now")
            if not isinstance(p_now, list | tuple) or len(p_now) != 2:
                raise RuntimeError("Realtime-VAP p_now must contain two speaker probabilities")
            self._last_p_user = float(p_now[0])
            if not 0.0 <= self._last_p_user <= 1.0:
                raise RuntimeError("Realtime-VAP emitted an invalid user probability")
            self._processed_samples = end

        if self._last_p_user is None:
            raise ValueError("Realtime-VAP needs at least 100 ms before its first decision")
        return TurnPrediction(
            example_id=request.example_id,
            decision_point_id=request.decision_point_id,
            available_at_s=request.observed_until_s,
            observed_until_s=request.observed_until_s,
            scores={"eot": 1.0 - self._last_p_user},
            emitted_at_ns=monotonic_ns(),
        )
