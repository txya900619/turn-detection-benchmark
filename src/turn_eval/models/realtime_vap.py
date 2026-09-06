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

# The pinned revision publishes one checkpoint per (frame rate, context length)
# pair, and the file name encodes both. Holding the frame rate fixed and varying
# only the context length gives a controlled ablation of context on long HOLD.
REALTIME_VAP_CONTEXT_CHOICES: dict[float, tuple[float, ...]] = {
    5.0: (3.0, 5.0, 20.0),
    10.0: (3.0, 5.0, 20.0),
    20.0: (2.5,),
}


def realtime_vap_checkpoint_filename(frame_rate_hz: float, context_s: float) -> str:
    """Name the pinned trilingual checkpoint for a frame rate and context length."""
    available = REALTIME_VAP_CONTEXT_CHOICES.get(frame_rate_hz)
    if available is None:
        raise ValueError(f"no pinned Realtime-VAP checkpoint at {frame_rate_hz} Hz")
    if context_s not in available:
        raise ValueError(
            f"no pinned Realtime-VAP checkpoint at {frame_rate_hz} Hz with "
            f"{context_s} s of context; available: {available}"
        )
    rate = f"{frame_rate_hz:g}"
    return f"vap_state_dict_tri_{rate}hz_{round(context_s * 1000)}msec.pt"


def realtime_vap_model_name(frame_rate_hz: float, context_s: float) -> str:
    """Identity used in run artifacts, so context variants never collide."""
    return f"maai-realtime-vap-trilingual-cpc-{frame_rate_hz:g}hz-{context_s:g}s"


class RealtimeVAPSingleChannelEOTAdapter:
    """Run dual-channel multilingual VAP with a silent second-channel fallback."""

    def __init__(
        self,
        model: Any,
        *,
        artifact_checksum: str | None = None,
        artifact_size_bytes: int | None = None,
        frame_rate_hz: float = REALTIME_VAP_FRAME_RATE_HZ,
        context_s: float = REALTIME_VAP_CONTEXT_S,
    ) -> None:
        self._model = model
        self._example_id: str | None = None
        self._processed_samples = 0
        self._last_p_user: float | None = None
        self._frame_samples = round(REALTIME_VAP_SAMPLE_RATE / frame_rate_hz)
        self._descriptor = ModelDescriptor(
            name=realtime_vap_model_name(frame_rate_hz, context_s),
            revision=REALTIME_VAP_MODEL_REVISION,
            native_sample_rate=REALTIME_VAP_SAMPLE_RATE,
            execution_modes=frozenset({ExecutionMode.STREAMING}),
            output_signals=frozenset({"eot"}),
            supported_languages=frozenset({"en", "zh", "ja"}),
            accepted_inputs=frozenset({"user_audio"}),
            frame_interval_ms=1_000.0 / frame_rate_hz,
            context_window_s=context_s,
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
        complete_samples = len(samples) // self._frame_samples * self._frame_samples

        silence = np.zeros(self._frame_samples, dtype=np.float32)
        while self._processed_samples < complete_samples:
            end = self._processed_samples + self._frame_samples
            self._model.process(samples[self._processed_samples : end], silence)
            try:
                result = self._model.result_dict_queue.get_nowait()
            except queue.Empty as error:
                frame = self._processed_samples // self._frame_samples
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
