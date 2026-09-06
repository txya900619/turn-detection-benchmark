from __future__ import annotations

import hashlib
from pathlib import Path
from time import monotonic_ns
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download

from turn_eval.core.types import (
    AudioData,
    ClipRequest,
    DecisionRequest,
    ExecutionMode,
    ModelDescriptor,
    StatePrediction,
    TurnPrediction,
    TurnState,
)

SMART_TURN_REPO_ID = "pipecat-ai/smart-turn-v3"
SMART_TURN_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
SMART_TURN_FILENAME = "smart-turn-v3.2-cpu.onnx"
SMART_TURN_SAMPLE_RATE = 16_000
SMART_TURN_CONTEXT_S = 8.0
SMART_TURN_SCORE_POINT_MS = 200.0
SMART_TURN_FASTTURN_THRESHOLD = 0.5


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SmartTurnV32Adapter:
    """Pinned Smart Turn v3.2 CPU adapter using the upstream preprocessing path."""

    def __init__(
        self,
        *,
        revision: str = SMART_TURN_REVISION,
        filename: str = SMART_TURN_FILENAME,
        intra_op_threads: int = 1,
        local_files_only: bool = False,
        session: Any | None = None,
        feature_extractor: Any | None = None,
        artifact_checksum: str | None = None,
        artifact_size_bytes: int | None = None,
    ) -> None:
        if intra_op_threads <= 0:
            raise ValueError("intra-op thread count must be positive")
        if (session is None) != (feature_extractor is None):
            raise ValueError("session and feature extractor must be supplied together")

        if session is None:
            model_path = Path(
                hf_hub_download(
                    repo_id=SMART_TURN_REPO_ID,
                    filename=filename,
                    revision=revision,
                    local_files_only=local_files_only,
                )
            )
            session, feature_extractor = self._load_runtime(
                model_path,
                intra_op_threads=intra_op_threads,
            )
            artifact_checksum = f"sha256:{_sha256_file(model_path)}"
            artifact_size_bytes = model_path.stat().st_size

        self._session = session
        self._feature_extractor = feature_extractor
        self._max_samples = round(SMART_TURN_SAMPLE_RATE * SMART_TURN_CONTEXT_S)
        self._descriptor = ModelDescriptor(
            name="smart-turn-v3.2-cpu",
            revision=revision,
            native_sample_rate=SMART_TURN_SAMPLE_RATE,
            execution_modes=frozenset({ExecutionMode.POINTWISE}),
            output_signals=frozenset({"eot"}),
            supported_languages=frozenset({"en", "zh"}),
            accepted_inputs=frozenset({"user_audio"}),
            context_window_s=SMART_TURN_CONTEXT_S,
            natural_score_point_ms=SMART_TURN_SCORE_POINT_MS,
            artifact_checksum=artifact_checksum,
            artifact_size_bytes=artifact_size_bytes,
            model_license="BSD-2-Clause",
            code_license="BSD-2-Clause",
            known_training_datasets=(
                "pipecat-ai/smart-turn-data-v3.1-train",
                "pipecat-ai/smart-turn-data-v3.1-test",
            ),
        )

    @staticmethod
    def _load_runtime(model_path: Path, *, intra_op_threads: int) -> tuple[Any, Any]:
        try:
            import onnxruntime as ort
            from transformers import WhisperFeatureExtractor
        except ImportError as error:  # pragma: no cover - exercised without optional extra
            raise RuntimeError(
                "SmartTurn dependencies are missing; run `uv sync --extra smart-turn`"
            ) from error

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = intra_op_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        return session, WhisperFeatureExtractor(chunk_length=SMART_TURN_CONTEXT_S)

    @property
    def descriptor(self) -> ModelDescriptor:
        return self._descriptor

    def predict(self, request: DecisionRequest) -> TurnPrediction:
        if request.user_audio.sample_rate != SMART_TURN_SAMPLE_RATE:
            raise ValueError("SmartTurn requires 16 kHz audio")
        audio = np.asarray(request.user_audio.samples, dtype=np.float32)
        if len(audio) > self._max_samples:
            audio = audio[-self._max_samples :]
        elif len(audio) < self._max_samples:
            audio = np.pad(audio, (self._max_samples - len(audio), 0))

        extracted = self._feature_extractor(
            audio,
            sampling_rate=SMART_TURN_SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=self._max_samples,
            truncation=True,
            do_normalize=True,
        )
        features = np.asarray(extracted["input_features"], dtype=np.float32)
        probability = float(
            np.asarray(self._session.run(None, {"input_features": features})[0]).reshape(-1)[0]
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"SmartTurn returned an invalid probability: {probability}")
        return TurnPrediction(
            example_id=request.example_id,
            decision_point_id=request.decision_point_id,
            available_at_s=request.observed_until_s,
            observed_until_s=request.observed_until_s,
            scores={"eot": probability},
            emitted_at_ns=monotonic_ns(),
        )


class SmartTurnFastTurnAdapter:
    """Fixed-threshold Smart Turn wrapper for the binary FastTurn EOT subset."""

    def __init__(
        self,
        output_labels: tuple[TurnState, ...],
        *,
        threshold: float = SMART_TURN_FASTTURN_THRESHOLD,
        revision: str = SMART_TURN_REVISION,
        intra_op_threads: int = 1,
        local_files_only: bool = False,
        pointwise: SmartTurnV32Adapter | None = None,
    ) -> None:
        if output_labels != (TurnState.COMPLETE, TurnState.INCOMPLETE):
            raise ValueError("SmartTurn supports only FastTurn eot")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("SmartTurn FastTurn threshold must be in [0, 1]")
        self._threshold = threshold
        self._pointwise = pointwise or SmartTurnV32Adapter(
            revision=revision,
            intra_op_threads=intra_op_threads,
            local_files_only=local_files_only,
        )
        source = self._pointwise.descriptor
        self._descriptor = ModelDescriptor(
            name=source.name,
            revision=source.revision,
            native_sample_rate=source.native_sample_rate,
            execution_modes=frozenset({ExecutionMode.OFFLINE}),
            output_signals=frozenset({"turn_state"}),
            supported_languages=source.supported_languages,
            accepted_inputs=source.accepted_inputs,
            context_window_s=source.context_window_s,
            artifact_checksum=source.artifact_checksum,
            artifact_size_bytes=source.artifact_size_bytes,
            model_license=source.model_license,
            code_license=source.code_license,
            known_training_datasets=source.known_training_datasets,
        )

    @property
    def descriptor(self) -> ModelDescriptor:
        return self._descriptor

    def resample_audio(self, audio: AudioData) -> AudioData:
        """Resample FastTurn clips with deterministic SoXR HQ."""
        if audio.sample_rate == SMART_TURN_SAMPLE_RATE:
            return audio
        try:
            import soxr
        except ImportError as error:  # pragma: no cover - optional model dependency
            raise RuntimeError("SmartTurn FastTurn evaluation requires soxr") from error
        samples = soxr.resample(
            audio.samples,
            audio.sample_rate,
            SMART_TURN_SAMPLE_RATE,
            quality="HQ",
        )
        return AudioData(np.asarray(samples, dtype=np.float32), SMART_TURN_SAMPLE_RATE)

    def predict_clip(self, request: ClipRequest) -> StatePrediction:
        if request.user_audio.sample_rate != SMART_TURN_SAMPLE_RATE:
            raise ValueError("SmartTurn requires 16 kHz audio")
        duration_s = request.user_audio.duration_s
        prediction = self._pointwise.predict(
            DecisionRequest(
                example_id=request.example_id,
                decision_point_id="clip-end",
                target_at_s=duration_s,
                observed_until_s=duration_s,
                silence_duration_s=0.0,
                user_audio=request.user_audio,
                transcript=None,
            )
        )
        eot_probability = prediction.scores["eot"]
        predicted = (
            TurnState.COMPLETE if eot_probability >= self._threshold else TurnState.INCOMPLETE
        )
        return StatePrediction(
            example_id=request.example_id,
            predicted_label=predicted,
            scores={
                TurnState.COMPLETE.value: eot_probability,
                TurnState.INCOMPLETE.value: 1.0 - eot_probability,
            },
            model_output={
                "native_eot_probability": eot_probability,
                "threshold": self._threshold,
                "threshold_source": "predeclared-upstream-default",
                "context_selection": "last-8-seconds-left-padded",
            },
            emitted_at_ns=monotonic_ns(),
        )
