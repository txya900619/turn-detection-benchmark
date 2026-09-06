from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import monotonic_ns
from typing import Any

import numpy as np

from turn_eval.core.types import (
    ClipRequest,
    ExecutionMode,
    ModelDescriptor,
    StatePrediction,
    TurnState,
)

X2_TURN_REPO_ID = "x-square-robot/X2-Turn-4B-0812"
X2_TURN_REVISION = "d188a8b86058ece5e293124a278a1a6256af81fa"
X2_TURN_SOURCE_REVISION = "3609a0fd11b4a1dc2b2629e95538bbe0860da96d"
X2_TURN_WEIGHT_SHA256 = "75c7fd12babb6d7ffebf6d0cb86388ff5e20b9a1e9ed079c45eaed62ef6a9117"
X2_TURN_WEIGHT_BYTES = 10_470_070_544
X2_TURN_SAMPLE_RATE = 16_000
X2_TURN_FRAME_MS = 80
X2_TURN_DEFAULT_DELAY_MS = 480
X2_TURN_FRONTEND_LOOKAHEAD_MS = 2.5
X2_TURN_RAW_LABELS = (
    "idle",
    "noidle",
    "speaking",
    "turn_end",
    "backchannel",
    "uncertain",
)
X2_TURN_TARGET_MAP = {
    "turn_end": TurnState.COMPLETE,
    "speaking": TurnState.INCOMPLETE,
    "backchannel": TurnState.BACKCHANNEL,
}


def last_non_idle_frame(frames: Sequence[Any]) -> Any | None:
    """Apply the paper's utterance reduction without inventing a Wait mapping."""
    return next((frame for frame in reversed(frames) if frame.label != "idle"), None)


def map_x2_frame(
    frame: Any | None,
    output_labels: tuple[TurnState, ...],
) -> tuple[TurnState | None, dict[str, float]]:
    """Map one raw X2 frame exactly; unsupported states become abstentions."""
    probabilities: Mapping[str, float] = {} if frame is None else frame.probabilities
    scores = {
        TurnState.COMPLETE.value: float(probabilities.get("turn_end", 0.0)),
        TurnState.INCOMPLETE.value: float(probabilities.get("speaking", 0.0)),
    }
    if TurnState.BACKCHANNEL in output_labels:
        scores[TurnState.BACKCHANNEL.value] = float(probabilities.get("backchannel", 0.0))
    raw_label = None if frame is None else frame.label
    predicted = X2_TURN_TARGET_MAP.get(raw_label)
    if predicted not in output_labels:
        predicted = None
    return predicted, scores


class X2TurnFastTurnAdapter:
    """Pinned X2-Turn offline two-pass adapter with an exact-only state mapping."""

    def __init__(
        self,
        output_labels: tuple[TurnState, ...],
        *,
        revision: str = X2_TURN_REVISION,
        device: str = "cuda:0",
        delay_ms: int = X2_TURN_DEFAULT_DELAY_MS,
        local_files_only: bool = False,
        model_path: str | Path = X2_TURN_REPO_ID,
        infer: Callable[[np.ndarray], Any] | None = None,
    ) -> None:
        supported_vocabularies = {
            (TurnState.COMPLETE, TurnState.INCOMPLETE),
            (TurnState.COMPLETE, TurnState.INCOMPLETE, TurnState.BACKCHANNEL),
        }
        if output_labels not in supported_vocabularies:
            raise ValueError("X2-Turn supports only FastTurn eot or real; Wait has no native class")
        if delay_ms <= 0 or delay_ms % X2_TURN_FRAME_MS:
            raise ValueError("X2-Turn target delay must be a positive multiple of 80 ms")

        self.output_labels = output_labels
        self.delay_ms = delay_ms
        self.device = device
        self._infer = infer or self._load_runtime(
            model_path=model_path,
            revision=revision,
            device=device,
            delay_ms=delay_ms,
            local_files_only=local_files_only,
        )
        self._descriptor = ModelDescriptor(
            name="x2-turn-4b-0812",
            revision=revision,
            native_sample_rate=X2_TURN_SAMPLE_RATE,
            execution_modes=frozenset({ExecutionMode.CAUSAL_OFFLINE}),
            output_signals=frozenset({"turn_state"}),
            supported_languages=frozenset({"en", "zh"}),
            frame_interval_ms=X2_TURN_FRAME_MS,
            algorithmic_lookahead_ms=X2_TURN_FRONTEND_LOOKAHEAD_MS,
            artifact_checksum=f"sha256:{X2_TURN_WEIGHT_SHA256}",
            artifact_size_bytes=X2_TURN_WEIGHT_BYTES,
            model_license="Apache-2.0",
            code_license="Apache-2.0",
            known_training_datasets=("EasyTurn subset", "Fisher subset"),
        )

    @staticmethod
    def _load_runtime(
        *,
        model_path: str | Path,
        revision: str,
        device: str,
        delay_ms: int,
        local_files_only: bool,
    ) -> Callable[[np.ndarray], Any]:
        try:
            import torch
            from mistral_common.tokens.tokenizers.audio import Audio
            from transformers import AutoProcessor
            from voxtral_realtime.transformers import infer_asr_turn, load_mtp_checkpoint
        except ImportError as error:  # pragma: no cover - exercised in the isolated project
            raise RuntimeError(
                "X2-Turn dependencies are missing; use environments/x2_turn"
            ) from error

        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        processor = AutoProcessor.from_pretrained(
            model_path,
            revision=revision,
            local_files_only=local_files_only,
        )
        model = load_mtp_checkpoint(
            model_path,
            revision=revision,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
        ).eval()

        def run(samples: np.ndarray) -> Any:
            audio = Audio(
                audio_array=np.asarray(samples, dtype=np.float32),
                sampling_rate=X2_TURN_SAMPLE_RATE,
                format="wav",
            )
            return infer_asr_turn(model, processor, audio, delay_ms=delay_ms)

        return run

    @property
    def descriptor(self) -> ModelDescriptor:
        return self._descriptor

    def predict_clip(self, request: ClipRequest) -> StatePrediction:
        if request.user_audio.sample_rate != X2_TURN_SAMPLE_RATE:
            raise ValueError("X2-Turn requires 16 kHz audio")
        result = self._infer(request.user_audio.samples)
        frame = last_non_idle_frame(result.turn_frames)
        predicted, scores = map_x2_frame(frame, self.output_labels)
        raw_frames = [
            {
                "index": int(item.index),
                "start_ms": int(item.start_ms),
                "end_ms": int(item.end_ms),
                "label": str(item.label),
                "confidence": float(item.confidence),
                "probabilities": {
                    name: float(item.probabilities[name]) for name in X2_TURN_RAW_LABELS
                },
            }
            for item in result.turn_frames
        ]
        return StatePrediction(
            example_id=request.example_id,
            predicted_label=predicted,
            scores=scores,
            model_output={
                "transcript": result.transcript,
                "frame_ms": int(result.frame_ms),
                "generated_token_ids": [int(item) for item in result.generated_token_ids],
                "selected_frame_index": None if frame is None else int(frame.index),
                "raw_argmax": None if frame is None else str(frame.label),
                "turn_frames": raw_frames,
            },
            emitted_at_ns=monotonic_ns(),
        )
