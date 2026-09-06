from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import monotonic_ns

import numpy as np

from turn_eval.core.types import (
    AudioData,
    ClipRequest,
    ExecutionMode,
    ModelDescriptor,
    StatePrediction,
    TurnState,
)

TURNSENSE_REPO_ID = "Baiji-Team/TurnSense"
TURNSENSE_REVISION = "92924ae57f61b59cc6195d1367f8c705b2ba06b4"
TURNSENSE_SOURCE_REVISION = "e1ae083a361182614f15eff5c293dadb07314ec4"
TURNSENSE_MODEL_FILE = "TenSense/model_int8.onnx"
TURNSENSE_CMVN_FILE = "TenSense/am.mvn"
TURNSENSE_MODEL_SHA256 = "9ffa8fd0bb423986065afdedcb822fc957bb0794bcb6023c94575d8d82dd8e92"
TURNSENSE_MODEL_BYTES = 52_211_266
TURNSENSE_SAMPLE_RATE = 16_000
TURNSENSE_CONTEXT_SECONDS = 8
TURNSENSE_NATIVE_LABELS = ("complete", "incomplete", "invalid")


def _normalize_turnsense_output(output: np.ndarray) -> np.ndarray:
    values = np.asarray(output, dtype=np.float64).reshape(-1)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("TurnSense must emit three finite class probabilities")
    if np.any(values < 0) or np.any(values > 1) or not np.isclose(values.sum(), 1, atol=1e-4):
        shifted = values - values.max()
        values = np.exp(shifted) / np.exp(shifted).sum()
    return values


def map_turnsense_probabilities(
    probabilities: np.ndarray,
) -> tuple[TurnState | None, dict[str, float], str]:
    values = _normalize_turnsense_output(probabilities)
    native_label = TURNSENSE_NATIVE_LABELS[int(np.argmax(values))]
    predicted = {
        "complete": TurnState.COMPLETE,
        "incomplete": TurnState.INCOMPLETE,
    }.get(native_label)
    scores = {
        TurnState.COMPLETE.value: float(values[0]),
        TurnState.INCOMPLETE.value: float(values[1]),
    }
    return predicted, scores, native_label


class _AudioFrontend:
    """Pinned implementation of the upstream Kaldi fbank/LFR/CMVN frontend."""

    def __init__(self, cmvn_path: Path) -> None:
        try:
            import kaldi_native_fbank as knf
        except ImportError as error:  # pragma: no cover - isolated environment
            raise RuntimeError("TurnSense requires kaldi-native-fbank") from error
        options = knf.FbankOptions()
        options.frame_opts.samp_freq = TURNSENSE_SAMPLE_RATE
        options.frame_opts.dither = 0.0
        options.frame_opts.window_type = "hamming"
        options.frame_opts.frame_shift_ms = 10.0
        options.frame_opts.frame_length_ms = 25.0
        options.mel_opts.num_bins = 80
        options.energy_floor = 0
        options.frame_opts.snip_edges = True
        options.mel_opts.debug_mel = False
        self._knf = knf
        self._options = options
        self._cmvn = self._load_cmvn(cmvn_path)

    @staticmethod
    def _load_cmvn(path: Path) -> np.ndarray:
        lines = path.read_text(encoding="utf-8").splitlines()
        means: list[str] = []
        variances: list[str] = []
        for index, line in enumerate(lines[:-1]):
            values = line.split()
            following = lines[index + 1].split()
            if not following or following[0] != "<LearnRateCoef>":
                continue
            if values and values[0] == "<AddShift>":
                means = following[3:-1]
            elif values and values[0] == "<Rescale>":
                variances = following[3:-1]
        if not means or len(means) != len(variances):
            raise ValueError(f"invalid TurnSense CMVN file: {path}")
        return np.asarray([means, variances], dtype=np.float64)

    @staticmethod
    def _apply_lfr(features: np.ndarray, width: int = 7, stride: int = 6) -> np.ndarray:
        output = []
        frame_count = features.shape[0]
        output_count = int(np.ceil(frame_count / stride))
        left = np.tile(features[0], ((width - 1) // 2, 1))
        padded = np.vstack((left, features))
        padded_count = frame_count + (width - 1) // 2
        for index in range(output_count):
            start = index * stride
            available = padded_count - start
            if width <= available:
                frame = padded[start : start + width]
            else:
                missing = width - available
                frame = np.vstack((padded[start:], np.tile(padded[-1], (missing, 1))))
            output.append(frame.reshape(1, -1))
        return np.vstack(output).astype(np.float32)

    def extract(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        samples = np.asarray(samples, dtype=np.float32) * (1 << 15)
        fbank = self._knf.OnlineFbank(self._options)
        fbank.accept_waveform(TURNSENSE_SAMPLE_RATE, samples.tolist())
        frame_count = fbank.num_frames_ready
        if frame_count == 0:
            raise ValueError("TurnSense requires at least one 25 ms audio frame")
        features = np.empty((frame_count, 80), dtype=np.float32)
        for index in range(frame_count):
            features[index] = fbank.get_frame(index)
        features = self._apply_lfr(features)
        dimension = features.shape[1]
        features = (features + self._cmvn[0, :dimension]) * self._cmvn[1, :dimension]
        return features[None].astype(np.float32), np.asarray([len(features)], dtype=np.int64)


class TurnSenseFastTurnAdapter:
    """TurnSense INT8 audio adapter for the binary FastTurn EOT subset."""

    def __init__(
        self,
        output_labels: tuple[TurnState, ...],
        *,
        revision: str = TURNSENSE_REVISION,
        threads: int = 4,
        local_files_only: bool = False,
        infer: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if output_labels != (TurnState.COMPLETE, TurnState.INCOMPLETE):
            raise ValueError(
                "TurnSense supports only FastTurn eot; backchannel/wait are unsupported"
            )
        if threads <= 0:
            raise ValueError("TurnSense thread count must be positive")
        self._infer = infer or self._load_runtime(revision, threads, local_files_only)
        self._descriptor = ModelDescriptor(
            name="turnsense-int8",
            revision=revision,
            native_sample_rate=TURNSENSE_SAMPLE_RATE,
            execution_modes=frozenset({ExecutionMode.OFFLINE}),
            output_signals=frozenset({"turn_state"}),
            supported_languages=frozenset({"zh"}),
            context_window_s=TURNSENSE_CONTEXT_SECONDS,
            artifact_checksum=f"sha256:{TURNSENSE_MODEL_SHA256}",
            artifact_size_bytes=TURNSENSE_MODEL_BYTES,
            model_license="Apache-2.0",
            code_license="Apache-2.0",
            known_training_datasets=("upstream training set not disclosed",),
        )

    @staticmethod
    def _load_runtime(
        revision: str, threads: int, local_files_only: bool
    ) -> Callable[[np.ndarray], np.ndarray]:
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
        except ImportError as error:  # pragma: no cover - isolated environment
            raise RuntimeError(
                "TurnSense dependencies are missing; use environments/turnsense"
            ) from error
        model_path = Path(
            hf_hub_download(
                TURNSENSE_REPO_ID,
                TURNSENSE_MODEL_FILE,
                revision=revision,
                local_files_only=local_files_only,
            )
        )
        cmvn_path = Path(
            hf_hub_download(
                TURNSENSE_REPO_ID,
                TURNSENSE_CMVN_FILE,
                revision=revision,
                local_files_only=local_files_only,
            )
        )
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = threads
        session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        frontend = _AudioFrontend(cmvn_path)

        def run(samples: np.ndarray) -> np.ndarray:
            features, lengths = frontend.extract(samples)
            return np.asarray(
                session.run(None, {"feats": features, "feat_lengths": lengths})[0][0],
                dtype=np.float64,
            )

        return run

    @property
    def descriptor(self) -> ModelDescriptor:
        return self._descriptor

    def resample_audio(self, audio: AudioData) -> AudioData:
        """Match upstream librosa's default SoXR-HQ preprocessing."""
        if audio.sample_rate == TURNSENSE_SAMPLE_RATE:
            return audio
        try:
            import soxr
        except ImportError as error:  # pragma: no cover - isolated environment
            raise RuntimeError("TurnSense requires soxr for upstream-parity resampling") from error
        samples = soxr.resample(
            audio.samples,
            audio.sample_rate,
            TURNSENSE_SAMPLE_RATE,
            quality="HQ",
        )
        return AudioData(np.asarray(samples, dtype=np.float32), TURNSENSE_SAMPLE_RATE)

    def predict_clip(self, request: ClipRequest) -> StatePrediction:
        if request.user_audio.sample_rate != TURNSENSE_SAMPLE_RATE:
            raise ValueError("TurnSense requires 16 kHz audio")
        maximum = TURNSENSE_CONTEXT_SECONDS * TURNSENSE_SAMPLE_RATE
        samples = request.user_audio.samples[-maximum:]
        output = self._infer(samples)
        probabilities = _normalize_turnsense_output(output)
        predicted, scores, native_label = map_turnsense_probabilities(probabilities)
        native_probabilities = {
            label: float(value)
            for label, value in zip(
                TURNSENSE_NATIVE_LABELS,
                probabilities,
                strict=True,
            )
        }
        return StatePrediction(
            example_id=request.example_id,
            predicted_label=predicted,
            scores=scores,
            model_output={
                "native_argmax": native_label,
                "native_probabilities": native_probabilities,
                "invalid_handling": "abstain-closed-set-incorrect",
                "input_samples": len(samples),
            },
            emitted_at_ns=monotonic_ns(),
        )
