from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from transformers import WhisperFeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from turn_eval.benchmarks.turnbench import (  # noqa: E402
    TURNBENCH_DEV_DATASET,
    TURNBENCH_DEV_REVISION,
    TurnBenchConversationProbabilities,
    TurnBenchProbabilityFile,
    TurnBenchProbabilityTask,
    TurnBenchSpeakerProbabilities,
    load_turnbench,
)
from turn_eval.models.smart_turn_timeline import (  # noqa: E402
    SMART_TURN_TIMELINE_OUTPUT_HZ,
    SMART_TURN_TIMELINE_SAMPLE_RATE,
    score_smart_turn_channel,
)

MODEL_REPOSITORY = "pipecat-ai/smart-turn-v3"
MODEL_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
MODEL_LICENSE = "BSD-2-Clause"
SMART_TURN_CODE_REVISION = "4786657e242dfe77dd138699ac564ee074a2a543"
SILERO_CODE_REVISION = "867c2aa692646a1f1de3e94a15c9dd9f614c0acb"
SILERO_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/"
    f"{SILERO_CODE_REVISION}/src/silero_vad/data/silero_vad.onnx"
)
SILERO_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _silero_path(*, local_files_only: bool) -> Path:
    import os

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        raise RuntimeError("HF_HOME must be set for SmartTurn TurnBench execution")
    destination = Path(hf_home) / "turn-eval-assets" / "smart-turn" / "silero_vad.onnx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        if local_files_only:
            raise FileNotFoundError(f"Silero VAD is not cached at {destination}")
        temporary = destination.with_suffix(".onnx.part")
        urllib.request.urlretrieve(SILERO_URL, temporary)
        if _sha256(temporary) != SILERO_SHA256:
            raise RuntimeError("downloaded Silero VAD artifact checksum mismatch")
        temporary.replace(destination)
    if _sha256(destination) != SILERO_SHA256:
        raise RuntimeError("cached Silero VAD artifact checksum mismatch")
    return destination


class SileroVAD:
    """Exact wrapper used by the upstream SmartTurn TurnBench predictor."""

    def __init__(self, model_path: Path) -> None:
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"], sess_options=options
        )
        self.context_size = 64
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, self.context_size), dtype=np.float32)
        self.last_reset_time = time.time()

    def probability(self, chunk: np.ndarray) -> float:
        audio = np.reshape(chunk, (1, -1))
        if audio.shape[1] != 512:
            raise ValueError(f"Silero VAD expected 512 samples, got {audio.shape[1]}")
        audio = np.concatenate((self.context, audio), axis=1)
        output, self.state = self.session.run(
            None,
            {
                "input": audio.astype(np.float32),
                "state": self.state,
                "sr": np.asarray(16_000, dtype=np.int64),
            },
        )
        self.context = audio[:, -self.context_size :]
        # Preserve the official implementation's wall-clock reset behavior for
        # strict baseline parity; record it explicitly in metadata.
        if time.time() - self.last_reset_time >= 5.0:
            self.state = np.zeros((2, 1, 128), dtype=np.float32)
            self.context = np.zeros((1, self.context_size), dtype=np.float32)
            self.last_reset_time = time.time()
        return float(output[0][0])


class SmartTurnEndpoint:
    def __init__(self, model_path: Path, provider: str) -> None:
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if provider == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("onnxruntime CUDAExecutionProvider is unavailable")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=providers
        )
        self.extractor = WhisperFeatureExtractor(chunk_length=8)
        self.max_samples = 8 * SMART_TURN_TIMELINE_SAMPLE_RATE
        self.lock = Lock()

    def probability(self, segment: np.ndarray) -> float:
        if len(segment) > self.max_samples:
            segment = segment[-self.max_samples :]
        elif len(segment) < self.max_samples:
            segment = np.pad(segment, (self.max_samples - len(segment), 0))
        # ORT documents sessions as callable from multiple threads, but the
        # v3.1 GPU graph produced a corrupted negative value under nested
        # TurnBench concurrency on this host. Serialize the small endpoint call
        # and fail loudly if the model contract is violated again.
        with self.lock:
            inputs = self.extractor(
                segment,
                sampling_rate=SMART_TURN_TIMELINE_SAMPLE_RATE,
                return_tensors="np",
                padding="max_length",
                max_length=self.max_samples,
                truncation=True,
                do_normalize=True,
            )
            features = np.asarray(inputs["input_features"], dtype=np.float32)
            probability = float(
                np.asarray(self.session.run(None, {"input_features": features})[0]).reshape(-1)[0]
            )
        if not 0 <= probability <= 1:
            raise RuntimeError(f"SmartTurn emitted invalid probability {probability}")
        return probability


def _resample(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == SMART_TURN_TIMELINE_SAMPLE_RATE:
        return np.asarray(samples, dtype=np.float32)
    tensor = torch.from_numpy(np.array(samples, dtype=np.float32, copy=True))
    return torchaudio.functional.resample(
        tensor, sample_rate, SMART_TURN_TIMELINE_SAMPLE_RATE
    ).numpy()


def _score_channel(
    samples: np.ndarray,
    sample_rate: int,
    endpoint: SmartTurnEndpoint,
    silero_path: Path,
) -> tuple[np.ndarray, float, float]:
    resampling_started = perf_counter()
    waveform = _resample(samples, sample_rate)
    resampling_s = perf_counter() - resampling_started
    vad = SileroVAD(silero_path)
    inference_started = perf_counter()
    scores = score_smart_turn_channel(
        waveform,
        vad_probability=vad.probability,
        endpoint_probability=endpoint.probability,
    )
    return scores, perf_counter() - inference_started, resampling_s


def _score_conversation(conversation, endpoint, silero_path):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            _score_channel,
            conversation.speaker_1_audio.samples,
            conversation.speaker_1_audio.sample_rate,
            endpoint,
            silero_path,
        )
        second = pool.submit(
            _score_channel,
            conversation.speaker_2_audio.samples,
            conversation.speaker_2_audio.sample_rate,
            endpoint,
            silero_path,
        )
        score_1, inference_1, resampling_1 = first.result()
        score_2, inference_2, resampling_2 = second.result()
    return score_1, score_2, inference_1 + inference_2, resampling_1 + resampling_2


def run(args: argparse.Namespace) -> dict[str, object]:
    dataset = load_turnbench(
        source=args.dataset,
        revision=args.revision,
        split="development",
        local_files_only=args.local_files_only,
    )
    filename = f"smart-turn-v{args.version}-{args.artifact}.onnx"
    model_path = Path(
        hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            filename=filename,
            revision=MODEL_REVISION,
            local_files_only=args.local_files_only,
        )
    )
    silero_path = _silero_path(local_files_only=args.local_files_only)
    initialization_started = perf_counter()
    endpoint = SmartTurnEndpoint(model_path, args.provider)
    initialization_s = perf_counter() - initialization_started

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    resume_dir = output.with_name(f"{output.name}.partial")
    resume_dir.mkdir(parents=True, exist_ok=True)

    raw_scores: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    inference_s = 0.0
    resampling_s = 0.0
    conversations = list(dataset.iter_conversations(limit=args.limit))
    if not conversations:
        raise RuntimeError("TurnBench dataset produced no conversations")
    pending = []
    for conversation in conversations:
        cached = resume_dir / f"{conversation.conversation_id}.npz"
        if not cached.exists():
            pending.append(conversation)
            continue
        with np.load(cached) as arrays:
            raw_scores[conversation.conversation_id] = (
                arrays["speaker_1"],
                arrays["speaker_2"],
                float(arrays["duration_s"]),
            )
            inference_s += float(arrays["inference_s"])
            resampling_s += float(arrays["resampling_s"])
        print(f"conversation={conversation.conversation_id} resumed", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_score_conversation, conversation, endpoint, silero_path): conversation
            for conversation in pending
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            conversation = futures[future]
            score_1, score_2, channel_inference_s, channel_resampling_s = future.result()
            raw_scores[conversation.conversation_id] = (
                score_1,
                score_2,
                conversation.duration_s,
            )
            inference_s += channel_inference_s
            resampling_s += channel_resampling_s
            cached = resume_dir / f"{conversation.conversation_id}.npz"
            temporary = cached.with_suffix(".npz.part")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    speaker_1=score_1,
                    speaker_2=score_2,
                    duration_s=conversation.duration_s,
                    inference_s=channel_inference_s,
                    resampling_s=channel_resampling_s,
                )
            temporary.replace(cached)
            print(
                f"conversation={conversation.conversation_id} completed="
                f"{len(raw_scores)}/{len(conversations)} new={completed}/{len(pending)} "
                f"duration_s={conversation.duration_s:.1f}",
                flush=True,
            )

    eot_entries: list[TurnBenchConversationProbabilities] = []
    int_entries: list[TurnBenchConversationProbabilities] = []
    for conversation in conversations:
        score_1, score_2, duration_s = raw_scores[conversation.conversation_id]
        frame_count = int(np.floor(duration_s * SMART_TURN_TIMELINE_OUTPUT_HZ))
        speaker_1 = tuple(float(value) for value in score_1[:frame_count])
        speaker_2 = tuple(float(value) for value in score_2[:frame_count])
        eot_entries.append(
            TurnBenchConversationProbabilities(
                conversation.conversation_id,
                TurnBenchSpeakerProbabilities(speaker_1),
                TurnBenchSpeakerProbabilities(speaker_2),
            )
        )
        int_entries.append(
            TurnBenchConversationProbabilities(
                conversation.conversation_id,
                TurnBenchSpeakerProbabilities(tuple(1.0 - value for value in speaker_1)),
                TurnBenchSpeakerProbabilities(tuple(1.0 - value for value in speaker_2)),
            )
        )

    output.mkdir(parents=True, exist_ok=False)
    TurnBenchProbabilityFile(
        task=TurnBenchProbabilityTask.EOT,
        frame_rate_hz=SMART_TURN_TIMELINE_OUTPUT_HZ,
        probs=tuple(eot_entries),
    ).write(output / "probs-eot.json")
    TurnBenchProbabilityFile(
        task=TurnBenchProbabilityTask.INTERRUPTION,
        frame_rate_hz=SMART_TURN_TIMELINE_OUTPUT_HZ,
        probs=tuple(int_entries),
    ).write(output / "probs-int.json")
    audio_s = sum(item.duration_s for item in conversations)
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": {
            "name": "turnbench",
            "dataset": dataset.source,
            "dataset_revision": dataset.revision,
            "conversation_count": len(conversations),
            "leaderboard_eligible": args.limit is None,
        },
        "model": {
            "name": f"smart-turn-v{args.version}-{args.artifact}",
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "filename": filename,
            "sha256": _sha256(model_path),
            "license": MODEL_LICENSE,
            "smart_turn_code_revision": SMART_TURN_CODE_REVISION,
            "silero_code_revision": SILERO_CODE_REVISION,
            "silero_sha256": SILERO_SHA256,
        },
        "evaluation": {
            "mode": "native-single-channel-per-speaker",
            "execution": "streaming-vad-gated-pointwise",
            "frame_rate_hz": SMART_TURN_TIMELINE_OUTPUT_HZ,
            "frame_timestamp": "frame-end",
            "resampler": f"torchaudio-{torchaudio.__version__}",
            "vad_state_reset": "official-wall-clock-5s",
            "endpoint_concurrency": "serialized-after-invalid-cuda-output",
            "eot_mapping": "p_complete",
            "interruption_mapping": "1 - p_complete",
            "signal_provenance": {"eot": "native", "interruption": "derived"},
        },
        "runtime": {
            "provider": args.provider,
            "workers": args.workers,
            "initialization_s": initialization_s,
            "inference_channel_s": inference_s,
            "resampling_channel_s": resampling_s,
            "audio_s": audio_s,
            "rtf_channel_sum": inference_s / audio_s,
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {"status": "ok", "output_dir": str(output), **metadata}


def main() -> int:
    parser = argparse.ArgumentParser(description="Pinned SmartTurn TurnBench backend")
    parser.add_argument("--version", choices=("3.1", "3.2"), required=True)
    parser.add_argument("--artifact", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dataset", default=TURNBENCH_DEV_DATASET)
    parser.add_argument("--revision", default=TURNBENCH_DEV_REVISION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    print(json.dumps(run(args), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
