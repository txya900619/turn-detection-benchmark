from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import librosa
import numpy as np
import torch
from huggingface_hub import snapshot_download

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

MODEL_REPOSITORY = "espnet/Turn_taking_prediction_SWBD"
MODEL_REVISION = "766378e6119f254270f0dd4a4be13480ec6b78fc"
MODEL_SUBDIRECTORY = "exp/asr_train_asr_whisper_turn_taking_raw_en_word"
MODEL_FILENAME = "valid.loss.ave.pth"
MODEL_LICENSE = "CC-BY-4.0"
CODE_REPOSITORY = "https://github.com/espnet/espnet.git"
CODE_REVISION = "750e3749fc37a09187fe0fc6fb278ccb007181e8"
CODE_LICENSE = "Apache-2.0"
WHISPER_REPOSITORY = "https://github.com/espnet/whisper.git"
WHISPER_REVISION = "8a65df1760894b91857c9904410cca1123fa1bb0"

SAMPLE_RATE = 16_000
FRAME_RATE_HZ = 25.0
START_CHUNK = 3_200
HOP = 640
WINDOW = 480_000
DEFAULT_BATCH_SIZE = 8
ATTRIBUTION_FRAMES = 12
PREROLL_FRAMES = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint(*, local_files_only: bool) -> Path:
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_REPOSITORY,
            revision=MODEL_REVISION,
            allow_patterns=[f"{MODEL_SUBDIRECTORY}/*"],
            local_files_only=local_files_only,
        )
    )
    experiment = snapshot / MODEL_SUBDIRECTORY
    if not (experiment / "config.yaml").is_file():
        raise FileNotFoundError(f"missing ESPnet config under {experiment}")
    if not (experiment / MODEL_FILENAME).is_file():
        raise FileNotFoundError(f"missing ESPnet checkpoint under {experiment}")
    return experiment


def _load_model(device: str, *, local_files_only: bool, tf32: bool):
    from espnet2.tasks.slu import SLUTask

    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    experiment = _checkpoint(local_files_only=local_files_only)
    model, _ = SLUTask.build_model_from_file(
        str(experiment / "config.yaml"),
        str(experiment / MODEL_FILENAME),
        device,
    )
    return model.eval(), experiment


def _to_16khz(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    signal = np.asarray(samples, dtype=np.float32)
    if signal.ndim != 1:
        raise ValueError("ESPnet TurnTaking input must be mono")
    if sample_rate != SAMPLE_RATE:
        signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return np.asarray(signal, dtype=np.float32)


def _probs_for_signal(
    signal: np.ndarray,
    model,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Exact official growing/sliding-window inference: [T, C/NA/I/BC/T]."""
    with torch.no_grad():
        speech = torch.from_numpy(signal).to(device)
        frame_count = (len(signal) - START_CHUNK) // HOP
        if frame_count <= 0:
            return np.zeros((0, 5), dtype=np.float32)
        output = np.empty((frame_count, 5), dtype=np.float32)

        def heads(encoded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            features = model.transform_mean(model.act_fn(encoded))
            last = torch.stack(
                [features[index, lengths[index] - 1] for index in range(features.shape[0])]
            )
            return torch.softmax(model.transform_linear(last), dim=-1)

        index = 0
        while index < frame_count:
            end = (index + 1) * HOP + START_CHUNK
            if end - max(0, end - WINDOW) >= WINDOW:
                break
            window = speech[max(0, end - WINDOW) : end].unsqueeze(0)
            lengths = torch.tensor([window.shape[1]], device=device)
            encoded, encoded_lengths = model.encode(window, lengths)
            output[index] = heads(encoded, encoded_lengths)[0].cpu().numpy()
            index += 1

        while index < frame_count:
            indices = list(range(index, min(index + batch_size, frame_count)))
            windows = [
                speech[(item + 1) * HOP + START_CHUNK - WINDOW : (item + 1) * HOP + START_CHUNK]
                for item in indices
            ]
            lengths = torch.full((len(indices),), WINDOW, dtype=torch.long, device=device)
            encoded, encoded_lengths = model.encode(torch.stack(windows), lengths)
            output[indices[0] : indices[-1] + 1] = heads(encoded, encoded_lengths).cpu().numpy()
            index = indices[-1] + 1
    return output


def _frame_energy(signal: np.ndarray, frame_count: int) -> np.ndarray:
    energy = np.zeros(frame_count, dtype=np.float32)
    for index in range(frame_count):
        start = START_CHUNK + index * HOP
        segment = signal[start : start + HOP]
        if len(segment):
            energy[index] = np.sqrt(np.mean(segment.astype(np.float64) ** 2) + 1e-12)
    return energy


def _smooth_previous(values: np.ndarray, frames: int) -> np.ndarray:
    smoothed = np.zeros_like(values)
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    for index in range(len(values)):
        lower = max(0, index - frames)
        smoothed[index] = (cumulative[index] - cumulative[lower]) / max(1, index - lower)
    return smoothed


def _attribute(probabilities: np.ndarray, energy_1: np.ndarray, energy_2: np.ndarray):
    interruption = probabilities[:, 2]
    turn_change = probabilities[:, 4]
    total = energy_1 + energy_2 + 1e-8
    activity_1 = energy_1 / total
    activity_2 = energy_2 / total
    voiced = (energy_1 + energy_2) > 0.25 * np.median(energy_1 + energy_2 + 1e-8)
    activity_1 = activity_1 * voiced
    activity_2 = activity_2 * voiced
    hold_1 = _smooth_previous(activity_1, ATTRIBUTION_FRAMES)
    hold_2 = _smooth_previous(activity_2, ATTRIBUTION_FRAMES)
    onset_1 = np.clip(activity_1 - _smooth_previous(activity_1, ATTRIBUTION_FRAMES), 0, 1) * hold_2
    onset_2 = np.clip(activity_2 - _smooth_previous(activity_2, ATTRIBUTION_FRAMES), 0, 1) * hold_1
    return (
        np.asarray(turn_change * hold_1, dtype=np.float32),
        np.asarray(turn_change * hold_2, dtype=np.float32),
        np.asarray(interruption * onset_1, dtype=np.float32),
        np.asarray(interruption * onset_2, dtype=np.float32),
    )


def _canonical(values: np.ndarray, duration_s: float) -> np.ndarray:
    target = math.floor(duration_s * FRAME_RATE_HZ)
    result = np.concatenate(
        (np.zeros(PREROLL_FRAMES, dtype=np.float32), np.asarray(values, dtype=np.float32))
    )
    if len(result) < target:
        result = np.pad(result, (0, target - len(result)))
    return np.asarray(result[:target], dtype=np.float32)


def _infer_conversation(conversation, model, device: str, batch_size: int):
    resampling_started = perf_counter()
    speaker_1 = _to_16khz(
        conversation.speaker_1_audio.samples,
        conversation.speaker_1_audio.sample_rate,
    )
    speaker_2 = _to_16khz(
        conversation.speaker_2_audio.samples,
        conversation.speaker_2_audio.sample_rate,
    )
    sample_count = min(len(speaker_1), len(speaker_2))
    speaker_1 = speaker_1[:sample_count]
    speaker_2 = speaker_2[:sample_count]
    resampling_s = perf_counter() - resampling_started

    inference_started = perf_counter()
    probabilities = _probs_for_signal(speaker_1 + speaker_2, model, device, batch_size)
    inference_s = perf_counter() - inference_started
    energy_1 = _frame_energy(speaker_1, len(probabilities))
    energy_2 = _frame_energy(speaker_2, len(probabilities))
    return probabilities, energy_1, energy_2, inference_s, resampling_s


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must satisfy 0 <= index < count")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")

    dataset = load_turnbench(
        source=args.dataset,
        revision=args.revision,
        split="development",
        local_files_only=args.local_files_only,
    )
    selected_shards = dataset.parquet_shards[args.shard_index :: args.shard_count]
    if not selected_shards:
        raise ValueError("selected TurnBench shard contains no Parquet files")
    dataset = replace(dataset, parquet_shards=selected_shards)
    expected = min(args.limit or len(dataset.conversation_ids()), len(dataset.conversation_ids()))

    initialization_started = perf_counter()
    model, experiment = _load_model(
        args.device,
        local_files_only=args.local_files_only,
        tf32=args.tf32,
    )
    initialization_s = perf_counter() - initialization_started

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    resume_dir = output.with_name(f"{output.name}.partial")
    resume_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    inference_s = 0.0
    resampling_s = 0.0
    for index, conversation in enumerate(dataset.iter_conversations(limit=args.limit), start=1):
        cached = resume_dir / f"{conversation.conversation_id}.npz"
        if cached.exists():
            with np.load(cached) as arrays:
                probabilities = arrays["probabilities"]
                energy_1 = arrays["energy_1"]
                energy_2 = arrays["energy_2"]
                item_inference_s = float(arrays["inference_s"])
                item_resampling_s = float(arrays["resampling_s"])
            print(f"conversation={conversation.conversation_id} resumed={index}", flush=True)
        else:
            probabilities, energy_1, energy_2, item_inference_s, item_resampling_s = (
                _infer_conversation(conversation, model, args.device, args.batch_size)
            )
            temporary = cached.with_suffix(".npz.part")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    probabilities=probabilities,
                    energy_1=energy_1,
                    energy_2=energy_2,
                    duration_s=conversation.duration_s,
                    inference_s=item_inference_s,
                    resampling_s=item_resampling_s,
                )
            temporary.replace(cached)
            print(
                f"conversation={conversation.conversation_id} completed={index}/{expected} "
                f"duration_s={conversation.duration_s:.1f} inference_s={item_inference_s:.1f}",
                flush=True,
            )
        inference_s += item_inference_s
        resampling_s += item_resampling_s
        eot_1, eot_2, int_1, int_2 = _attribute(probabilities, energy_1, energy_2)
        rows.append(
            (
                conversation.conversation_id,
                conversation.duration_s,
                _canonical(eot_1, conversation.duration_s),
                _canonical(eot_2, conversation.duration_s),
                _canonical(int_1, conversation.duration_s),
                _canonical(int_2, conversation.duration_s),
            )
        )

    if not rows:
        raise RuntimeError("TurnBench execution produced no conversations")
    durations = {conversation_id: duration_s for conversation_id, duration_s, *_ in rows}

    def artifact(task: TurnBenchProbabilityTask, offsets: tuple[int, int]):
        return TurnBenchProbabilityFile(
            task=task,
            frame_rate_hz=FRAME_RATE_HZ,
            probs=tuple(
                TurnBenchConversationProbabilities(
                    conversation_id,
                    TurnBenchSpeakerProbabilities(tuple(float(value) for value in row[offsets[0]])),
                    TurnBenchSpeakerProbabilities(tuple(float(value) for value in row[offsets[1]])),
                )
                for row in rows
                for conversation_id in [row[0]]
            ),
        )

    eot_probabilities = artifact(TurnBenchProbabilityTask.EOT, (2, 3))
    int_probabilities = artifact(TurnBenchProbabilityTask.INTERRUPTION, (4, 5))
    eot_probabilities.validate(durations)
    int_probabilities.validate(durations)

    output.mkdir(parents=True, exist_ok=False)
    eot_probabilities.write(output / "probs-eot.json")
    int_probabilities.write(output / "probs-int.json")
    audio_s = sum(row[1] for row in rows)
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": {
            "name": "turnbench",
            "dataset": dataset.source,
            "dataset_revision": dataset.revision,
            "conversation_count": len(rows),
            "leaderboard_eligible": args.limit is None and args.shard_count == 1,
            "sharding": {
                "index": args.shard_index,
                "count": args.shard_count,
                "limited": args.limit is not None,
                "files": [path.name for path in selected_shards],
            },
        },
        "model": {
            "name": "espnet-turntaking-swbd-mixed-mono",
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "filename": f"{MODEL_SUBDIRECTORY}/{MODEL_FILENAME}",
            "sha256": _sha256(experiment / MODEL_FILENAME),
            "license": MODEL_LICENSE,
            "code_repository": CODE_REPOSITORY,
            "code_revision": CODE_REVISION,
            "code_license": CODE_LICENSE,
            "whisper_repository": WHISPER_REPOSITORY,
            "whisper_revision": WHISPER_REVISION,
            "backbone": "whisper-medium",
        },
        "evaluation": {
            "mode": "mixed-mono-single-stream-derived-speaker-attribution",
            "execution": "causal-prefix-then-30s-sliding-window",
            "frame_rate_hz": FRAME_RATE_HZ,
            "frame_timestamp": "frame-end",
            "first_prediction_s": (START_CHUNK + HOP) / SAMPLE_RATE,
            "native_sample_rate": SAMPLE_RATE,
            "context_window_s": WINDOW / SAMPLE_RATE,
            "algorithmic_lookahead_ms": 0,
            "resampler": f"librosa-{librosa.__version__}",
            "input_mapping": "speaker_1 + speaker_2",
            "eot_mapping": "P(turn-change) * causal floor-holder energy share",
            "interruption_mapping": "P(interruption) * causal barge-in onset energy share",
            "signal_provenance": {
                "eot": "native-class-probability-derived-speaker-attribution",
                "interruption": "native-class-probability-derived-speaker-attribution",
            },
        },
        "runtime": {
            "device": args.device,
            "tf32": args.tf32,
            "batch_size": args.batch_size,
            "initialization_s": initialization_s,
            "inference_s": inference_s,
            "resampling_s": resampling_s,
            "audio_s": audio_s,
            "rtf": inference_s / audio_s,
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {"status": "ok", "output_dir": str(output), **metadata}


def main() -> int:
    parser = argparse.ArgumentParser(description="Pinned ESPnet TurnTaking TurnBench backend")
    parser.add_argument("--dataset", default=TURNBENCH_DEV_DATASET)
    parser.add_argument("--revision", default=TURNBENCH_DEV_REVISION)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("XDG_CACHE_HOME", os.environ.get("HF_HOME", str(Path.home() / ".cache")))
    print(json.dumps(run(args), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
