from __future__ import annotations

import argparse
import hashlib
import json
import queue
import sys
import types
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from prepare import MAAI_VERSION, MAAI_WHEEL_SHA256, prepare

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
from turn_eval.core.audio import StreamingLinearResampler  # noqa: E402

MAAI_CODE_REVISION = "0.2.18"
MODEL_REPOSITORY = "maai-kyoto/vap_tri"
MODEL_REVISION = "1a7721ad12a26ac8cd6025bec5543f9c005b9f51"
MODEL_FILENAME = "vap_state_dict_tri_10hz_5000msec.pt"
MODEL_LICENSE = "CC-BY-NC-SA-4.0"
FRAME_RATE_HZ = 10.0
CONTEXT_WINDOW_S = 5.0
NATIVE_SAMPLE_RATE = 16_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_maai():
    vendor, cpc = prepare()
    sys.path.insert(0, str(vendor))
    sys.modules.setdefault("pyaudio", types.ModuleType("pyaudio"))
    sys.modules.setdefault("pygame", types.ModuleType("pygame"))
    from maai import Maai

    return Maai, cpc


class _NullInput:
    def subscribe(self):
        return queue.Queue()


def _build_model(device: str, *, local_files_only: bool = False):
    torch.manual_seed(0)
    np.random.seed(0)
    Maai, cpc_path = _load_maai()
    checkpoint = Path(
        hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
            local_files_only=local_files_only,
        )
    )
    model = Maai(
        mode="vap",
        lang="tri",
        audio_ch1=_NullInput(),
        audio_ch2=_NullInput(),
        frame_rate=FRAME_RATE_HZ,
        context_len_sec=int(CONTEXT_WINDOW_S),
        device=device,
        cpc_model=str(cpc_path),
        local_model=str(checkpoint),
        model_type="normal",
        use_kv_cache=True,
    )
    return model, checkpoint, cpc_path


def _run_frames(
    model, channel_1: np.ndarray, channel_2: np.ndarray
) -> tuple[list[float], list[float]]:
    if channel_1.shape != channel_2.shape or channel_1.ndim != 1:
        raise ValueError("Realtime-VAP input channels must be aligned mono arrays")
    samples_per_frame = round(NATIVE_SAMPLE_RATE / FRAME_RATE_HZ)
    frame_count = len(channel_1) // samples_per_frame
    model.reset_runtime_state()
    model.result_dict_queue.queue.clear()
    p_speaker_1: list[float] = []
    p_speaker_2: list[float] = []
    for frame_index in range(frame_count):
        start = frame_index * samples_per_frame
        end = start + samples_per_frame
        model.process(channel_1[start:end], channel_2[start:end])
        try:
            result = model.result_dict_queue.get_nowait()
        except queue.Empty as error:
            raise RuntimeError(f"Realtime-VAP emitted no result for frame {frame_index}") from error
        p_now = result["p_now"]
        if len(p_now) != 2:
            raise RuntimeError("Realtime-VAP p_now must contain two speaker probabilities")
        p_speaker_1.append(float(p_now[0]))
        p_speaker_2.append(float(p_now[1]))
    return p_speaker_1, p_speaker_2


def _resample(samples: np.ndarray, input_rate: int) -> np.ndarray:
    if input_rate == NATIVE_SAMPLE_RATE:
        return np.asarray(samples, dtype=np.float32)
    return StreamingLinearResampler(input_rate, NATIVE_SAMPLE_RATE).push(samples)


def _run_turnbench(args: argparse.Namespace) -> dict[str, object]:
    dataset = load_turnbench(
        source=args.dataset,
        revision=args.revision,
        split="development",
        local_files_only=args.local_files_only,
    )
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must satisfy 0 <= index < count")
    selected_shards = dataset.parquet_shards[args.shard_index :: args.shard_count]
    if not selected_shards:
        raise ValueError("selected TurnBench shard contains no Parquet files")
    dataset = replace(dataset, parquet_shards=selected_shards)
    initialization_started = perf_counter()
    model, checkpoint, cpc_path = _build_model(args.device, local_files_only=args.local_files_only)
    initialization_s = perf_counter() - initialization_started

    eot_entries: list[TurnBenchConversationProbabilities] = []
    int_entries: list[TurnBenchConversationProbabilities] = []
    durations: dict[str, float] = {}
    inference_s = 0.0
    resampling_s = 0.0
    audio_s = 0.0
    for index, conversation in enumerate(dataset.iter_conversations(limit=args.limit), start=1):
        resampling_started = perf_counter()
        channel_1 = _resample(
            conversation.speaker_1_audio.samples,
            conversation.speaker_1_audio.sample_rate,
        )
        channel_2 = _resample(
            conversation.speaker_2_audio.samples,
            conversation.speaker_2_audio.sample_rate,
        )
        resampling_s += perf_counter() - resampling_started
        inference_started = perf_counter()
        p1, p2 = _run_frames(model, channel_1, channel_2)
        inference_s += perf_counter() - inference_started

        expected_frames = int(np.floor(conversation.duration_s * FRAME_RATE_HZ))
        if len(p1) != expected_frames or len(p2) != expected_frames:
            raise RuntimeError(
                f"conversation {conversation.conversation_id}: emitted {len(p1)}/{len(p2)} "
                f"frames, expected {expected_frames}"
            )
        p1_tuple = tuple(p1)
        p2_tuple = tuple(p2)
        eot_entries.append(
            TurnBenchConversationProbabilities(
                conversation_id=conversation.conversation_id,
                speaker_1=TurnBenchSpeakerProbabilities(tuple(1.0 - value for value in p1)),
                speaker_2=TurnBenchSpeakerProbabilities(tuple(1.0 - value for value in p2)),
            )
        )
        int_entries.append(
            TurnBenchConversationProbabilities(
                conversation_id=conversation.conversation_id,
                speaker_1=TurnBenchSpeakerProbabilities(p1_tuple),
                speaker_2=TurnBenchSpeakerProbabilities(p2_tuple),
            )
        )
        durations[conversation.conversation_id] = conversation.duration_s
        audio_s += conversation.duration_s
        print(
            f"conversation={conversation.conversation_id} completed={index} "
            f"duration_s={conversation.duration_s:.1f}",
            flush=True,
        )

    if not eot_entries:
        raise RuntimeError("TurnBench execution produced no conversations")
    eot_probabilities = TurnBenchProbabilityFile(
        task=TurnBenchProbabilityTask.EOT,
        frame_rate_hz=FRAME_RATE_HZ,
        probs=tuple(eot_entries),
    )
    int_probabilities = TurnBenchProbabilityFile(
        task=TurnBenchProbabilityTask.INTERRUPTION,
        frame_rate_hz=FRAME_RATE_HZ,
        probs=tuple(int_entries),
    )
    eot_probabilities.validate(durations)
    int_probabilities.validate(durations)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    eot_probabilities.write(output / "probs-eot.json")
    int_probabilities.write(output / "probs-int.json")
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": {
            "name": "turnbench",
            "dataset": dataset.source,
            "dataset_revision": dataset.revision,
            "conversation_count": len(eot_entries),
            "leaderboard_eligible": args.limit is None and args.shard_count == 1,
            "sharding": {
                "index": args.shard_index,
                "count": args.shard_count,
                "files": [path.name for path in selected_shards],
                "limited": args.limit is not None,
            },
        },
        "model": {
            "name": "maai-realtime-vap-trilingual-cpc-10hz-5s",
            "maai_version": MAAI_VERSION,
            "maai_wheel_sha256": MAAI_WHEEL_SHA256,
            "checkpoint_repository": MODEL_REPOSITORY,
            "checkpoint_revision": MODEL_REVISION,
            "checkpoint_filename": MODEL_FILENAME,
            "checkpoint_sha256": _sha256(checkpoint),
            "cpc_sha256": _sha256(cpc_path),
            "license": MODEL_LICENSE,
            "code_license": "MIT",
            "supported_languages": ["en", "zh", "ja"],
        },
        "evaluation": {
            "mode": "native-dual-channel",
            "execution": "streaming",
            "frame_rate_hz": FRAME_RATE_HZ,
            "frame_timestamp": "frame_end",
            "context_window_s": CONTEXT_WINDOW_S,
            "native_sample_rate": NATIVE_SAMPLE_RATE,
            "source_resampler": "causal-linear-streaming",
            "eot_mapping": "1 - p_now[speaker]",
            "interruption_mapping": "p_now[speaker]",
            "signal_provenance": {"eot": "derived", "interruption": "derived"},
        },
        "runtime": {
            "device": args.device,
            "initialization_s": initialization_s,
            "inference_s": inference_s,
            "resampling_s": resampling_s,
            "audio_s": audio_s,
            "rtf": inference_s / audio_s if audio_s else None,
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {"status": "ok", "output_dir": str(output), **metadata}


def _self_test(device: str) -> dict[str, object]:
    started = perf_counter()
    model, checkpoint, cpc_path = _build_model(device)
    initialization_s = perf_counter() - started
    time_axis = np.arange(NATIVE_SAMPLE_RATE // 2, dtype=np.float32) / NATIVE_SAMPLE_RATE
    channel_1 = (0.05 * np.sin(2 * np.pi * 220 * time_axis)).astype(np.float32)
    channel_2 = np.zeros_like(channel_1)
    inference_started = perf_counter()
    p1, p2 = _run_frames(model, channel_1, channel_2)
    inference_s = perf_counter() - inference_started
    if len(p1) != 5 or len(p2) != 5:
        raise RuntimeError("Realtime-VAP self-test produced the wrong frame count")
    if any(not 0 <= probability <= 1 for probability in (*p1, *p2)):
        raise RuntimeError("Realtime-VAP self-test produced an invalid probability")
    return {
        "status": "ok",
        "device": device,
        "maai_version": MAAI_VERSION,
        "maai_wheel_sha256": MAAI_WHEEL_SHA256,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "model_filename": MODEL_FILENAME,
        "model_sha256": _sha256(checkpoint),
        "cpc_sha256": _sha256(cpc_path),
        "frame_count": len(p1),
        "initialization_s": initialization_s,
        "inference_s": inference_s,
        "p_now_speaker_1": p1,
        "p_now_speaker_2": p2,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Pinned multilingual Realtime-VAP backend")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dataset", default=TURNBENCH_DEV_DATASET)
    parser.add_argument("--revision", default=TURNBENCH_DEV_REVISION)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = _self_test(args.device)
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required for TurnBench execution")
        result = _run_turnbench(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
