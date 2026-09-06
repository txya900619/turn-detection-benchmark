from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from prepare import VAP_REPOSITORY, VAP_REVISION, prepare

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

MODEL_REPOSITORY = "viks66/VAP_checkpoints"
MODEL_REVISION = "b9aa0ba1718221153e04ef343c7f3b8cf84bfbc3"
MODEL_FILENAME = "oto.ckpt"
MODEL_LICENSE = "CC-BY-4.0"
CODE_LICENSE = "MIT"
SAMPLE_RATE = 16_000
FRAME_RATE_HZ = 50.0
CONTEXT_TIME_S = 20.0
STEP_TIME_S = 5.0
STEP_EXTRACTION_LIMIT_S = 160.0
CPC_SHA256 = "d0f474de968c7243c20f43dee1626681618f2f150d3e638e82b79e22563b17f1"
OTO_SHA256 = "8e73c37537e2b18ca1552d4b733bd2bda9dfe0a4e6885a0b400e4a6589db265c"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(device: str, *, local_files_only: bool):
    source = prepare(local_files_only=local_files_only)
    sys.path.insert(0, str(source))
    from vap.model import VapConfig, VapGPT

    checkpoint = Path(
        hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
            local_files_only=local_files_only,
        )
    )
    cpc_checkpoint = source / "assets" / "checkpoints" / "cpc" / "60k_epoch4-d0f474de.pt"
    if _sha256(checkpoint) != OTO_SHA256:
        raise RuntimeError("VapGPT OTO checkpoint checksum mismatch")
    if _sha256(cpc_checkpoint) != CPC_SHA256:
        raise RuntimeError("VapGPT CPC checkpoint checksum mismatch")

    model = VapGPT(VapConfig())
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = {
            key.replace("net.", ""): value
            for key, value in state["state_dict"].items()
            if "VAP.codebook" not in key
        }
    model.load_state_dict(state)
    return model.to(device).eval(), checkpoint, cpc_checkpoint


def _append_tail(output: dict[str, torch.Tensor], update: dict[str, torch.Tensor], frames: int):
    output["p_now"] = torch.cat((output["p_now"], update["p_now"][:, -frames:]), dim=1)
    output["p_future"] = torch.cat((output["p_future"], update["p_future"][:, -frames:]), dim=1)


def _step_extraction(waveform: torch.Tensor, model, device: str) -> dict[str, torch.Tensor]:
    """Run the exact bounded-context windowing policy from TurnBench's VAP baseline."""
    step_samples = int(STEP_TIME_S * model.sample_rate)
    chunk_samples = int((CONTEXT_TIME_S + STEP_TIME_S) * model.sample_rate)
    step_frames = int(STEP_TIME_S * model.frame_hz)
    expected_frames = round(waveform.shape[-1] / model.sample_rate * model.frame_hz)
    folds = waveform.unfold(-1, chunk_samples, step_samples).permute(2, 0, 1, 3)
    if not len(folds):
        raise RuntimeError("VapGPT long-audio extraction produced no windows")

    with torch.no_grad():
        output = model.probs(folds[0].to(device))
        for window in folds[1:]:
            update = model.probs(window.to(device))
            _append_tail(output, update, step_frames)
        processed_frames = output["p_now"].shape[1]
        if processed_frames < expected_frames:
            omitted_frames = expected_frames - processed_frames
            update = model.probs(waveform[..., -chunk_samples:].to(device))
            _append_tail(output, update, omitted_frames)
    return {key: value.cpu() for key, value in output.items()}


def _load_audio(samples: np.ndarray, sample_rate: int) -> torch.Tensor:
    tensor = torch.from_numpy(np.array(samples, dtype=np.float32, copy=True))
    if sample_rate != SAMPLE_RATE:
        tensor = torchaudio.functional.resample(tensor, sample_rate, SAMPLE_RATE)
    return tensor


def _infer_conversation(conversation, model, device: str):
    resampling_started = perf_counter()
    speaker_1 = _load_audio(
        conversation.speaker_1_audio.samples,
        conversation.speaker_1_audio.sample_rate,
    )
    speaker_2 = _load_audio(
        conversation.speaker_2_audio.samples,
        conversation.speaker_2_audio.sample_rate,
    )
    resampling_s = perf_counter() - resampling_started
    waveform = torch.stack((speaker_1, speaker_2)).unsqueeze(0).to(device)
    inference_started = perf_counter()
    if waveform.shape[-1] / SAMPLE_RATE > STEP_EXTRACTION_LIMIT_S:
        output = _step_extraction(waveform, model, device)
    else:
        with torch.no_grad():
            output = {key: value.cpu() for key, value in model.probs(waveform).items()}
    inference_s = perf_counter() - inference_started
    probabilities = output["p_now"][0].numpy()
    return probabilities[:, 0], probabilities[:, 1], inference_s, resampling_s


def run(args: argparse.Namespace) -> dict[str, object]:
    dataset = load_turnbench(
        source=args.dataset,
        revision=args.revision,
        split="development",
        local_files_only=args.local_files_only,
    )
    initialization_started = perf_counter()
    model, checkpoint, cpc_checkpoint = _load_model(
        args.device, local_files_only=args.local_files_only
    )
    initialization_s = perf_counter() - initialization_started

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    resume_dir = output.with_name(f"{output.name}.partial")
    resume_dir.mkdir(parents=True, exist_ok=True)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must satisfy 0 <= index < count")
    selected_shards = dataset.parquet_shards[args.shard_index :: args.shard_count]
    if not selected_shards:
        raise ValueError("selected TurnBench shard contains no Parquet files")
    dataset = replace(dataset, parquet_shards=selected_shards)
    selected_ids = dataset.conversation_ids()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    expected_count = min(args.limit or len(selected_ids), len(selected_ids))
    if not expected_count:
        raise RuntimeError("TurnBench dataset produced no conversations")

    raw_scores: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    conversation_rows: list[tuple[str, float]] = []
    inference_s = 0.0
    resampling_s = 0.0
    for index, conversation in enumerate(dataset.iter_conversations(limit=args.limit), start=1):
        conversation_rows.append((conversation.conversation_id, conversation.duration_s))
        cached = resume_dir / f"{conversation.conversation_id}.npz"
        if cached.exists():
            with np.load(cached) as arrays:
                raw_scores[conversation.conversation_id] = (
                    arrays["speaker_1"],
                    arrays["speaker_2"],
                    float(arrays["duration_s"]),
                )
                inference_s += float(arrays["inference_s"])
                resampling_s += float(arrays["resampling_s"])
            print(f"conversation={conversation.conversation_id} resumed={index}", flush=True)
            continue

        p1, p2, conversation_inference_s, conversation_resampling_s = _infer_conversation(
            conversation, model, args.device
        )
        raw_scores[conversation.conversation_id] = (p1, p2, conversation.duration_s)
        inference_s += conversation_inference_s
        resampling_s += conversation_resampling_s
        temporary = cached.with_suffix(".npz.part")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                speaker_1=p1,
                speaker_2=p2,
                duration_s=conversation.duration_s,
                inference_s=conversation_inference_s,
                resampling_s=conversation_resampling_s,
            )
        temporary.replace(cached)
        print(
            f"conversation={conversation.conversation_id} completed={index}/{expected_count} "
            f"duration_s={conversation.duration_s:.1f} inference_s={conversation_inference_s:.1f}",
            flush=True,
        )

    eot_entries: list[TurnBenchConversationProbabilities] = []
    int_entries: list[TurnBenchConversationProbabilities] = []
    durations: dict[str, float] = {}
    for conversation_id, duration_s in conversation_rows:
        p1, p2, duration_s = raw_scores[conversation_id]
        frame_count = int(np.floor(duration_s * FRAME_RATE_HZ))
        p1_values = tuple(float(value) for value in p1[:frame_count])
        p2_values = tuple(float(value) for value in p2[:frame_count])
        if len(p1_values) != frame_count or len(p2_values) != frame_count:
            raise RuntimeError(f"conversation {conversation_id}: VapGPT emitted too few frames")
        eot_entries.append(
            TurnBenchConversationProbabilities(
                conversation_id,
                TurnBenchSpeakerProbabilities(
                    tuple(float(value) for value in np.float32(1.0) - p1[:frame_count])
                ),
                TurnBenchSpeakerProbabilities(
                    tuple(float(value) for value in np.float32(1.0) - p2[:frame_count])
                ),
            )
        )
        int_entries.append(
            TurnBenchConversationProbabilities(
                conversation_id,
                TurnBenchSpeakerProbabilities(p1_values),
                TurnBenchSpeakerProbabilities(p2_values),
            )
        )
        durations[conversation_id] = duration_s

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
    output.mkdir(parents=True, exist_ok=False)
    eot_probabilities.write(output / "probs-eot.json")
    int_probabilities.write(output / "probs-int.json")

    audio_s = sum(duration_s for _, duration_s in conversation_rows)
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": {
            "name": "turnbench",
            "dataset": dataset.source,
            "dataset_revision": dataset.revision,
            "conversation_count": len(conversation_rows),
            "leaderboard_eligible": args.limit is None and args.shard_count == 1,
            "sharding": {
                "index": args.shard_index,
                "count": args.shard_count,
                "limited": args.limit is not None,
                "files": [path.name for path in selected_shards],
            },
        },
        "model": {
            "name": "vap-gpt-oto",
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "filename": MODEL_FILENAME,
            "sha256": _sha256(checkpoint),
            "license": MODEL_LICENSE,
            "code_repository": VAP_REPOSITORY,
            "code_revision": VAP_REVISION,
            "code_license": CODE_LICENSE,
            "cpc_sha256": _sha256(cpc_checkpoint),
        },
        "evaluation": {
            "mode": "native-dual-channel",
            "execution": "bounded-context-overlapping-windows",
            "frame_rate_hz": FRAME_RATE_HZ,
            "frame_timestamp": "frame-end",
            "native_sample_rate": SAMPLE_RATE,
            "context_time_s": CONTEXT_TIME_S,
            "step_time_s": STEP_TIME_S,
            "resampler": f"torchaudio-{torchaudio.__version__}",
            "eot_mapping": "1 - p_now[speaker]",
            "interruption_mapping": "p_now[speaker]",
            "signal_provenance": {"eot": "derived", "interruption": "native-floor-state"},
        },
        "runtime": {
            "device": args.device,
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
    parser = argparse.ArgumentParser(description="Pinned official VapGPT TurnBench backend")
    parser.add_argument("--dataset", default=TURNBENCH_DEV_DATASET)
    parser.add_argument("--revision", default=TURNBENCH_DEV_REVISION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
