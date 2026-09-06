from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download

from turn_eval.core.types import AudioData, EvaluationProtocol, InterruptionPredictionRecord

SID_BENCH_REPO_ID = "kxxia/SID-bench"
SID_BENCH_REVISION = "6eb13b573ad588646ce0de513d4255e55caf858b"
SID_BENCH_SCORER_REVISION = "0a9e5001e1fe534b3eec4f76c4455ae4bff9ccf1"
SID_BENCH_DATA_LICENSE = "unspecified; source-corpus terms apply"
SID_BENCH_SPLITS = frozenset({"en", "zh", "noise"})
SID_BENCH_ANNOTATION_SHA256 = {
    "en": "ceb18d41d2e014e225b3d729bd3a380514cb1fec911c9747325094138a104544",
    "zh": "b5faadcc6d1f2ddbc87b93765592f44056bcac6b60f8ce5ff952dee16b9f178f",
    "noise": "d5a73649653e40e51cd7b96cf181df71f379378aaeaec190a9cde786aa6979cf",
}
SID_BENCH_EXPECTED_CATEGORIES = {
    "en": Counter({"interruption_beginning": 500, "interruption_middle": 600, "nonbreak": 500}),
    "zh": Counter({"interruption_beginning": 500, "interruption_middle": 600, "nonbreak": 500}),
    "noise": Counter({"noise_or_silence": 500}),
}

_ANNOTATIONS = {
    "en": "en_test_lines.jsonl",
    "zh": "zh_test_lines.jsonl",
    "noise": "silence_noise_test.jsonl",
}
_AUDIO_DIRS = {"en": "en", "zh": "zh", "noise": "silence_or_noise"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_sid_bench_dataset(
    split: str,
    *,
    with_audio: bool = True,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
) -> Path:
    """Fetch a pinned SID-Bench split into the Hugging Face cache."""
    if split not in SID_BENCH_SPLITS:
        raise ValueError(f"unsupported SID-Bench split: {split}")
    patterns = [
        "README.md",
        "test.ipynb",
        f"test_wavs/{_ANNOTATIONS[split]}",
    ]
    if with_audio:
        patterns.append(f"test_wavs/{_AUDIO_DIRS[split]}/*.wav")
    return Path(
        snapshot_download(
            repo_id=SID_BENCH_REPO_ID,
            repo_type="dataset",
            revision=SID_BENCH_REVISION,
            allow_patterns=patterns,
            cache_dir=None if cache_dir is None else str(cache_dir),
            local_files_only=local_files_only,
        )
    )


@dataclass(frozen=True, slots=True)
class SidBenchItem:
    id: str
    audio_path: Path
    language: str
    category: str
    duration_s: float
    audio_duration_s: float | None
    gold_break_time_s: float | None
    transcript: str | None

    def load_audio(self) -> AudioData:
        samples, sample_rate = sf.read(self.audio_path, dtype="float32", always_2d=False)
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError(f"SID-Bench audio must be mono: {self.audio_path}")
        return AudioData(samples, int(sample_rate))


@dataclass(frozen=True, slots=True)
class SidBenchDataset:
    root: Path
    split: str
    items: tuple[SidBenchItem, ...]
    revision: str
    annotation_sha256: str

    @property
    def metadata(self) -> dict[str, object]:
        inspected = [item for item in self.items if item.audio_duration_s is not None]
        duration_deltas = [
            abs(item.audio_duration_s - item.duration_s)
            for item in inspected
            if item.audio_duration_s is not None
        ]
        sample_rates = Counter(
            sf.info(item.audio_path).samplerate for item in inspected if item.audio_path.is_file()
        )
        return {
            "repository": SID_BENCH_REPO_ID,
            "revision": self.revision,
            "split": self.split,
            "configuration": self.split,
            "annotation_checksum": f"sha256:{self.annotation_sha256}",
            "source_rows": len(self.items),
            "class_counts": dict(sorted(Counter(item.category for item in self.items).items())),
            "license": SID_BENCH_DATA_LICENSE,
            "license_note": "The public repository has no LICENSE file or license tag.",
            "grouping": "unavailable; each clip is treated as its own group",
            "audio_header_validation": {
                "inspected": len(inspected),
                "native_sample_rates": {
                    str(key): value for key, value in sorted(sample_rates.items())
                },
                "maximum_absolute_duration_delta_s": max(duration_deltas, default=None),
                "duration_deltas_over_50ms": sum(delta > 0.05 for delta in duration_deltas),
                "scoring_duration_source": "annotation",
                "runtime_duration_source": "wav_header",
            },
        }

    @property
    def benchmark_metadata(self) -> dict[str, object]:
        return {
            "name": "sid-bench",
            "revision": self.revision,
            "official_scorer_revision": SID_BENCH_SCORER_REVISION,
            "protocol": EvaluationProtocol.SINGLE_CHANNEL_INTERRUPTION.value,
            "task": "semantic_interruption",
            "language": "und" if self.split == "noise" else self.split,
            "dataset_license": SID_BENCH_DATA_LICENSE,
            "agent_audio_available": False,
            "decision_signal": "first committed interruption",
            "model_team_authored_benchmark": True,
            "possible_test_overlap": "unknown",
        }

    def prediction_records(
        self,
        predicted_break_times: Mapping[str, float | None],
        *,
        require_exact_coverage: bool = True,
    ) -> tuple[InterruptionPredictionRecord, ...]:
        expected = {item.id for item in self.items}
        supplied = set(predicted_break_times)
        if require_exact_coverage and supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise ValueError(
                f"SID-Bench prediction coverage mismatch: missing={missing[:3]}, extra={extra[:3]}"
            )
        records = []
        for item in self.items:
            if item.id not in predicted_break_times:
                continue
            records.append(
                InterruptionPredictionRecord(
                    example_id=item.id,
                    group_id=item.id,
                    language=item.language,
                    category=item.category,
                    duration_s=item.duration_s,
                    gold_break_time_s=item.gold_break_time_s,
                    predicted_break_time_s=predicted_break_times[item.id],
                )
            )
        return tuple(records)


def _category(row: dict[str, object], split: str) -> str:
    if split == "noise":
        return "noise_or_silence"
    if row["total_nonbreak"] is True:
        return "nonbreak"
    transcript = row["text_with_break"]
    if not isinstance(transcript, str) or "<break>" not in transcript:
        raise ValueError("SID-Bench interruption transcript must contain a <break> marker")
    return (
        "interruption_beginning"
        if transcript.lstrip().startswith("<break>")
        else "interruption_middle"
    )


def load_sid_bench(
    root: Path,
    split: str,
    *,
    strict_counts: bool = True,
    validate_audio_headers: bool = True,
    require_audio: bool = True,
) -> SidBenchDataset:
    if split not in SID_BENCH_SPLITS:
        raise ValueError(f"unsupported SID-Bench split: {split}")
    root = root.resolve()
    data_root = root / "test_wavs" if (root / "test_wavs").is_dir() else root
    annotations = data_root / _ANNOTATIONS[split]
    if not annotations.is_file():
        raise FileNotFoundError(f"SID-Bench annotations not found: {annotations}")
    checksum = _sha256(annotations)
    if strict_counts and checksum != SID_BENCH_ANNOTATION_SHA256[split]:
        raise ValueError(f"SID-Bench {split} annotation checksum mismatch: {checksum}")

    language = "und" if split == "noise" else split
    audio_root = data_root / _AUDIO_DIRS[split]
    items: list[SidBenchItem] = []
    seen: set[str] = set()
    with annotations.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"SID-Bench line {line_number}: invalid JSON") from error
            expected_fields = {
                "audio",
                "total_nonbreak",
                "duration",
                "break_time",
                "text_with_break",
            }
            if not isinstance(row, dict) or set(row) != expected_fields:
                raise ValueError(f"SID-Bench line {line_number}: unexpected schema")
            filename = row["audio"]
            if (
                not isinstance(filename, str)
                or not filename
                or filename in seen
                or Path(filename).name != filename
            ):
                raise ValueError(f"SID-Bench line {line_number}: invalid or duplicate audio name")
            seen.add(filename)
            if not isinstance(row["total_nonbreak"], bool):
                raise ValueError(f"SID-Bench line {line_number}: total_nonbreak must be boolean")
            duration_s = float(row["duration"])
            raw_break_time = float(row["break_time"])
            if duration_s <= 0 or raw_break_time < 0 or raw_break_time > duration_s + 1e-6:
                raise ValueError(f"SID-Bench line {line_number}: invalid annotated timing")
            if row["total_nonbreak"]:
                if split != "noise" and row["text_with_break"] is None:
                    raise ValueError(f"SID-Bench line {line_number}: missing speech transcript")
                gold_break_time_s = None
            else:
                if split == "noise":
                    raise ValueError("SID-Bench noise split cannot contain interruption labels")
                gold_break_time_s = raw_break_time
            audio_path = audio_root / filename
            if require_audio and not audio_path.is_file():
                raise ValueError(f"SID-Bench line {line_number}: audio file is missing or unsafe")
            audio_duration_s = None
            if validate_audio_headers and require_audio:
                info = sf.info(audio_path)
                if info.channels != 1 or info.format != "WAV":
                    raise ValueError(f"SID-Bench line {line_number}: audio must be mono WAV")
                audio_duration_s = float(info.duration)
                if gold_break_time_s is not None and gold_break_time_s > audio_duration_s + 1e-6:
                    raise ValueError(
                        f"SID-Bench line {line_number}: gold break exceeds the WAV duration"
                    )
            transcript = row["text_with_break"]
            items.append(
                SidBenchItem(
                    id=filename,
                    audio_path=audio_path,
                    language=language,
                    category=_category(row, split),
                    duration_s=duration_s,
                    audio_duration_s=audio_duration_s,
                    gold_break_time_s=gold_break_time_s,
                    transcript=transcript if isinstance(transcript, str) else None,
                )
            )

    counts = Counter(item.category for item in items)
    if strict_counts and counts != SID_BENCH_EXPECTED_CATEGORIES[split]:
        raise ValueError(f"SID-Bench {split} category counts mismatch: {dict(counts)}")
    return SidBenchDataset(root, split, tuple(items), SID_BENCH_REVISION, checksum)


def load_sid_bench_predictions(path: Path) -> dict[str, float | None]:
    """Load the official JSONL prediction schema without trusting its label flag."""
    predictions: dict[str, float | None] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"SID-Bench prediction line {line_number}: invalid JSON"
                ) from error
            if not isinstance(row, dict) or set(row) != {
                "audio",
                "total_nonbreak",
                "break_time",
            }:
                raise ValueError(f"SID-Bench prediction line {line_number}: unexpected schema")
            audio = row["audio"]
            nonbreak = row["total_nonbreak"]
            break_time = row["break_time"]
            if not isinstance(audio, str) or not audio or audio in predictions:
                raise ValueError(f"SID-Bench prediction line {line_number}: invalid duplicate ID")
            if not isinstance(nonbreak, bool):
                raise ValueError(
                    f"SID-Bench prediction line {line_number}: total_nonbreak must be boolean"
                )
            if isinstance(break_time, bool) or not isinstance(break_time, int | float):
                raise ValueError(f"SID-Bench prediction line {line_number}: invalid break_time")
            if nonbreak:
                if float(break_time) != -1.0:
                    raise ValueError(
                        f"SID-Bench prediction line {line_number}: nonbreak requires break_time=-1"
                    )
                predictions[audio] = None
            else:
                if float(break_time) < 0:
                    raise ValueError(
                        f"SID-Bench prediction line {line_number}: break time must be non-negative"
                    )
                predictions[audio] = float(break_time)
    return predictions
