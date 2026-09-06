from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

from turn_eval.core.types import (
    AudioData,
    ChatMessage,
    DecisionLabel,
    DecisionPoint,
    DecisionPointExample,
    TimedWord,
)

EOT_BENCH_REPO_ID = "livekit/eot-bench-data"
EOT_BENCH_REVISION = "ca9d98a9686b920a2d8c9eb984224ba9be74e4dd"
EOT_BENCH_SPLIT = "validation"
EOT_BENCH_DATA_LICENSE = "CC-BY-4.0"
EOT_BENCH_SCORER_REVISION = "6594d8b3b8af385b15f116dde310ce45af92d646"
SUPPORTED_CONFIGS = frozenset({"en", "zh"})
_EPS = 1e-6


@dataclass(frozen=True, slots=True)
class EotBenchDataset:
    examples: tuple[DecisionPointExample, ...]
    language: str
    split: str
    revision: str
    parquet_path: Path
    parquet_sha256: str
    total_rows: int

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "repository": EOT_BENCH_REPO_ID,
            "configuration": self.language,
            "split": self.split,
            "revision": self.revision,
            "checksum": f"sha256:{self.parquet_sha256}",
            "license": EOT_BENCH_DATA_LICENSE,
            "source_rows": self.total_rows,
            "evaluated_rows": len(self.examples),
        }

    @property
    def benchmark_metadata(self) -> dict[str, object]:
        return {
            "name": "livekit-eot-bench",
            "revision": self.revision,
            "official_scorer_revision": EOT_BENCH_SCORER_REVISION,
            "scorer_revision": "turn-eval-eot-v1",
            "protocol": "decision_point",
            "task": "eot_hold",
            "language": self.language,
            "locale": None,
            "dataset_license": EOT_BENCH_DATA_LICENSE,
            "model_team_authored_benchmark": False,
            "possible_test_overlap": "unknown",
            "primary_leaderboard_eligible": len(self.examples) == self.total_rows,
            "partial_run": len(self.examples) != self.total_rows,
        }


def _data_filename(language: str, split: str) -> str:
    return f"data/{language}/{split}-00000-of-00001.parquet"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_audio(audio: Any) -> AudioData:
    if not isinstance(audio, dict):
        raise ValueError("eot-bench audio must be an Arrow struct")
    audio_bytes = audio.get("bytes")
    audio_path = audio.get("path")
    if audio_bytes is not None:
        source: io.BytesIO | str = io.BytesIO(audio_bytes)
    elif audio_path:
        source = str(audio_path)
    else:
        raise ValueError("eot-bench audio has neither embedded bytes nor a path")

    samples, sample_rate = sf.read(source, dtype="float32", always_2d=False)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError("eot-bench audio could not be converted to mono")
    return AudioData(samples, int(sample_rate))


def _time_grid(start_s: float, end_s: float, interval_s: float) -> tuple[float, ...]:
    if interval_s <= 0:
        raise ValueError("inference interval must be positive")
    steps = max(0, int(math.floor((end_s - start_s) / interval_s + _EPS)))
    values = [round(start_s + index * interval_s, 6) for index in range(steps + 1)]
    if values and values[-1] < end_s - _EPS:
        values.append(round(end_s, 6))
    elif not values:
        values = [round(start_s, 6), round(end_s, 6)]
    return tuple(values)


def _conversation_id(example_id: str) -> str:
    marker = "__user_turn_"
    return example_id.rsplit(marker, 1)[0] if marker in example_id else example_id


def eot_bench_row_to_example(
    row: dict[str, Any],
    *,
    inference_interval_s: float = 0.1,
    min_silence_span_s: float = 0.1,
) -> DecisionPointExample:
    if min_silence_span_s <= 0:
        raise ValueError("minimum silence span must be positive")
    example_id = str(row["id"])
    language = str(row["language"]).strip().lower()
    if language not in SUPPORTED_CONFIGS:
        raise ValueError(f"unsupported eot-bench language: {language}")
    audio = _decode_audio(row["audio"])

    spans = list(row["silence_spans"] or [])
    if not spans:
        raise ValueError(f"eot-bench row {example_id} has no silence spans")
    points: list[DecisionPoint] = []
    for span_index, span in enumerate(spans):
        start_s = float(span["start"])
        end_s = float(span["end"])
        if end_s - start_s < min_silence_span_s - _EPS:
            continue
        label = DecisionLabel.EOT if span_index == len(spans) - 1 else DecisionLabel.HOLD
        span_id = f"{example_id}:span:{span_index}"
        for point_index, timestamp_s in enumerate(_time_grid(start_s, end_s, inference_interval_s)):
            silence_us = round((timestamp_s - start_s) * 1_000_000)
            points.append(
                DecisionPoint(
                    id=f"{span_id}:point:{point_index}:at:{silence_us}us",
                    span_id=span_id,
                    timestamp_s=timestamp_s,
                    silence_start_s=start_s,
                    silence_end_s=end_s,
                    label=label,
                )
            )
    if not points:
        raise ValueError(f"eot-bench row {example_id} has no eligible decision points")

    words = tuple(
        TimedWord(
            text=str(word["word"]).strip(),
            start_s=float(word["start"]),
            end_s=float(word["end"]),
        )
        for word in (row.get("words") or [])
        if str(word["word"]).strip()
    )
    messages = tuple(
        ChatMessage(role=str(message["role"]).strip(), content=str(message["content"]).strip())
        for message in (row.get("messages") or [])
        if str(message["role"]).strip() and str(message["content"]).strip()
    )
    conversation_id = _conversation_id(example_id)
    return DecisionPointExample(
        id=example_id,
        conversation_id=conversation_id,
        group_id=conversation_id,
        language=language,
        user_audio=audio,
        decision_points=tuple(points),
        words=words,
        messages=messages,
    )


def load_eot_bench(
    language: str,
    *,
    revision: str = EOT_BENCH_REVISION,
    split: str = EOT_BENCH_SPLIT,
    limit: int | None = None,
    inference_interval_s: float = 0.1,
    min_silence_span_s: float = 0.1,
    local_files_only: bool = False,
) -> EotBenchDataset:
    language = language.strip().lower()
    if language not in SUPPORTED_CONFIGS:
        raise ValueError(
            f"eot-bench supports these configs: {', '.join(sorted(SUPPORTED_CONFIGS))}"
        )
    if split != EOT_BENCH_SPLIT:
        raise ValueError(f"eot-bench public data only provides split={EOT_BENCH_SPLIT!r}")
    if limit is not None and limit <= 0:
        raise ValueError("dataset limit must be positive")

    downloaded = hf_hub_download(
        repo_id=EOT_BENCH_REPO_ID,
        repo_type="dataset",
        filename=_data_filename(language, split),
        revision=revision,
        local_files_only=local_files_only,
    )
    parquet_path = Path(downloaded)
    parquet = pq.ParquetFile(parquet_path)
    total_rows = parquet.metadata.num_rows
    examples: list[DecisionPointExample] = []
    columns = ("id", "audio", "language", "silence_spans", "words", "messages")
    for batch in parquet.iter_batches(batch_size=8, columns=columns):
        for row in batch.to_pylist():
            examples.append(
                eot_bench_row_to_example(
                    row,
                    inference_interval_s=inference_interval_s,
                    min_silence_span_s=min_silence_span_s,
                )
            )
            if limit is not None and len(examples) >= limit:
                break
        if limit is not None and len(examples) >= limit:
            break

    return EotBenchDataset(
        examples=tuple(examples),
        language=language,
        split=split,
        revision=revision,
        parquet_path=parquet_path,
        parquet_sha256=_sha256_file(parquet_path),
        total_rows=total_rows,
    )
