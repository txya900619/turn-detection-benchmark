from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HOME

from turn_eval.core.types import AudioData, ClipClassificationExample, TurnState

FASTTURN_REPO_ID = "ASLP-lab/FastTurn-Testset"
FASTTURN_REVISION = "4194458681149e2947cdb062ad61292cf9705a59"
FASTTURN_SOURCE_REVISION = "0c1c03dc03c9cc0bf7a743a066fc99107c8a5c59"
FASTTURN_ARCHIVE = "fasttrun-testset.tar.gz"
FASTTURN_ARCHIVE_SHA256 = "888cf6bfcc76df7a034060bbb11a7a1c2053c63901ab5e7f5adca48da14c2795"
FASTTURN_JSONL_SHA256 = "3e0825586eb7ed36c262e3bb6feb3c9eb2d93b350fb874aad8e1e91c3723ded2"
FASTTURN_EXPECTED_COUNTS = {
    TurnState.COMPLETE: 14_709,
    TurnState.INCOMPLETE: 3_643,
    TurnState.BACKCHANNEL: 3_080,
    TurnState.WAIT: 1_000,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_extracted_root(root: Path) -> None:
    manifest = root / "data.jsonl"
    if not manifest.is_file():
        raise ValueError(f"FastTurn extraction is missing {manifest}")
    actual = _sha256(manifest)
    if actual != FASTTURN_JSONL_SHA256:
        raise ValueError(f"FastTurn data.jsonl checksum mismatch: {actual}")


def prepare_fastturn_dataset(
    *,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
) -> Path:
    """Download and byte-preservingly extract the pinned FastTurn archive."""
    destination = (
        cache_dir
        if cache_dir is not None
        else Path(HF_HOME) / "turn-eval-datasets" / "fastturn" / FASTTURN_REVISION
    ).resolve()
    if destination.exists():
        _validate_extracted_root(destination)
        return destination

    archive = Path(
        hf_hub_download(
            repo_id=FASTTURN_REPO_ID,
            repo_type="dataset",
            filename=FASTTURN_ARCHIVE,
            revision=FASTTURN_REVISION,
            local_files_only=local_files_only,
        )
    )
    actual_archive_hash = _sha256(archive)
    if actual_archive_hash != FASTTURN_ARCHIVE_SHA256:
        raise ValueError(f"FastTurn archive checksum mismatch: {actual_archive_hash}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fastturn-extract-", dir=destination.parent) as staging:
        staging_path = Path(staging)
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            if any(not (member.isfile() or member.isdir()) for member in members):
                raise ValueError("FastTurn archive contains a non-regular entry")
            bundle.extractall(staging_path, members=members, filter="data")
        extracted = staging_path / "fasttrun-testset"
        _validate_extracted_root(extracted)
        try:
            os.replace(extracted, destination)
        except FileExistsError:
            _validate_extracted_root(destination)
    return destination


@dataclass(frozen=True, slots=True)
class FastTurnItem:
    key: str
    audio_path: Path
    raw_transcript: str
    transcript_without_target: str
    label: TurnState
    source_kind: str
    native_sample_rate: int
    duration_s: float


@dataclass(frozen=True, slots=True)
class FastTurnDataset:
    root: Path
    items: tuple[FastTurnItem, ...]
    revision: str
    jsonl_sha256: str

    @property
    def metadata(self) -> dict[str, object]:
        counts = Counter(item.label.value for item in self.items)
        return {
            "repository": FASTTURN_REPO_ID,
            "revision": self.revision,
            "split": "test",
            "configuration": "zh",
            "jsonl_checksum": f"sha256:{self.jsonl_sha256}",
            "source_rows": len(self.items),
            "class_counts": dict(sorted(counts.items())),
            "license": "ambiguous: HF card Apache-2.0; upstream source provenance CC-BY-NC-4.0",
            "grouping": "unavailable; each clip key is unique",
        }

    @property
    def benchmark_metadata(self) -> dict[str, object]:
        return {
            "name": "fastturn-testset",
            "revision": self.revision,
            "source_revision": FASTTURN_SOURCE_REVISION,
            "protocol": "clip_classification",
            "language": "zh",
            "tasks": ["fastturn-real-3state", "fastturn-full-4state", "fastturn-eot-2state"],
            "model_team_authored_benchmark": True,
            "possible_test_overlap": "unknown",
            "official_scorer": None,
        }

    def iter_examples(
        self,
        *,
        labels: Iterable[TurnState] | None = None,
        limit: int | None = None,
        include_transcript: bool = False,
        strip_inline_annotations: bool = True,
    ) -> Iterable[ClipClassificationExample]:
        if limit is not None and limit <= 0:
            raise ValueError("dataset limit must be positive")
        selected = set(labels) if labels is not None else None
        emitted = 0
        for item in self.items:
            if selected is not None and item.label not in selected:
                continue
            samples, sample_rate = sf.read(item.audio_path, dtype="float32", always_2d=False)
            samples = np.asarray(samples, dtype=np.float32)
            if samples.ndim != 1:
                raise ValueError(f"FastTurn audio must be mono: {item.audio_path}")
            if sample_rate != item.native_sample_rate:
                raise ValueError(f"FastTurn audio header changed: {item.audio_path}")
            transcript = None
            if include_transcript:
                transcript = item.transcript_without_target
                if strip_inline_annotations:
                    transcript = re.sub(r"<[^<>]+>", "", transcript).strip()
                if not transcript:
                    raise ValueError(
                        f"FastTurn transcript is empty after annotation stripping: {item.key}"
                    )
            yield ClipClassificationExample(
                id=item.key,
                group_id=item.key,
                language="zh",
                user_audio=AudioData(samples, int(sample_rate)),
                label=item.label,
                transcript=transcript,
                metadata={
                    "source_kind": item.source_kind,
                    "native_sample_rate": str(item.native_sample_rate),
                },
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def _clean_transcript(raw: str, label: TurnState, line_number: int) -> str:
    marker = f"<{label.value}>"
    if label is TurnState.WAIT:
        cleaned = raw
        if raw.endswith(marker):
            raise ValueError(
                f"FastTurn line {line_number}: Wait unexpectedly contains label marker"
            )
    else:
        if not raw.endswith(marker):
            raise ValueError(f"FastTurn line {line_number}: transcript label marker mismatch")
        cleaned = raw[: -len(marker)]
    cleaned = cleaned.strip()
    if not cleaned:
        raise ValueError(f"FastTurn line {line_number}: empty transcript after marker removal")
    return cleaned


def load_fastturn(
    root: Path,
    *,
    strict_counts: bool = True,
    validate_audio_headers: bool = True,
) -> FastTurnDataset:
    root = root.resolve()
    manifest = root / "data.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"FastTurn manifest not found: {manifest}")

    items: list[FastTurnItem] = []
    seen: set[str] = set()
    with manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"FastTurn line {line_number}: invalid JSON") from error
            if not isinstance(row, dict) or set(row) != {"key", "wav", "txt", "state", "lang"}:
                raise ValueError(f"FastTurn line {line_number}: unexpected schema")
            key = row["key"]
            if not isinstance(key, str) or not key or key in seen:
                raise ValueError(f"FastTurn line {line_number}: invalid or duplicate key")
            seen.add(key)
            if row["lang"] != "zh":
                raise ValueError(f"FastTurn line {line_number}: language must be zh")
            try:
                label = TurnState(row["state"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"FastTurn line {line_number}: unknown state") from error
            relative_audio = Path(row["wav"])
            if relative_audio != Path("wavs") / f"{key}.wav":
                raise ValueError(f"FastTurn line {line_number}: audio path/key mismatch")
            audio_path = (root / relative_audio).resolve()
            if not audio_path.is_relative_to(root) or not audio_path.is_file():
                raise ValueError(f"FastTurn line {line_number}: audio file is missing or unsafe")
            raw_transcript = row["txt"]
            if not isinstance(raw_transcript, str):
                raise ValueError(f"FastTurn line {line_number}: txt must be a string")
            expected_rate = 22_050 if label is TurnState.WAIT else 48_000
            if validate_audio_headers:
                info = sf.info(audio_path)
                if (
                    info.channels != 1
                    or info.format != "WAV"
                    or info.subtype != "PCM_16"
                    or info.samplerate != expected_rate
                ):
                    raise ValueError(f"FastTurn line {line_number}: unexpected WAV encoding")
                native_sample_rate = int(info.samplerate)
                duration_s = float(info.duration)
            else:
                native_sample_rate = expected_rate
                duration_s = 0.0
            items.append(
                FastTurnItem(
                    key=key,
                    audio_path=audio_path,
                    raw_transcript=raw_transcript,
                    transcript_without_target=_clean_transcript(raw_transcript, label, line_number),
                    label=label,
                    source_kind="synthetic" if label is TurnState.WAIT else "real",
                    native_sample_rate=native_sample_rate,
                    duration_s=duration_s,
                )
            )

    counts = Counter(item.label for item in items)
    if strict_counts and counts != Counter(FASTTURN_EXPECTED_COUNTS):
        raise ValueError(f"FastTurn class counts mismatch: {dict(counts)}")
    manifest_hash = _sha256(manifest)
    if strict_counts and manifest_hash != FASTTURN_JSONL_SHA256:
        raise ValueError(f"FastTurn data.jsonl checksum mismatch: {manifest_hash}")
    return FastTurnDataset(
        root=root,
        items=tuple(items),
        revision=FASTTURN_REVISION,
        jsonl_sha256=manifest_hash,
    )
