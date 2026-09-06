from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import snapshot_download

from turn_eval.core.types import AudioData

TURNBENCH_SCORER_REVISION = "e02d547803fd0e45221afcaeb1cd24b61ae7568e"
TURNBENCH_DEV_DATASET = "mundo-ai/turn-benchmark-dev"
TURNBENCH_DEV_REVISION = "8fa18a24be51528a45397b35cbcaecd84202062b"
TURNBENCH_TEST_DATASET = "mundo-ai/turn-benchmark-test"
TURNBENCH_SUBMISSION_SCHEMA_VERSION = 1
TURNBENCH_SAMPLE_RATE = 48_000
TURNBENCH_REQUIRED_AUDIO_COLUMNS = frozenset(
    {"conversation_id", "speaker_1_audio", "speaker_2_audio"}
)


class TurnBenchTask(StrEnum):
    EOT = "eot"
    INTERRUPTION = "interruption"


class TurnBenchProbabilityTask(StrEnum):
    EOT = "eot"
    INTERRUPTION = "int"


@dataclass(frozen=True, slots=True)
class TurnBenchConversation:
    conversation_id: str
    speaker_1_audio: AudioData
    speaker_2_audio: AudioData
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.conversation_id:
            raise ValueError("conversation_id must not be empty")
        if self.speaker_1_audio.sample_rate != self.speaker_2_audio.sample_rate:
            raise ValueError("TurnBench speaker channels must have equal sample rates")
        if self.speaker_1_audio.sample_rate != TURNBENCH_SAMPLE_RATE:
            raise ValueError(f"TurnBench source audio must be {TURNBENCH_SAMPLE_RATE} Hz")
        if len(self.speaker_1_audio.samples) != len(self.speaker_2_audio.samples):
            raise ValueError("TurnBench speaker channels must be sample-aligned")

    @property
    def duration_s(self) -> float:
        return self.speaker_1_audio.duration_s


@dataclass(frozen=True, slots=True)
class TurnBenchDataset:
    source: str
    revision: str
    parquet_shards: tuple[Path, ...]
    split: str

    def __post_init__(self) -> None:
        if not self.source or not self.revision or not self.parquet_shards:
            raise ValueError("TurnBench dataset source, revision, and shards are required")
        for shard in self.parquet_shards:
            columns = set(pq.ParquetFile(shard).schema_arrow.names)
            if missing := TURNBENCH_REQUIRED_AUDIO_COLUMNS - columns:
                raise ValueError(f"TurnBench shard {shard} is missing columns: {sorted(missing)}")

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "repository": self.source,
            "configuration": "turnbench",
            "split": self.split,
            "revision": self.revision,
            "license": "Dataset Public License v1.0 (non-commercial)",
            "redistribution": "restricted",
        }

    @property
    def benchmark_metadata(self) -> dict[str, object]:
        return {
            "name": "turnbench",
            "revision": TURNBENCH_SCORER_REVISION,
            "official_scorer_revision": TURNBENCH_SCORER_REVISION,
            "protocol": "timeline",
            "tasks": [TurnBenchTask.EOT, TurnBenchTask.INTERRUPTION],
            "language": "en",
            "locale": "en-US",
            "dataset_license": "Dataset Public License v1.0 (non-commercial)",
            "model_team_authored_benchmark": False,
            "possible_test_overlap": "unknown",
        }

    def conversation_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        for shard in self.parquet_shards:
            table = pq.read_table(shard, columns=["conversation_id"])
            ids.extend(str(item) for item in table.column("conversation_id").to_pylist())
        if len(ids) != len(set(ids)):
            raise ValueError("TurnBench conversation IDs must be unique across shards")
        return tuple(ids)

    def iter_conversations(self, *, limit: int | None = None) -> Iterable[TurnBenchConversation]:
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        emitted = 0
        for shard in self.parquet_shards:
            parquet_file = pq.ParquetFile(shard)
            columns = list(TURNBENCH_REQUIRED_AUDIO_COLUMNS)
            if "metadata" in parquet_file.schema_arrow.names:
                columns.append("metadata")
            for batch in parquet_file.iter_batches(batch_size=1, columns=columns):
                row = batch.to_pylist()[0]
                yield _conversation_from_row(row)
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def _decode_audio_cell(value: object, context: str) -> AudioData:
    encoded: bytes | None = None
    if isinstance(value, bytes):
        encoded = value
    elif isinstance(value, Mapping):
        candidate = value.get("bytes")
        if isinstance(candidate, bytes):
            encoded = candidate
    if encoded is None:
        raise ValueError(f"{context} must contain embedded audio bytes")
    samples, sample_rate = sf.read(BytesIO(encoded), dtype="float32", always_2d=False)
    if samples.ndim != 1:
        raise ValueError(f"{context} must decode to mono audio")
    return AudioData(samples, int(sample_rate))


def _conversation_from_row(row: Mapping[str, object]) -> TurnBenchConversation:
    conversation_id = row.get("conversation_id")
    if not isinstance(conversation_id, str) or not conversation_id:
        raise ValueError("TurnBench conversation_id must be a non-empty string")
    metadata = row.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise ValueError(f"conversation {conversation_id} metadata must be an object")
    return TurnBenchConversation(
        conversation_id=conversation_id,
        speaker_1_audio=_decode_audio_cell(
            row.get("speaker_1_audio"), f"conversation {conversation_id} speaker_1_audio"
        ),
        speaker_2_audio=_decode_audio_cell(
            row.get("speaker_2_audio"), f"conversation {conversation_id} speaker_2_audio"
        ),
        metadata=dict(metadata),
    )


def load_turnbench(
    *,
    source: str | Path = TURNBENCH_DEV_DATASET,
    revision: str | None = None,
    split: str = "development",
    local_files_only: bool = False,
) -> TurnBenchDataset:
    """Resolve TurnBench Parquet without decoding or permanently transforming audio."""
    local_source = Path(source)
    if local_source.is_dir():
        shards = tuple(sorted(local_source.glob("*.parquet")))
        resolved_source = str(local_source.resolve())
        resolved_revision = revision or "local-unversioned"
    else:
        requested_revision = (
            TURNBENCH_DEV_REVISION
            if revision is None and str(source) == TURNBENCH_DEV_DATASET
            else revision
        )
        snapshot = Path(
            snapshot_download(
                repo_id=str(source),
                repo_type="dataset",
                revision=requested_revision,
                allow_patterns="*.parquet",
                local_files_only=local_files_only,
            )
        )
        shards = tuple(sorted(snapshot.rglob("*.parquet")))
        resolved_source = str(source)
        resolved_revision = snapshot.name
    if not shards:
        raise FileNotFoundError(f"no TurnBench Parquet shards found for {source}")
    return TurnBenchDataset(
        source=resolved_source,
        revision=resolved_revision,
        parquet_shards=shards,
        split=split,
    )


def _require_exact_keys(value: Mapping[str, object], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{context} keys differ; missing={missing}, extra={extra}")


def _as_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} keys must be strings")
    return value


def _event_times(value: object, context: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    times: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ValueError(f"{context} values must be numbers")
        time_s = float(item)
        if not isfinite(time_s) or time_s < 0:
            raise ValueError(f"{context} values must be finite and non-negative")
        times.append(time_s)
    if any(left >= right for left, right in zip(times, times[1:], strict=False)):
        raise ValueError(f"{context} values must be strictly increasing")
    return tuple(times)


@dataclass(frozen=True, slots=True)
class TurnBenchSpeakerEvents:
    eot: tuple[float, ...] = ()
    interruption: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for task, times in (
            (TurnBenchTask.EOT, self.eot),
            (TurnBenchTask.INTERRUPTION, self.interruption),
        ):
            if any(not isfinite(time_s) or time_s < 0 for time_s in times):
                raise ValueError(f"{task} event times must be finite and non-negative")
            if any(left >= right for left, right in zip(times, times[1:], strict=False)):
                raise ValueError(f"{task} event times must be strictly increasing")

    @classmethod
    def from_dict(cls, value: object, context: str) -> TurnBenchSpeakerEvents:
        mapping = _as_mapping(value, context)
        _require_exact_keys(mapping, {"eot", "interruption"}, context)
        return cls(
            eot=_event_times(mapping["eot"], f"{context}.eot"),
            interruption=_event_times(mapping["interruption"], f"{context}.interruption"),
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {"eot": list(self.eot), "interruption": list(self.interruption)}


@dataclass(frozen=True, slots=True)
class TurnBenchConversationPrediction:
    conversation_id: str
    speaker_1: TurnBenchSpeakerEvents
    speaker_2: TurnBenchSpeakerEvents

    def __post_init__(self) -> None:
        if not self.conversation_id:
            raise ValueError("conversation_id must not be empty")

    @classmethod
    def from_dict(cls, value: object, index: int) -> TurnBenchConversationPrediction:
        context = f"predictions[{index}]"
        mapping = _as_mapping(value, context)
        _require_exact_keys(mapping, {"conversation_id", "speaker_1", "speaker_2"}, context)
        conversation_id = mapping["conversation_id"]
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError(f"{context}.conversation_id must be a non-empty string")
        return cls(
            conversation_id=conversation_id,
            speaker_1=TurnBenchSpeakerEvents.from_dict(
                mapping["speaker_1"], f"{context}.speaker_1"
            ),
            speaker_2=TurnBenchSpeakerEvents.from_dict(
                mapping["speaker_2"], f"{context}.speaker_2"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "conversation_id": self.conversation_id,
            "speaker_1": self.speaker_1.to_dict(),
            "speaker_2": self.speaker_2.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TurnBenchCommittedEvent:
    conversation_id: str
    speaker: int
    task: TurnBenchTask
    commit_time_s: float
    observed_until_s: float

    def __post_init__(self) -> None:
        if not self.conversation_id:
            raise ValueError("conversation_id must not be empty")
        if self.speaker not in {1, 2}:
            raise ValueError("speaker must be 1 or 2")
        if not isfinite(self.commit_time_s) or self.commit_time_s < 0:
            raise ValueError("commit_time_s must be finite and non-negative")
        if not isfinite(self.observed_until_s) or self.observed_until_s < 0:
            raise ValueError("observed_until_s must be finite and non-negative")
        if self.observed_until_s > self.commit_time_s + 1e-9:
            raise ValueError("a committed event cannot depend on future audio")


@dataclass(frozen=True, slots=True)
class TurnBenchSpeakerProbabilities:
    prob: tuple[float, ...]

    def __post_init__(self) -> None:
        if any(not isfinite(value) or not 0 <= value <= 1 for value in self.prob):
            raise ValueError("TurnBench probabilities must be finite and in [0, 1]")

    @classmethod
    def from_dict(cls, value: object, context: str) -> TurnBenchSpeakerProbabilities:
        mapping = _as_mapping(value, context)
        _require_exact_keys(mapping, {"prob"}, context)
        raw_prob = mapping["prob"]
        if not isinstance(raw_prob, list):
            raise ValueError(f"{context}.prob must be a list")
        probabilities: list[float] = []
        for item in raw_prob:
            if isinstance(item, bool) or not isinstance(item, int | float):
                raise ValueError(f"{context}.prob values must be numbers")
            probabilities.append(float(item))
        return cls(tuple(probabilities))

    def to_dict(self) -> dict[str, list[float]]:
        return {"prob": list(self.prob)}


@dataclass(frozen=True, slots=True)
class TurnBenchConversationProbabilities:
    conversation_id: str
    speaker_1: TurnBenchSpeakerProbabilities
    speaker_2: TurnBenchSpeakerProbabilities

    def __post_init__(self) -> None:
        if not self.conversation_id:
            raise ValueError("conversation_id must not be empty")

    @classmethod
    def from_dict(cls, value: object, index: int) -> TurnBenchConversationProbabilities:
        context = f"probs[{index}]"
        mapping = _as_mapping(value, context)
        _require_exact_keys(mapping, {"conversation_id", "speaker_1", "speaker_2"}, context)
        conversation_id = mapping["conversation_id"]
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError(f"{context}.conversation_id must be a non-empty string")
        return cls(
            conversation_id=conversation_id,
            speaker_1=TurnBenchSpeakerProbabilities.from_dict(
                mapping["speaker_1"], f"{context}.speaker_1"
            ),
            speaker_2=TurnBenchSpeakerProbabilities.from_dict(
                mapping["speaker_2"], f"{context}.speaker_2"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "conversation_id": self.conversation_id,
            "speaker_1": self.speaker_1.to_dict(),
            "speaker_2": self.speaker_2.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TurnBenchProbabilityFile:
    task: TurnBenchProbabilityTask
    frame_rate_hz: float
    probs: tuple[TurnBenchConversationProbabilities, ...]
    schema_version: int = TURNBENCH_SUBMISSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TURNBENCH_SUBMISSION_SCHEMA_VERSION:
            raise ValueError("TurnBench probability schema_version must be 1")
        if not isfinite(self.frame_rate_hz) or self.frame_rate_hz <= 0:
            raise ValueError("frame_rate_hz must be finite and positive")
        conversation_ids = [entry.conversation_id for entry in self.probs]
        if len(conversation_ids) != len(set(conversation_ids)):
            raise ValueError("TurnBench probability conversation IDs must be unique")

    @classmethod
    def from_dict(cls, value: object) -> TurnBenchProbabilityFile:
        mapping = _as_mapping(value, "probability file")
        _require_exact_keys(
            mapping,
            {"schema_version", "task", "frame_rate_hz", "probs"},
            "probability file",
        )
        schema_version = mapping["schema_version"]
        if isinstance(schema_version, bool) or schema_version != 1:
            raise ValueError("probability schema_version must be integer 1")
        raw_task = mapping["task"]
        try:
            task = TurnBenchProbabilityTask(raw_task)
        except (TypeError, ValueError) as error:
            raise ValueError("probability task must be 'eot' or 'int'") from error
        raw_rate = mapping["frame_rate_hz"]
        if isinstance(raw_rate, bool) or not isinstance(raw_rate, int | float):
            raise ValueError("frame_rate_hz must be numeric")
        raw_probs = mapping["probs"]
        if not isinstance(raw_probs, list):
            raise ValueError("probs must be a list")
        return cls(
            task=task,
            frame_rate_hz=float(raw_rate),
            probs=tuple(
                TurnBenchConversationProbabilities.from_dict(item, index)
                for index, item in enumerate(raw_probs)
            ),
        )

    @classmethod
    def load(cls, path: Path) -> TurnBenchProbabilityFile:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid TurnBench probability JSON: {error}") from error
        return cls.from_dict(payload)

    def validate(self, durations_s: Mapping[str, float]) -> None:
        expected_ids = set(durations_s)
        submitted_ids = {entry.conversation_id for entry in self.probs}
        if missing := sorted(expected_ids - submitted_ids):
            raise ValueError(f"missing conversation_ids: {missing}")
        if unknown := sorted(submitted_ids - expected_ids):
            raise ValueError(f"unknown conversation_ids: {unknown}")
        for entry in self.probs:
            duration_s = durations_s[entry.conversation_id]
            if not isfinite(duration_s) or duration_s < 0:
                raise ValueError(f"invalid duration for {entry.conversation_id}")
            expected_frames = int(np.floor(duration_s * self.frame_rate_hz))
            for speaker, values in ((1, entry.speaker_1.prob), (2, entry.speaker_2.prob)):
                if len(values) != expected_frames:
                    raise ValueError(
                        f"conversation {entry.conversation_id} speaker_{speaker}: "
                        f"{len(values)} frames, expected {expected_frames}"
                    )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "task": self.task.value,
            "frame_rate_hz": self.frame_rate_hz,
            "probs": [entry.to_dict() for entry in self.probs],
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def commit(self, theta: float, *, refractory_s: float = 2.0) -> TurnBenchSubmission:
        if not isfinite(theta):
            raise ValueError("theta must be finite")
        task = (
            TurnBenchTask.EOT
            if self.task == TurnBenchProbabilityTask.EOT
            else TurnBenchTask.INTERRUPTION
        )
        events: list[TurnBenchCommittedEvent] = []
        for entry in self.probs:
            for speaker, probabilities in (
                (1, entry.speaker_1.prob),
                (2, entry.speaker_2.prob),
            ):
                for commit_time_s in commit_probability_events(
                    probabilities,
                    self.frame_rate_hz,
                    theta,
                    refractory_s=refractory_s,
                ):
                    events.append(
                        TurnBenchCommittedEvent(
                            conversation_id=entry.conversation_id,
                            speaker=speaker,
                            task=task,
                            commit_time_s=commit_time_s,
                            observed_until_s=commit_time_s,
                        )
                    )
        return TurnBenchSubmission.from_committed_events(
            events, [entry.conversation_id for entry in self.probs]
        )


def commit_probability_events(
    probabilities: Sequence[float],
    frame_rate_hz: float,
    theta: float,
    *,
    refractory_s: float = 2.0,
) -> tuple[float, ...]:
    """Apply the official strict rising-edge, frame-end commit rule."""
    if not isfinite(frame_rate_hz) or frame_rate_hz <= 0:
        raise ValueError("frame_rate_hz must be finite and positive")
    if not isfinite(theta):
        raise ValueError("theta must be finite")
    if not isfinite(refractory_s) or refractory_s < 0:
        raise ValueError("refractory_s must be finite and non-negative")
    if any(not isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("probabilities must be finite and in [0, 1]")
    refractory_frames = max(1, round(refractory_s * frame_rate_hz))
    above_previous = False
    last_committed_frame: int | None = None
    committed: list[float] = []
    for frame, probability in enumerate(probabilities):
        above = probability > theta
        if (
            above
            and not above_previous
            and (last_committed_frame is None or frame - last_committed_frame >= refractory_frames)
        ):
            committed.append((frame + 1) / frame_rate_hz)
            last_committed_frame = frame
        above_previous = above
    return tuple(committed)


def turnbench_candidate_thresholds(
    probability_file: TurnBenchProbabilityFile, *, count: int = 512
) -> tuple[float, ...]:
    if count <= 0:
        raise ValueError("threshold candidate count must be positive")
    pooled = np.asarray(
        [
            value
            for entry in probability_file.probs
            for speaker in (entry.speaker_1, entry.speaker_2)
            for value in speaker.prob
            if value > 0
        ],
        dtype=np.float64,
    )
    uniform = np.asarray([round(0.01 * index, 2) for index in range(1, 100)])
    if not len(pooled):
        return tuple(float(item) for item in uniform)
    quantiles = np.quantile(pooled, np.linspace(0, 1, count))
    return tuple(float(item) for item in np.unique(np.concatenate((quantiles, uniform))))


@dataclass(frozen=True, slots=True)
class TurnBenchSubmission:
    predictions: tuple[TurnBenchConversationPrediction, ...]
    schema_version: int = TURNBENCH_SUBMISSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TURNBENCH_SUBMISSION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported TurnBench submission schema version: {self.schema_version}"
            )
        submitted = [prediction.conversation_id for prediction in self.predictions]
        duplicates = sorted({item for item in submitted if submitted.count(item) > 1})
        if duplicates:
            raise ValueError(f"duplicate conversation_ids: {duplicates}")

    @classmethod
    def from_dict(cls, value: object) -> TurnBenchSubmission:
        mapping = _as_mapping(value, "submission")
        _require_exact_keys(mapping, {"schema_version", "predictions"}, "submission")
        schema_version = mapping["schema_version"]
        if isinstance(schema_version, bool) or schema_version != 1:
            raise ValueError("schema_version must be integer 1")
        raw_predictions = mapping["predictions"]
        if not isinstance(raw_predictions, list):
            raise ValueError("predictions must be a list")
        return cls(
            predictions=tuple(
                TurnBenchConversationPrediction.from_dict(item, index)
                for index, item in enumerate(raw_predictions)
            )
        )

    @classmethod
    def load(cls, path: Path) -> TurnBenchSubmission:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid TurnBench submission JSON: {error}") from error
        return cls.from_dict(payload)

    @classmethod
    def from_committed_events(
        cls,
        events: Iterable[TurnBenchCommittedEvent],
        conversation_ids: Sequence[str],
    ) -> TurnBenchSubmission:
        expected_ids = list(conversation_ids)
        if len(expected_ids) != len(set(expected_ids)):
            raise ValueError("expected conversation_ids must be unique")
        if any(not item for item in expected_ids):
            raise ValueError("expected conversation_ids must not be empty")

        grouped: dict[str, dict[int, dict[TurnBenchTask, list[float]]]] = {
            conversation_id: {speaker: {task: [] for task in TurnBenchTask} for speaker in (1, 2)}
            for conversation_id in expected_ids
        }
        for event in events:
            if event.conversation_id not in grouped:
                raise ValueError(f"unknown conversation_id: {event.conversation_id}")
            grouped[event.conversation_id][event.speaker][event.task].append(event.commit_time_s)

        predictions: list[TurnBenchConversationPrediction] = []
        for conversation_id in expected_ids:
            speakers: dict[int, TurnBenchSpeakerEvents] = {}
            for speaker in (1, 2):
                by_task = grouped[conversation_id][speaker]
                speakers[speaker] = TurnBenchSpeakerEvents(
                    eot=tuple(sorted(by_task[TurnBenchTask.EOT])),
                    interruption=tuple(sorted(by_task[TurnBenchTask.INTERRUPTION])),
                )
            predictions.append(
                TurnBenchConversationPrediction(
                    conversation_id=conversation_id,
                    speaker_1=speakers[1],
                    speaker_2=speakers[2],
                )
            )
        return cls(tuple(predictions))

    def validate_coverage(self, conversation_ids: Sequence[str]) -> None:
        expected = list(conversation_ids)
        if len(expected) != len(set(expected)):
            raise ValueError("expected conversation_ids must be unique")
        submitted = [prediction.conversation_id for prediction in self.predictions]
        missing = sorted(set(expected) - set(submitted))
        unknown = sorted(set(submitted) - set(expected))
        if missing:
            raise ValueError(f"missing conversation_ids: {missing}")
        if unknown:
            raise ValueError(f"unknown conversation_ids: {unknown}")

    def validate_durations(self, durations_s: Mapping[str, float]) -> None:
        self.validate_coverage(tuple(durations_s))
        for prediction in self.predictions:
            duration_s = durations_s[prediction.conversation_id]
            if not isfinite(duration_s) or duration_s < 0:
                raise ValueError(f"invalid duration for {prediction.conversation_id}")
            for speaker_index, speaker in (
                (1, prediction.speaker_1),
                (2, prediction.speaker_2),
            ):
                for task, times in (
                    (TurnBenchTask.EOT, speaker.eot),
                    (TurnBenchTask.INTERRUPTION, speaker.interruption),
                ):
                    past_end = [time_s for time_s in times if time_s > duration_s]
                    if past_end:
                        raise ValueError(
                            f"conversation {prediction.conversation_id} speaker_{speaker_index} "
                            f"{task}: events past duration {duration_s}: {past_end}"
                        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def score_with_official_turnbench(
    predictions_path: Path,
    *,
    dataset: str = TURNBENCH_DEV_DATASET,
    revision: str | None = TURNBENCH_DEV_REVISION,
    scorer_project: Path | None = None,
) -> dict[str, Any]:
    """Score with the pinned upstream package in its isolated uv environment."""
    project = scorer_project or Path(__file__).parents[3] / "environments" / "turnbench"
    command = [
        "uv",
        "run",
        "--frozen",
        "--project",
        str(project),
        "python",
        str(project / "score_bridge.py"),
        str(predictions_path.resolve()),
        "--dataset",
        dataset,
    ]
    if revision is not None:
        command.extend(("--revision", revision))
    environment = os.environ.copy()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"official TurnBench scorer failed: {detail}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("official TurnBench scorer returned invalid JSON") from error
    if not isinstance(result, dict):
        raise RuntimeError("official TurnBench scorer returned a non-object result")
    return result


def sweep_with_official_turnbench(
    eot_probabilities_path: Path,
    interruption_probabilities_path: Path,
    output_dir: Path,
    *,
    fp_budget: float = 0.10,
    dataset: str = TURNBENCH_DEV_DATASET,
    revision: str | None = TURNBENCH_DEV_REVISION,
    scorer_project: Path | None = None,
) -> dict[str, Any]:
    """Select independent EOT/INT dev operating points using pinned upstream code."""
    eot = TurnBenchProbabilityFile.load(eot_probabilities_path)
    interruption = TurnBenchProbabilityFile.load(interruption_probabilities_path)
    if eot.task != TurnBenchProbabilityTask.EOT:
        raise ValueError("EOT probabilities must declare task='eot'")
    if interruption.task != TurnBenchProbabilityTask.INTERRUPTION:
        raise ValueError("interruption probabilities must declare task='int'")
    if not 0 <= fp_budget <= 1:
        raise ValueError("fp_budget must be in [0, 1]")

    project = scorer_project or Path(__file__).parents[3] / "environments" / "turnbench"
    command = [
        "uv",
        "run",
        "--frozen",
        "--project",
        str(project),
        "python",
        str(project / "sweep_bridge.py"),
        "--eot-probs",
        str(eot_probabilities_path.resolve()),
        "--int-probs",
        str(interruption_probabilities_path.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--dataset",
        dataset,
        "--fp-budget",
        str(fp_budget),
    ]
    if revision is not None:
        command.extend(("--revision", revision))
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"official TurnBench sweep failed: {detail}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("official TurnBench sweep returned invalid JSON") from error
    if not isinstance(result, dict):
        raise RuntimeError("official TurnBench sweep returned a non-object result")
    return result
