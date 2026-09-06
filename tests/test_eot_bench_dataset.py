from __future__ import annotations

import io
import wave
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from turn_eval.benchmarks.eot_bench import eot_bench_row_to_example, load_eot_bench
from turn_eval.core.types import DecisionLabel


def _wav_bytes(*, sample_rate: int = 16_000, duration_s: float = 1.0) -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * round(sample_rate * duration_s))
    return stream.getvalue()


def _row() -> dict[str, object]:
    return {
        "id": "en__123_4__user_turn_007",
        "audio": {"bytes": _wav_bytes(), "path": None},
        "language": "en",
        "duration": 1.0,
        "silence_spans": [
            {"start": 0.2, "end": 0.45},
            {"start": 0.7, "end": 1.0},
        ],
        "words": [{"start": 0.0, "end": 0.1, "word": "hello"}],
        "messages": [{"role": "assistant", "content": "How can I help?"}],
    }


def test_eot_bench_row_uses_official_grid_and_final_span_label() -> None:
    example = eot_bench_row_to_example(_row())

    assert example.conversation_id == "en__123_4"
    assert example.user_audio.sample_rate == 16_000
    assert example.messages[0].role == "assistant"
    assert [point.timestamp_s for point in example.decision_points] == [
        0.2,
        0.3,
        0.4,
        0.45,
        0.7,
        0.8,
        0.9,
        1.0,
    ]
    assert example.decision_points[0].silence_duration_s == 0.0
    assert example.decision_points[-1].label is DecisionLabel.EOT
    assert {point.label for point in example.decision_points[:4]} == {DecisionLabel.HOLD}


def test_submillisecond_tail_points_have_unique_ids() -> None:
    row = _row()
    row["silence_spans"] = [{"start": 0.2, "end": 0.3005}, {"start": 0.7, "end": 1.0}]

    example = eot_bench_row_to_example(row)

    point_ids = [point.id for point in example.decision_points]
    assert len(point_ids) == len(set(point_ids))


def test_loader_pins_metadata_and_marks_limited_run_diagnostic(tmp_path: Path, monkeypatch) -> None:
    parquet_path = tmp_path / "validation.parquet"
    pq.write_table(
        pa.Table.from_pylist([_row(), {**_row(), "id": "en__123_4__user_turn_008"}]), parquet_path
    )
    monkeypatch.setattr(
        "turn_eval.benchmarks.eot_bench.hf_hub_download",
        lambda **_: str(parquet_path),
    )

    dataset = load_eot_bench("en", revision="fixture-revision", limit=1)

    assert len(dataset.examples) == 1
    assert dataset.total_rows == 2
    assert dataset.metadata["revision"] == "fixture-revision"
    assert dataset.metadata["checksum"].startswith("sha256:")
    assert dataset.benchmark_metadata["partial_run"] is True
    assert dataset.benchmark_metadata["primary_leaderboard_eligible"] is False
