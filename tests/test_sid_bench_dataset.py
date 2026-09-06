from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from turn_eval.benchmarks.sid_bench import load_sid_bench, load_sid_bench_predictions


def _fixture_root(tmp_path):
    root = tmp_path / "SID-bench"
    audio_root = root / "test_wavs" / "en"
    audio_root.mkdir(parents=True)
    rows = (
        {
            "audio": "begin.wav",
            "total_nonbreak": False,
            "duration": 0.1,
            "break_time": 0.0,
            "text_with_break": "<break>Hello",
        },
        {
            "audio": "middle.wav",
            "total_nonbreak": False,
            "duration": 0.1,
            "break_time": 0.05,
            "text_with_break": "Uh <break>wait",
        },
        {
            "audio": "hold.wav",
            "total_nonbreak": True,
            "duration": 0.1,
            "break_time": 0.1,
            "text_with_break": "Uh huh",
        },
    )
    for row in rows:
        sf.write(
            audio_root / row["audio"],
            np.zeros(1_600, dtype=np.float32),
            16_000,
            subtype="PCM_16",
        )
    (root / "test_wavs" / "en_test_lines.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return root


def test_sid_bench_loader_preserves_temporal_labels_without_exposing_agent_audio(tmp_path) -> None:
    dataset = load_sid_bench(_fixture_root(tmp_path), "en", strict_counts=False)

    assert [item.category for item in dataset.items] == [
        "interruption_beginning",
        "interruption_middle",
        "nonbreak",
    ]
    assert [item.gold_break_time_s for item in dataset.items] == [0.0, 0.05, None]
    assert dataset.items[0].load_audio().sample_rate == 16_000
    assert dataset.benchmark_metadata["agent_audio_available"] is False
    assert dataset.metadata["license"].startswith("unspecified")


def test_sid_bench_prediction_loader_and_coverage_are_strict(tmp_path) -> None:
    dataset = load_sid_bench(_fixture_root(tmp_path), "en", strict_counts=False)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "\n".join(
            (
                json.dumps({"audio": "begin.wav", "total_nonbreak": False, "break_time": 0.02}),
                json.dumps({"audio": "middle.wav", "total_nonbreak": True, "break_time": -1}),
                json.dumps({"audio": "hold.wav", "total_nonbreak": True, "break_time": -1}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_sid_bench_predictions(predictions)
    records = dataset.prediction_records(loaded)

    assert [record.predicted_break_time_s for record in records] == [0.02, None, None]
    with pytest.raises(ValueError, match="coverage mismatch"):
        dataset.prediction_records({"begin.wav": 0.02})


def test_sid_bench_loader_rejects_label_leaking_break_schema(tmp_path) -> None:
    root = _fixture_root(tmp_path)
    path = root / "test_wavs" / "en_test_lines.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["text_with_break"] = "Hello"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="a <break>"):
        load_sid_bench(root, "en", strict_counts=False)
