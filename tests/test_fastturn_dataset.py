from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from turn_eval.benchmarks.fastturn import load_fastturn
from turn_eval.core.types import TurnState


def _fixture_root(tmp_path):
    root = tmp_path / "fasttrun-testset"
    wavs = root / "wavs"
    wavs.mkdir(parents=True)
    rows = []
    for index, (state, text, sample_rate) in enumerate(
        (
            ("complete", "好<pause>了<complete>", 48_000),
            ("incomplete", "我想要<incomplete>", 48_000),
            ("backchannel", "嗯嗯<backchannel>", 48_000),
            ("wait", "你先別說", 22_050),
        )
    ):
        key = f"clip-{index}"
        path = wavs / f"{key}.wav"
        sf.write(path, np.zeros(sample_rate // 10, dtype=np.float32), sample_rate, subtype="PCM_16")
        rows.append(
            {"key": key, "wav": f"wavs/{key}.wav", "txt": text, "state": state, "lang": "zh"}
        )
    (root / "data.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return root


def test_fastturn_loader_strips_label_leakage_and_preserves_native_rates(tmp_path) -> None:
    dataset = load_fastturn(_fixture_root(tmp_path), strict_counts=False)

    assert [item.transcript_without_target for item in dataset.items] == [
        "好<pause>了",
        "我想要",
        "嗯嗯",
        "你先別說",
    ]
    assert [item.source_kind for item in dataset.items] == ["real", "real", "real", "synthetic"]
    examples = list(dataset.iter_examples())
    assert [example.label for example in examples] == list(TurnState)
    assert [example.user_audio.sample_rate for example in examples] == [
        48_000,
        48_000,
        48_000,
        22_050,
    ]
    assert all(example.transcript is None for example in examples)
    oracle_examples = list(dataset.iter_examples(include_transcript=True))
    assert [example.transcript for example in oracle_examples] == [
        "好了",
        "我想要",
        "嗯嗯",
        "你先別說",
    ]
    assert dataset.metadata["class_counts"] == {
        "backchannel": 1,
        "complete": 1,
        "incomplete": 1,
        "wait": 1,
    }


def test_fastturn_loader_supports_native_two_state_subset(tmp_path) -> None:
    dataset = load_fastturn(_fixture_root(tmp_path), strict_counts=False)

    examples = list(
        dataset.iter_examples(labels=(TurnState.COMPLETE, TurnState.INCOMPLETE), limit=2)
    )

    assert [example.label for example in examples] == [TurnState.COMPLETE, TurnState.INCOMPLETE]


def test_fastturn_loader_rejects_marker_mismatch(tmp_path) -> None:
    root = _fixture_root(tmp_path)
    rows = [json.loads(line) for line in (root / "data.jsonl").read_text().splitlines()]
    rows[0]["txt"] = "好了<incomplete>"
    (root / "data.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="marker mismatch"):
        load_fastturn(root, strict_counts=False)


def test_fastturn_loader_rejects_partial_dataset_in_strict_mode(tmp_path) -> None:
    with pytest.raises(ValueError, match="class counts mismatch"):
        load_fastturn(_fixture_root(tmp_path))
