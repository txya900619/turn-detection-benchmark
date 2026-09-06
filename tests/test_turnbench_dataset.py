from __future__ import annotations

from io import BytesIO

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from turn_eval.benchmarks.turnbench import load_turnbench


def audio_bytes(samples: np.ndarray, sample_rate: int = 48_000) -> bytes:
    stream = BytesIO()
    sf.write(stream, samples, sample_rate, format="FLAC", subtype="PCM_24")
    return stream.getvalue()


def write_fixture(path, *, unequal_channels: bool = False) -> None:
    first = np.linspace(-0.2, 0.2, 480, dtype=np.float32)
    second = first[:-1] if unequal_channels else -first
    table = pa.Table.from_pylist(
        [
            {
                "conversation_id": "20",
                "speaker_1_audio": {"bytes": audio_bytes(first), "path": None},
                "speaker_2_audio": {"bytes": audio_bytes(second), "path": None},
                "speaker_1_audio_preview": {"bytes": b"not audio", "path": None},
                "speaker_2_audio_preview": {"bytes": b"not audio", "path": None},
                "metadata": {"conversation_type": "Synthetic"},
            }
        ]
    )
    pq.write_table(table, path / "dev-00000.parquet")


def test_local_turnbench_is_lazy_and_preserves_physical_channels(tmp_path) -> None:
    write_fixture(tmp_path)
    dataset = load_turnbench(source=tmp_path, revision="fixture-v1")

    assert dataset.conversation_ids() == ("20",)
    assert dataset.revision == "fixture-v1"
    conversation = next(iter(dataset.iter_conversations()))
    assert conversation.conversation_id == "20"
    assert conversation.speaker_1_audio.sample_rate == 48_000
    assert len(conversation.speaker_1_audio.samples) == 480
    assert np.allclose(
        conversation.speaker_1_audio.samples,
        -conversation.speaker_2_audio.samples,
        atol=2e-7,
    )
    assert conversation.metadata["conversation_type"] == "Synthetic"


def test_turnbench_rejects_unaligned_channels(tmp_path) -> None:
    write_fixture(tmp_path, unequal_channels=True)
    dataset = load_turnbench(source=tmp_path, revision="fixture-v1")

    with pytest.raises(ValueError, match="sample-aligned"):
        next(iter(dataset.iter_conversations()))


def test_turnbench_limit_must_be_positive(tmp_path) -> None:
    write_fixture(tmp_path)
    dataset = load_turnbench(source=tmp_path, revision="fixture-v1")

    with pytest.raises(ValueError, match="positive"):
        next(iter(dataset.iter_conversations(limit=0)))
