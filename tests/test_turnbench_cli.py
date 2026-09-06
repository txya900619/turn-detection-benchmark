from __future__ import annotations

import json

from turn_eval.cli import main


def test_turnbench_validate_cli(tmp_path, capsys) -> None:
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "predictions": [
                    {
                        "conversation_id": "20",
                        "speaker_1": {"eot": [1.0], "interruption": []},
                        "speaker_2": {"eot": [], "interruption": []},
                    }
                ],
            }
        )
    )
    durations = tmp_path / "durations.json"
    durations.write_text(json.dumps({"20": 1.0}))

    assert main(["turnbench-validate", str(predictions), "--durations", str(durations)]) == 0
    assert "1 conversations" in capsys.readouterr().out
