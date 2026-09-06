from __future__ import annotations

from pathlib import Path

import pytest

from turn_eval.cli import main


def test_help_is_available(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "run-synthetic" in capsys.readouterr().out


def test_cli_runs_and_reports_without_rerunning_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run-synthetic", "--output-dir", str(tmp_path)]) == 0
    run_dir = Path(capsys.readouterr().out.strip())
    assert (run_dir / "SUCCESS").is_file()

    assert main(["report", str(run_dir)]) == 0
    report = capsys.readouterr().out
    assert "False cutoff @ 300 ms" in report
    assert "RTF" in report
    assert "n/a" not in report
