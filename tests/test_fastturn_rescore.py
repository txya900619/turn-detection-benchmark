from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from turn_eval.artifacts import (
    build_state_manifest,
    make_state_run_id,
    write_state_run_artifacts,
)
from turn_eval.core.types import (
    AudioData,
    ChannelMode,
    ClipClassificationExample,
    ClipRequest,
    EvaluationMode,
    ExecutionMode,
    ModelDescriptor,
    ServiceMode,
    SignalProvenance,
    StatePrediction,
    SupportStatus,
    TranscriptMode,
    TurnState,
)
from turn_eval.metrics.state import evaluate_turn_states
from turn_eval.reports.fastturn_rescore import (
    EOT_STATES,
    derive_eot_run,
    derived_benchmark_metadata,
    derived_model_options,
    descriptor_from_manifest,
    evaluation_mode_from_manifest,
    read_source_run,
)

FOUR_STATES = (TurnState.COMPLETE, TurnState.INCOMPLETE, TurnState.BACKCHANNEL, TurnState.WAIT)

_MODE = EvaluationMode(
    ChannelMode.NATIVE_SINGLE,
    ExecutionMode.OFFLINE,
    SignalProvenance.NATIVE,
    TranscriptMode.NONE,
    ServiceMode.LOCAL,
    SupportStatus.SUPPORTED,
)


def _write_source_run(
    tmp_path: Path,
    *,
    pairs: tuple[tuple[TurnState, TurnState | None], ...],
    benchmark_name: str = "fastturn-full",
    partial: bool = False,
) -> Path:
    """Write a completed clip-classification run whose predictions are scripted."""
    scripted = {f"clip-{index}": predicted for index, (_, predicted) in enumerate(pairs)}

    class Adapter:
        descriptor = ModelDescriptor(
            "rescore-fixture",
            "v1",
            16_000,
            frozenset({ExecutionMode.OFFLINE}),
            frozenset({"turn_state"}),
            frozenset({"zh"}),
        )

        def predict_clip(self, request: ClipRequest) -> StatePrediction:
            predicted = scripted[request.example_id]
            scores = {state.value: 0.0 for state in FOUR_STATES}
            if predicted is not None:
                scores[predicted.value] = 1.0
            return StatePrediction(request.example_id, predicted, scores)

    from turn_eval.runner.clip_classification import run_clip_classification

    examples = tuple(
        ClipClassificationExample(
            f"clip-{index}",
            "zh",
            AudioData(np.zeros(1_600, dtype=np.float32), 16_000),
            label,
        )
        for index, (label, _) in enumerate(pairs)
    )
    mapping = {
        "strategy": "native_terminal_tag_exact_only_v1",
        "output_labels": [state.value for state in FOUR_STATES],
        "out_of_suite_labels": "abstain-closed-set-incorrect",
        "score_semantics": "hard-decoded-one-hot-not-probabilities",
        "calibration_source": "none-native-generation",
        "split_role": "none",
    }
    model = Adapter()
    run = run_clip_classification(examples, model, _MODE)
    metrics = evaluate_turn_states(run.predictions, FOUR_STATES)
    run_id = make_state_run_id(model.descriptor, mapping)
    manifest = build_state_manifest(
        run_id=run_id,
        model=model.descriptor,
        mode=_MODE,
        label_mapping=mapping,
        dataset_metadata={
            "repository": "fixture",
            "revision": "v1",
            "class_counts": {state.value: 1 for state in FOUR_STATES},
        },
        benchmark_metadata={
            "name": benchmark_name,
            "protocol": "clip_classification",
            "language": "zh",
            "partial_run": partial,
        },
    )
    return write_state_run_artifacts(
        tmp_path,
        run_id=run_id,
        manifest=manifest,
        run=run,
        label_mapping=mapping,
        metrics=metrics,
    )


def test_derivation_restricts_gold_and_abstains_on_out_of_suite(tmp_path: Path) -> None:
    run_dir = _write_source_run(
        tmp_path,
        pairs=(
            (TurnState.COMPLETE, TurnState.COMPLETE),
            (TurnState.INCOMPLETE, TurnState.COMPLETE),
            (TurnState.COMPLETE, TurnState.BACKCHANNEL),
            (TurnState.INCOMPLETE, TurnState.WAIT),
            (TurnState.BACKCHANNEL, TurnState.BACKCHANNEL),
            (TurnState.WAIT, TurnState.WAIT),
        ),
    )
    derived = derive_eot_run(read_source_run(run_dir))

    assert len(derived.records) == 4
    assert {record.label for record in derived.records} == set(EOT_STATES)
    # Backchannel and Wait predictions on a two-state clip become abstentions.
    assert [record.predicted_label for record in derived.records] == [
        TurnState.COMPLETE,
        TurnState.COMPLETE,
        None,
        None,
    ]
    assert derived.derivation["out_of_suite_predictions_abstained"] == 2
    assert derived.derivation["source_abstentions_preserved"] == 0
    assert derived.derivation["inference_rerun"] is False
    assert derived.derivation["derived_examples"] == 4
    assert derived.derivation["source_examples"] == 6
    assert derived.label_mapping["abstain_states"] == ["backchannel", "wait"]


def test_abstentions_are_scored_as_closed_set_errors(tmp_path: Path) -> None:
    run_dir = _write_source_run(
        tmp_path,
        pairs=(
            (TurnState.COMPLETE, TurnState.COMPLETE),
            (TurnState.INCOMPLETE, TurnState.BACKCHANNEL),
            (TurnState.WAIT, TurnState.WAIT),
        ),
    )
    derived = derive_eot_run(read_source_run(run_dir))
    metrics = evaluate_turn_states(derived.records, EOT_STATES)

    assert metrics["counts"]["examples"] == 2
    assert metrics["abstentions"]["count"] == 1
    # The abstained clip stays in the denominator rather than being dropped.
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_source_abstentions_are_preserved_not_recounted(tmp_path: Path) -> None:
    run_dir = _write_source_run(
        tmp_path,
        pairs=(
            (TurnState.COMPLETE, None),
            (TurnState.INCOMPLETE, TurnState.INCOMPLETE),
            (TurnState.WAIT, TurnState.WAIT),
        ),
    )
    derived = derive_eot_run(read_source_run(run_dir))

    assert derived.derivation["source_abstentions_preserved"] == 1
    assert derived.derivation["out_of_suite_predictions_abstained"] == 0


def test_raw_scores_survive_the_derivation(tmp_path: Path) -> None:
    run_dir = _write_source_run(
        tmp_path,
        pairs=(
            (TurnState.COMPLETE, TurnState.BACKCHANNEL),
            (TurnState.WAIT, TurnState.WAIT),
        ),
    )
    derived = derive_eot_run(read_source_run(run_dir))
    record = derived.records[0]

    # The prediction is abstained, but the original four-way output is auditable.
    assert record.predicted_label is None
    assert record.scores["backchannel"] == pytest.approx(1.0)


def test_partial_source_run_is_refused(tmp_path: Path) -> None:
    run_dir = _write_source_run(
        tmp_path,
        pairs=((TurnState.COMPLETE, TurnState.COMPLETE), (TurnState.WAIT, TurnState.WAIT)),
        partial=True,
    )
    with pytest.raises(ValueError, match="partial source run"):
        read_source_run(run_dir)


def test_non_fastturn_source_run_is_refused(tmp_path: Path) -> None:
    run_dir = _write_source_run(
        tmp_path,
        pairs=((TurnState.COMPLETE, TurnState.COMPLETE), (TurnState.WAIT, TurnState.WAIT)),
        benchmark_name="easy-turn-testset",
    )
    with pytest.raises(ValueError, match="not a FastTurn run"):
        read_source_run(run_dir)


def test_two_state_source_run_is_refused(tmp_path: Path) -> None:
    run_dir = _write_source_run(
        tmp_path,
        pairs=(
            (TurnState.COMPLETE, TurnState.COMPLETE),
            (TurnState.INCOMPLETE, TurnState.INCOMPLETE),
        ),
    )
    with pytest.raises(ValueError, match="already a two-state run"):
        derive_eot_run(read_source_run(run_dir))


def test_unfinished_source_run_is_refused(tmp_path: Path) -> None:
    run_dir = _write_source_run(
        tmp_path,
        pairs=((TurnState.COMPLETE, TurnState.COMPLETE), (TurnState.WAIT, TurnState.WAIT)),
    )
    (run_dir / "SUCCESS").unlink()
    with pytest.raises(ValueError, match="SUCCESS marker"):
        read_source_run(run_dir)


def test_derived_manifest_keeps_model_identity_and_capability_labels(tmp_path: Path) -> None:
    run_dir = _write_source_run(
        tmp_path,
        pairs=(
            (TurnState.COMPLETE, TurnState.COMPLETE),
            (TurnState.WAIT, TurnState.WAIT),
        ),
    )
    source = read_source_run(run_dir)
    derived = derive_eot_run(source)

    descriptor = descriptor_from_manifest(source.manifest)
    assert descriptor.name == "rescore-fixture"
    assert descriptor.revision == "v1"
    assert descriptor.output_signals == frozenset({"turn_state"})

    # Honest capability labels are carried over rather than re-declared.
    assert evaluation_mode_from_manifest(source.manifest) == _MODE

    options = derived_model_options(source.manifest, derived.derivation)
    assert options["suite"] == "eot"
    assert options["derivation"]["inference_rerun"] is False

    benchmark = derived_benchmark_metadata(source.manifest, derived.records)
    assert benchmark["name"] == "fastturn-eot"
    assert benchmark["derived_from"] == "fastturn-full"
    assert benchmark["evaluated_rows"] == 1


def test_source_hashes_pin_the_exact_artifact(tmp_path: Path) -> None:
    run_dir = _write_source_run(
        tmp_path,
        pairs=(
            (TurnState.COMPLETE, TurnState.COMPLETE),
            (TurnState.WAIT, TurnState.WAIT),
        ),
    )
    derivation = derive_eot_run(read_source_run(run_dir)).derivation
    expected = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert derivation["source_run_id"] == expected["run_id"]
    assert len(derivation["source_manifest_sha256"]) == 64
    assert len(derivation["source_predictions_sha256"]) == 64
