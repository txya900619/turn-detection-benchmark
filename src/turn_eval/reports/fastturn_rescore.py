"""Derive a FastTurn two-state EOT run from a completed multi-state run.

The FastTurn ``eot`` leaderboard scores only the native Complete/Incomplete
clips. A model evaluated on the ``real`` or ``full`` suite has already produced
a prediction for every one of those clips, so its two-state row can be derived
without running inference again.

The derivation is deliberately narrow. It restricts the gold set, it never
remaps a label, and it never selects a threshold. A prediction that falls
outside the two-state vocabulary becomes an abstention that
:func:`turn_eval.metrics.state.evaluate_turn_states` keeps in the denominator as
a closed-set error, which is the behaviour the source run already declared
through ``label_mapping.out_of_suite_labels``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from turn_eval.artifacts import content_hash
from turn_eval.core.types import (
    ChannelMode,
    EvaluationMode,
    ExecutionMode,
    ModelDescriptor,
    ServiceMode,
    SignalProvenance,
    StatePredictionRecord,
    SupportStatus,
    TranscriptMode,
    TurnState,
)

EOT_STATES: tuple[TurnState, ...] = (TurnState.COMPLETE, TurnState.INCOMPLETE)
DERIVATION_STRATEGY = "eot_subset_restriction_from_multistate_run_v1"


@dataclass(frozen=True, slots=True)
class SourceRun:
    """A completed clip-classification run that a two-state row can be derived from."""

    run_dir: Path
    manifest: Mapping[str, object]
    label_mapping: Mapping[str, object]
    records: tuple[StatePredictionRecord, ...]


@dataclass(frozen=True, slots=True)
class DerivedEotRun:
    """The restricted record set plus the provenance a manifest must record."""

    records: tuple[StatePredictionRecord, ...]
    label_mapping: dict[str, object]
    derivation: dict[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_records(parquet_path: Path) -> tuple[StatePredictionRecord, ...]:
    table = pq.read_table(parquet_path).to_pylist()
    records: list[StatePredictionRecord] = []
    for row in table:
        predicted = row["predicted_label"]
        records.append(
            StatePredictionRecord(
                example_id=row["example_id"],
                group_id=row["group_id"],
                language=row["language"],
                label=TurnState(row["label"]),
                predicted_label=None if predicted is None else TurnState(predicted),
                scores=json.loads(row["scores_json"]),
                inference_latency_ms=row["inference_latency_ms"],
                metadata=json.loads(row["metadata_json"]),
                model_output=json.loads(row["model_output_json"]),
            )
        )
    return tuple(records)


def read_source_run(run_dir: Path) -> SourceRun:
    """Read a completed multi-state run, refusing partial or unfinished artifacts."""
    run_dir = run_dir.resolve()
    _require((run_dir / "SUCCESS").is_file(), f"source run has no SUCCESS marker: {run_dir}")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    label_mapping = json.loads((run_dir / "label_mapping.json").read_text(encoding="utf-8"))

    benchmark = manifest.get("benchmark", {})
    _require(
        not benchmark.get("partial_run", False),
        "a partial source run cannot produce a leaderboard-eligible derived row",
    )
    _require(
        str(benchmark.get("name", "")).startswith("fastturn-"),
        f"source run is not a FastTurn run: {benchmark.get('name')!r}",
    )
    records = _load_records(run_dir / "predictions.parquet")
    _require(bool(records), "source run contains no predictions")
    return SourceRun(
        run_dir=run_dir,
        manifest=manifest,
        label_mapping=label_mapping,
        records=records,
    )


def derive_eot_run(source: SourceRun) -> DerivedEotRun:
    """Restrict a multi-state run to the native two-state EOT subset."""
    allowed = set(EOT_STATES)
    subset = tuple(record for record in source.records if record.label in allowed)
    _require(
        bool(subset),
        "source run contains no Complete or Incomplete clips to derive a two-state row from",
    )
    _require(
        any(record.label not in allowed for record in source.records),
        "source run is already a two-state run; score it directly instead of deriving",
    )

    restricted: list[StatePredictionRecord] = []
    out_of_suite = 0
    source_abstentions = 0
    for record in subset:
        predicted = record.predicted_label
        if predicted is None:
            source_abstentions += 1
        elif predicted not in allowed:
            out_of_suite += 1
            predicted = None
        # Scores are preserved verbatim, including the out-of-suite classes, so
        # the raw model output stays auditable in the derived artifact.
        restricted.append(
            StatePredictionRecord(
                example_id=record.example_id,
                group_id=record.group_id,
                language=record.language,
                label=record.label,
                predicted_label=predicted,
                scores=record.scores,
                inference_latency_ms=record.inference_latency_ms,
                metadata=record.metadata,
                model_output=record.model_output,
            )
        )

    label_mapping = {
        "strategy": DERIVATION_STRATEGY,
        "output_labels": [state.value for state in EOT_STATES],
        "native_mapping": {state.value: state.value for state in EOT_STATES},
        "abstain_states": sorted(
            {
                record.predicted_label.value
                for record in subset
                if record.predicted_label is not None and record.predicted_label not in allowed
            }
        ),
        "abstain_scoring": "closed-set-incorrect",
        "out_of_suite_labels": "abstain-closed-set-incorrect",
        "source_label_mapping_strategy": source.label_mapping.get("strategy"),
        "score_semantics": source.label_mapping.get("score_semantics"),
        "calibration_source": "none-derived-from-completed-run",
        "split_role": "none",
    }
    derivation = {
        "strategy": DERIVATION_STRATEGY,
        "inference_rerun": False,
        "source_run": str(source.run_dir),
        "source_run_id": source.manifest.get("run_id"),
        "source_benchmark": source.manifest.get("benchmark", {}).get("name"),
        "source_manifest_sha256": _sha256(source.run_dir / "manifest.json"),
        "source_predictions_sha256": _sha256(source.run_dir / "predictions.parquet"),
        "source_label_mapping_hash": content_hash(dict(source.label_mapping)),
        "source_examples": len(source.records),
        "derived_examples": len(restricted),
        "out_of_suite_predictions_abstained": out_of_suite,
        "source_abstentions_preserved": source_abstentions,
    }
    return DerivedEotRun(
        records=tuple(restricted),
        label_mapping=label_mapping,
        derivation=derivation,
    )


def descriptor_from_manifest(manifest: Mapping[str, object]) -> ModelDescriptor:
    """Rebuild the source descriptor so the derived manifest keeps its identity.

    ``execution_modes`` and ``supported_languages`` are not serialized into a
    manifest. They are reconstructed here from the recorded evaluation mode and
    benchmark language purely to satisfy descriptor validation; neither field is
    written back out by :func:`turn_eval.artifacts.build_manifest`.
    """
    model = dict(manifest["model"])  # type: ignore[arg-type]
    mode = dict(manifest["evaluation_mode"])  # type: ignore[arg-type]
    benchmark = dict(manifest.get("benchmark", {}))  # type: ignore[arg-type]
    return ModelDescriptor(
        name=model["name"],
        revision=model["revision"],
        native_sample_rate=model["native_sample_rate"],
        execution_modes=frozenset({ExecutionMode(mode["execution_mode"])}),
        output_signals=frozenset(model["output_signals"]),
        supported_languages=frozenset({benchmark.get("language", "zh")}),
        accepted_inputs=frozenset(model["accepted_inputs"]),
        frame_interval_ms=model["frame_interval_ms"],
        context_window_s=model["context_window_s"],
        algorithmic_lookahead_ms=model["algorithmic_lookahead_ms"],
        natural_score_point_ms=model["natural_score_point_ms"],
        artifact_checksum=model["artifact_checksum"],
        artifact_size_bytes=model["artifact_size_bytes"],
        model_license=model["model_license"],
        code_license=model["code_license"],
        known_training_datasets=tuple(model["known_training_datasets"]),
        allow_unverified_zero_shot=model["allow_unverified_zero_shot"],
    )


def evaluation_mode_from_manifest(manifest: Mapping[str, object]) -> EvaluationMode:
    """Carry the source run's honest capability labels into the derived run."""
    mode = dict(manifest["evaluation_mode"])  # type: ignore[arg-type]
    return EvaluationMode(
        channel_mode=ChannelMode(mode["channel_mode"]),
        execution_mode=ExecutionMode(mode["execution_mode"]),
        signal_provenance=SignalProvenance(mode["signal_provenance"]),
        transcript_mode=TranscriptMode(mode["transcript_mode"]),
        service_mode=ServiceMode(mode["service_mode"]),
        language_support=SupportStatus(mode["language_support"]),
    )


def derived_model_options(
    source_manifest: Mapping[str, object],
    derivation: Mapping[str, object],
) -> dict[str, object]:
    """Preserve the source model options and append the derivation record."""
    options = dict(source_manifest.get("model", {}).get("options", {}))  # type: ignore[union-attr]
    options["suite"] = "eot"
    options["derivation"] = dict(derivation)
    return options


def derived_benchmark_metadata(
    source_manifest: Mapping[str, object],
    records: Sequence[StatePredictionRecord],
) -> dict[str, object]:
    """Relabel the benchmark as the two-state suite without losing provenance."""
    benchmark = dict(source_manifest.get("benchmark", {}))  # type: ignore[arg-type]
    benchmark["name"] = "fastturn-eot"
    benchmark["derived_from"] = source_manifest.get("benchmark", {}).get("name")
    benchmark["partial_run"] = False
    benchmark["evaluated_rows"] = len(records)
    return benchmark
