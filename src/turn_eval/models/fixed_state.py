from __future__ import annotations

from turn_eval.core.types import (
    ClipRequest,
    ExecutionMode,
    ModelDescriptor,
    StatePrediction,
    TurnState,
)


class FixedStateAdapter:
    """Deterministic diagnostic adapter for clip-runner integration checks."""

    def __init__(self, predicted_label: TurnState, output_labels: tuple[TurnState, ...]) -> None:
        if predicted_label not in output_labels:
            raise ValueError("fixed prediction must belong to the output label vocabulary")
        self.predicted_label = predicted_label
        self.output_labels = output_labels
        self._descriptor = ModelDescriptor(
            name=f"fixed-state-{predicted_label.value}",
            revision="diagnostic-v1",
            native_sample_rate=16_000,
            execution_modes=frozenset({ExecutionMode.OFFLINE}),
            output_signals=frozenset({"turn_state"}),
            supported_languages=frozenset({"zh"}),
            code_license="MIT",
        )

    @property
    def descriptor(self) -> ModelDescriptor:
        return self._descriptor

    def predict_clip(self, request: ClipRequest) -> StatePrediction:
        scores = {label.value: float(label is self.predicted_label) for label in self.output_labels}
        return StatePrediction(request.example_id, self.predicted_label, scores)
