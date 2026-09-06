from __future__ import annotations

from typing import Protocol

from turn_eval.core.types import (
    ClipRequest,
    DecisionRequest,
    ModelDescriptor,
    StatePrediction,
    TurnPrediction,
)


class PointwiseModelAdapter(Protocol):
    """Interface implemented by causal pointwise model adapters."""

    @property
    def descriptor(self) -> ModelDescriptor: ...

    def predict(self, request: DecisionRequest) -> TurnPrediction: ...


class ClipClassificationModelAdapter(Protocol):
    """Interface for bounded clip classification without gold annotations."""

    @property
    def descriptor(self) -> ModelDescriptor: ...

    def predict_clip(self, request: ClipRequest) -> StatePrediction: ...
