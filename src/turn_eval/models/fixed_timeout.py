from __future__ import annotations

from time import monotonic_ns

from turn_eval.core.types import (
    DecisionRequest,
    ExecutionMode,
    ModelDescriptor,
    TurnPrediction,
)


class FixedTimeoutAdapter:
    """Deterministic oracle-VAD timeout baseline.

    The request exposes elapsed silence but never its label or future endpoint.
    This matches the fixed endpoint delay used by eot-bench's VAD baseline.
    """

    def __init__(
        self,
        *,
        timeout_ms: float = 500.0,
        sample_rate: int = 1_000,
        lookahead_ms: float = 0.0,
    ) -> None:
        if timeout_ms < 0:
            raise ValueError("timeout_ms cannot be negative")
        self.timeout_ms = timeout_ms
        self._descriptor = ModelDescriptor(
            name="fixed-timeout-oracle-vad",
            revision="m1-v1",
            native_sample_rate=sample_rate,
            execution_modes=frozenset({ExecutionMode.POINTWISE}),
            output_signals=frozenset({"eot"}),
            supported_languages=frozenset({"en", "zh"}),
            accepted_inputs=frozenset({"user_audio", "silence_duration"}),
            algorithmic_lookahead_ms=lookahead_ms,
            code_license="MIT",
        )

    @property
    def descriptor(self) -> ModelDescriptor:
        return self._descriptor

    def predict(self, request: DecisionRequest) -> TurnPrediction:
        if request.user_audio.sample_rate != self.descriptor.native_sample_rate:
            raise ValueError("adapter received audio at a non-native sample rate")
        p_eot = float(request.silence_duration_s * 1_000 + 1e-9 >= self.timeout_ms)
        return TurnPrediction(
            example_id=request.example_id,
            decision_point_id=request.decision_point_id,
            available_at_s=request.observed_until_s,
            observed_until_s=request.observed_until_s,
            scores={"eot": p_eot},
            emitted_at_ns=monotonic_ns(),
        )
