from __future__ import annotations

import numpy as np
import pytest

from turn_eval.core.types import AudioData
from turn_eval.models.rms_vad import RMSVADAdapter


def test_rms_vad_matches_turnbench_edge_and_timestamp_rule() -> None:
    # Four complete 20 ms windows plus a trailing partial window. The complete
    # windows are silence, speech, speech, silence.
    samples = np.concatenate(
        (
            np.zeros(20),
            np.full(20, 0.02),
            np.full(20, -0.02),
            np.zeros(20),
            np.full(10, 0.02),
        )
    ).astype(np.float32)
    adapter = RMSVADAdapter(native_sample_rate=1_000)

    prediction = adapter.predict(AudioData(samples, 1_000))

    assert prediction.interruption == pytest.approx((0.04,))
    assert prediction.eot == pytest.approx((0.08,))


def test_rms_vad_uses_strict_threshold_and_ignores_initial_state() -> None:
    samples = np.concatenate((np.full(20, 0.01), np.full(20, 0.02))).astype(np.float32)
    adapter = RMSVADAdapter(native_sample_rate=1_000)

    prediction = adapter.predict(AudioData(samples, 1_000))

    assert prediction.interruption == pytest.approx((0.04,))
    assert prediction.eot == ()


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("window_ms", 0.0, "window_ms"),
        ("threshold", -0.1, "threshold"),
        ("native_sample_rate", 0, "native_sample_rate"),
    ],
)
def test_rms_vad_rejects_invalid_configuration(keyword, value, message) -> None:
    with pytest.raises(ValueError, match=message):
        RMSVADAdapter(**{keyword: value})
