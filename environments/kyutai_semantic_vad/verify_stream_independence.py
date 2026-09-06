"""Verify that Kyutai's two streams are processed independently.

The SID-Bench runner reports Kyutai as ``native_single`` on the grounds that its
TurnBench inference path concatenates the two speakers along the batch dimension
and never attends across them, so putting silence in the partner slot costs the
user's stream nothing. `docs/BENCHMARK_COVERAGE_AUDIT.md` turns that property
into the criterion that separates the models SID-Bench can evaluate from the
ones it cannot, so it should be checked rather than assumed.

This runs the same waveform in slot 1 against three different slot-2 contents
and asserts the slot-1 scores are identical every time.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_sidbench import _Audio, _Pair  # noqa: E402
from run_turnbench import _infer, _load_models  # noqa: E402

DURATION_S = 4.0
SAMPLE_RATE = 24_000
SEED = 0


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    samples: np.ndarray


def main() -> int:
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    info, mimi, generator, _ = _load_models(device, True)

    rng = np.random.default_rng(SEED)
    count = int(DURATION_S * SAMPLE_RATE)
    user = rng.standard_normal(count).astype(np.float32) * 0.1

    cases = (
        Case("silence", np.zeros(count, dtype=np.float32)),
        Case("noise", rng.standard_normal(count).astype(np.float32) * 0.1),
        Case("copy_of_user", user.copy()),
    )

    reference: np.ndarray | None = None
    for case in cases:
        pair = _Pair(
            conversation_id=f"independence-{case.name}",
            speaker_1_audio=_Audio(user, SAMPLE_RATE),
            speaker_2_audio=_Audio(case.samples, SAMPLE_RATE),
            duration_s=DURATION_S,
        )
        with torch.no_grad():
            first, _, _, _ = _infer(pair, info, mimi, generator, device)
        first = np.asarray(first, dtype=np.float32)
        if reference is None:
            reference = first
            print(f"{case.name:<14} frames={first.shape[0]} (reference)")
            continue
        difference = float(np.abs(first - reference).max())
        print(f"{case.name:<14} frames={first.shape[0]} max_abs_difference={difference:.3e}")
        if difference != 0.0:
            raise SystemExit(
                f"slot-1 scores changed when slot-2 content changed to {case.name!r}: "
                f"{difference}. The streams are NOT independent and the native_single "
                f"label on the SID-Bench run is wrong."
            )

    print("independent: partner-slot content does not affect the user stream's scores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
