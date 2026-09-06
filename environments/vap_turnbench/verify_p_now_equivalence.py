"""Verify that the forward-based p_now path reproduces upstream ``VapGPT.probs``.

``VapGPT.probs`` computes ``p_now`` and then, unconditionally, a training loss
whose label extractor requires at least ``sum(bin_frames)`` future frames. Its
``vad`` parameter cannot switch that off, because the body reassigns
``vad = out["vad"].sigmoid()`` before the branch that tests it. On SID-Bench,
where 746 of the 1,600 English clips are shorter than about 2.02 seconds, that
label computation raises and no probability can be obtained at all.

``p_now`` itself does not depend on those labels. This script asserts that
recomputing it from the model forward pass, using only upstream's own public
objective, is numerically identical to what ``probs`` returns on inputs long
enough for ``probs`` to run. Run it before trusting a VapGPT SID-Bench artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_sidbench import _p_now_from_forward  # noqa: E402
from run_turnbench import SAMPLE_RATE, _load_model  # noqa: E402

# Long enough that upstream ``probs`` can build its projection windows.
DURATION_S = 10.0
SEED = 0


def main() -> int:
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    model, _, _ = _load_model(device, local_files_only=True)

    generator = torch.Generator().manual_seed(SEED)
    samples = int(DURATION_S * SAMPLE_RATE)
    waveform = torch.randn(1, 2, samples, generator=generator).clamp(-1.0, 1.0).to(device)

    with torch.no_grad():
        upstream = model.probs(waveform)["p_now"][0].cpu().numpy()
    derived = _p_now_from_forward(model, waveform)

    if upstream.shape != derived.shape:
        raise SystemExit(f"shape mismatch: upstream {upstream.shape} vs derived {derived.shape}")
    maximum_difference = float(abs(upstream - derived).max())
    print(f"frames={upstream.shape[0]} max_abs_difference={maximum_difference:.3e}")
    if maximum_difference != 0.0:
        raise SystemExit(f"p_now paths disagree by {maximum_difference}")
    print("identical: the forward-based p_now path reproduces upstream VapGPT.probs exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
