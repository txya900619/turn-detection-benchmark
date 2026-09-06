from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor
from voxtral_realtime.transformers import infer_asr_turn, load_mtp_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the pinned X2-Turn checkpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--delay-ms", type=int, default=480)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = load_mtp_checkpoint(
        args.model,
        device=args.device,
        dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        local_files_only=True,
    ).eval()
    result = infer_asr_turn(model, processor, args.audio, delay_ms=args.delay_ms)
    histogram = {
        label: sum(frame.label == label for frame in result.turn_frames)
        for label in ("idle", "noidle", "speaking", "turn_end", "backchannel", "uncertain")
    }
    assert result.frame_ms == 80
    assert 51 <= len(result.turn_frames) <= 55
    assert "hello can you tell me what the weather is like today" in result.transcript.lower()
    assert histogram["speaking"] > 0
    assert histogram["turn_end"] > 0
    for index, frame in enumerate(result.turn_frames):
        assert frame.start_ms == index * 80
        assert frame.end_ms == (index + 1) * 80
        assert abs(sum(frame.probabilities.values()) - 1.0) < 1e-5
    print(
        json.dumps(
            {
                "transcript": result.transcript,
                "frame_ms": result.frame_ms,
                "frame_count": len(result.turn_frames),
                "histogram": histogram,
                "tail": [
                    {
                        "start_ms": frame.start_ms,
                        "end_ms": frame.end_ms,
                        "label": frame.label,
                        "confidence": frame.confidence,
                    }
                    for frame in result.turn_frames[-10:]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
