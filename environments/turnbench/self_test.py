from __future__ import annotations

import json
from dataclasses import asdict

from turnbench.data import ANNOTATORS, Conversation
from turnbench.score import score_conversation
from turnbench.submission import ConversationPrediction, SpeakerEvents


def unanimous(*events):
    return {annotator: list(events) for annotator in ANNOTATORS}


def main() -> int:
    speaker_1 = unanimous(
        (1.0, 2.0, "Normal Turn", ""),
        (2.3, 3.0, "Normal Turn", ""),
    )
    speaker_2 = unanimous(
        (1.5, 1.7, "Acknowledgement Backchannel", ""),
        (4.5, 4.8, "Floor-taking Competitive Interruption", ""),
        (3.5, 5.0, "Normal Turn", ""),
    )
    annotations = {
        (speaker, annotator): tracks[annotator]
        for speaker, tracks in ((1, speaker_1), (2, speaker_2))
        for annotator in ANNOTATORS
    }
    conversation = Conversation(
        conversation_id="1",
        duration_s=10.0,
        annotations=annotations,
        audio_bytes={},
    )
    prediction = ConversationPrediction(
        conversation_id="1",
        speaker_1=SpeakerEvents(eot=[3.0], interruption=[]),
        speaker_2=SpeakerEvents(eot=[5.0], interruption=[4.5]),
    )
    scores = score_conversation(prediction, conversation)

    assert (scores.task_eot.tp, scores.task_eot.fn) == (2, 0)
    assert (scores.task_eot.fp, scores.task_eot.tn) == (0, 1)
    assert scores.task_eot.latencies_ms == [0.0, 0.0]
    assert (scores.task_int.tp, scores.task_int.fn) == (1, 0)
    assert (scores.task_int.fp, scores.task_int.tn) == (0, 1)
    assert scores.task_int.latencies_ms == [0.0]

    print(
        json.dumps(
            {
                "scorer_revision": "e02d547803fd0e45221afcaeb1cd24b61ae7568e",
                "oracle_parity": True,
                "eot": asdict(scores.task_eot),
                "interruption": asdict(scores.task_int),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
