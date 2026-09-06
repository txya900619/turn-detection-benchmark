# SID-Bench Protocol

Status: pinned implementation contract  
Dataset revision: `6eb13b573ad588646ce0de513d4255e55caf858b`  
Official repository/scorer revision: `0a9e5001e1fe534b3eec4f76c4455ae4bff9ccf1`

## Scope and released inputs

SID-Bench releases 1,600 English clips, 1,600 Mandarin Chinese clips, and a
separate 500-clip silence/noise robustness set. Each language split contains
500 interruptions whose intent begins immediately, 600 whose intent appears
after an initial backchannel, and 500 non-break utterances.

The released WAV files contain only the interrupting user's mono channel. They
do not contain the agent/other-speaker channel or a shared source-conversation
clock. Runs therefore use the `single_channel_interruption_timeline` protocol.
A dual-channel model must be labeled `single_channel_fallback`, and a signal
constructed from VAD/FVAD must be labeled `derived`. SID-Bench results alone do
not establish full-agent stop latency.

The transcript includes the gold `<break>` marker and MUST NOT be passed to a
model. Inference consumes causal audio prefixes and records the first committed
interruption time. A no-trigger result is represented by `null` internally and
by `total_nonbreak=true, break_time=-1` in the official JSONL schema.

## Frozen splits and calibration

All three public splits are test sets. No threshold, sustain window, mapping,
or retry choice may be selected from their labels. A model run must use a
native fixed policy or a policy frozen on an external named calibration set.
EN, ZH, and Noise/Silence are scored and reported separately. Any cross-split
average is secondary and must state its weighting.

The pinned DualTurn fallback policy uses the model's `fvad[0,0]` user
future-activity output for the 0-240 ms horizon with `user=CH0, silence=CH1`.
Its threshold, `0.9896328579889585`, was frozen on TurnBench EN dev at the
derived-interruption operating point with FPR at or below 0.10. It commits at
the end of the first complete 80 ms frame meeting the threshold; a final
partial frame is not zero-padded because that would expose an end-of-file cue
that does not exist in a continuing live stream. Reports must retain both
`single_channel_fallback` and `derived-fvad` labels.

## Official metric behavior

The repository reproduces the released `test.ipynb` case order and 50 ms
tolerance:

- a prediction within ±50 ms of a positive break receives zero penalty;
- a late positive detection receives `predicted_time - break_time` penalty;
- an early positive detection is a false interruption and receives the full
  annotated clip duration;
- a missed positive receives `duration - break_time`;
- a correct non-break receives zero;
- a false trigger on a non-break receives the full annotated duration.

Official FIR divides notebook false interruptions by all evaluated examples,
not by only non-break examples. Official mean IRL averages only positive late
delays greater than the 50 ms tolerance. APT averages the case penalties over
all examples. Repository diagnostics additionally report interruption recall,
latency p50/p90/p95 over correct interruptions (within-tolerance values become
zero), and non-break false-positive rate.

The notebook checks the ±50 ms boundary before checking `total_nonbreak`.
Consequently, a trigger within 50 ms of a non-break clip's annotated end gets
zero official penalty. This implementation preserves that behavior for numeric
parity and separately reports `strict_paper_fir`, which counts those triggers as
false interruptions. The official README's FIR denominator and absolute-IRL
wording conflict with both the paper equation and notebook; reports must expose
the denominator and IRL population explicitly.

## Dataset integrity and license

Pinned annotation hashes are:

- EN: `ceb18d41d2e014e225b3d729bd3a380514cb1fec911c9747325094138a104544`
- ZH: `b5faadcc6d1f2ddbc87b93765592f44056bcac6b60f8ce5ff952dee16b9f178f`
- Noise: `d5a73649653e40e51cd7b96cf181df71f379378aaeaec190a9cde786aa6979cf`

The public Hugging Face dataset has no license tag, and the official repository
contains no `LICENSE` file even though its README asks users to consult one.
Some audio is attributed to vendor data and Switchboard. The harness records
the license as `unspecified; source-corpus terms apply`; public availability
must not be interpreted as redistribution or commercial-use permission.

The annotations also contain a small number of transcripts with multiple
`<break>` tokens and at least one cross-language-looking clip in the EN split.
The pinned labels and first `break_time` remain authoritative; the transcript is
retained only as provenance and is never an inference input.

The EN annotated duration differs from its WAV-header duration by more than
50 ms for 521 of 1,600 clips (maximum absolute delta 124 ms), while the pinned
ZH and Noise/Silence durations match exactly. Official FIR/APT scoring uses the
annotated duration for parity. Audio processing and RTF use the actual WAV-header
duration; both sources and the discrepancy summary are recorded in metadata.
