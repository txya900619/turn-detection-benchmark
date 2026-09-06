# TurnBench Official Protocol Contract

Status: implementation research, captured 2026-09-04 UTC. This document covers
TurnBench, the English dual-channel benchmark from Sesame AI Labs. It is not the
contract for LiveKit `eot-bench` or SID-Bench.

The source of truth used here is the official TurnBench repository at commit
[`e02d547803fd0e45221afcaeb1cd24b61ae7568e`](https://github.com/SesameAILabs/turnbench/tree/e02d547803fd0e45221afcaeb1cd24b61ae7568e).
The repository has no release tags at the time of this audit, so every run must
record the full scorer commit rather than a package version. The accompanying
paper is [Jiang et al., *TurnBench: A Multi-Domain Benchmark for Turn-Taking
Dynamics in Spoken Dialogue*](https://arxiv.org/abs/2608.25218).

## 1. What TurnBench measures

TurnBench is a continuous-timeline, two-speaker benchmark with two independent
tasks:

- **EOT:** the time at which the floor leaves the speaker whose channel is being
  scored.
- **INT:** the time at which a listener takes the floor mid-turn, recorded on
  the interrupter's channel.

It contains 154 English dyadic conversations, approximately 30 hours, 106
speakers in 53 pairs, and six interaction types: Casual, Task-Oriented,
Instructional, Collaborative, Argumentative, and Narrative. The benchmark split
is speaker-disjoint and approximately 25/75: 38 labeled dev conversations and
116 label-withheld test conversations. The checked-in duration artifacts total
7.307 h dev and 22.865 h test
([dev durations](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/durations-dev.json),
[test durations](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/durations-test.json)).
A separate approximately 104-hour, speaker-disjoint training corpus uses the
same annotation protocol. The corpus and split design are described in
[Sections III-IV of the official
paper](https://arxiv.org/pdf/2608.25218) and the [official repository
README](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/README.md).

TurnBench is not a frame-accuracy benchmark. Models emit discrete committed
event times over the complete conversation clock; the official scorer evaluates
only gold positive windows, hard-negative spans, and task-specific excluded
intervals. Predictions elsewhere are score-neutral. This limitation is explicit
in the [official scoring caveats](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/README.md#caveats).

## 2. Dataset access, revisions, and physical layout

### 2.1 Official repositories

| Purpose | Hugging Face repository | Official revision behavior |
| --- | --- | --- |
| Labeled development set | [`mundo-ai/turn-benchmark-dev`](https://huggingface.co/datasets/mundo-ai/turn-benchmark-dev) | Official scorer pins `8fa18a24be51528a45397b35cbcaecd84202062b`. |
| Public label-withheld test audio | [`mundo-ai/turn-benchmark-test`](https://huggingface.co/datasets/mundo-ai/turn-benchmark-test) | Upstream does not pin a public-test revision. |
| Private labeled test data | `mundo-ai/turn-benchmark-test-golden` | Official scorer code pins `7b34876db0a883fdba5d1b6d67e9c3bf3303569a`; access is private. |
| Separate training set | [`otoearth/otoSpeech-full-duplex-turn-104h`](https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-turn-104h) | The benchmark repository does not pin it as an evaluation dependency. |

The evaluation repository IDs and dev/golden pins are constants in the official
[`turnbench/data.py`](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/data.py#L34-L52).
The checked-in split lists contain 38 dev IDs and 116 test IDs
([`dev.txt`](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/splits/dev.txt),
[`test.txt`](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/splits/test.txt)).

At the time of this audit, the Hugging Face `main` SHAs were
[`c29aa4e6422122a8dccbe23598016a089bea2121`](https://huggingface.co/datasets/mundo-ai/turn-benchmark-dev/tree/c29aa4e6422122a8dccbe23598016a089bea2121)
for dev and
[`0c12a77a0134449757e4fd4144913bfcff0c376d`](https://huggingface.co/datasets/mundo-ai/turn-benchmark-test/tree/0c12a77a0134449757e4fd4144913bfcff0c376d)
for public test. Dev `main` is therefore not the scorer's pinned revision. An
implementation must default to the official dev pin, and must resolve then
persist the exact public-test SHA when test is downloaded. Never record the
string `main` as a dataset revision.

### 2.2 Access conditions

Both public evaluation datasets are gated. The official flow is to request
access on each dataset page and authenticate using ambient Hugging Face
credentials; the scorer accepts `HF_TOKEN`, a prior HF login, or `HF_TOKEN` in
the upstream repository's `.env`. A source may instead be a local directory of
Parquet shards. See the official [submission setup](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/docs/SUBMISSION_FORMAT.md#score-on-the-dev-set)
and [loader contract](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/data.py#L113-L165).

For this repository, commands must additionally set:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
```

### 2.3 Row and column layout

The dev and public-test repositories are Parquet datasets with one row per
conversation. The dev repository contains three `dev-*.parquet` shards and the
public test contains eight `test-*.parquet` shards. The pinned dev dataset card
defines the following relevant columns
([dataset card at the official pin](https://huggingface.co/datasets/mundo-ai/turn-benchmark-dev/blob/8fa18a24be51528a45397b35cbcaecd84202062b/README.md)):

| Column | Meaning |
| --- | --- |
| `conversation_id: string` | Stable numeric-string conversation identifier. |
| `speaker_1_audio`, `speaker_2_audio` | Time-aligned mono, 48 kHz, full-quality 24-bit FLAC speaker channels. These are the benchmark inputs. |
| `speaker_1_audio_preview`, `speaker_2_audio_preview` | 64 kbps CBR Opus previews. They must not be used for predictions. |
| `speaker_{1,2}_annotation_{a,b,c}` | Three raw annotator tracks per speaker, each a list of events. Empty tracks are represented by `[]`. |
| `metadata` | Struct containing conversation type, anonymized actor IDs, and actor genders. |

Every annotation event is a struct with `start_s: float`, `end_s: float`,
`label: string`, and `text: string`. Times are seconds with millisecond
precision; `label` preserves the annotator's fine label and `text` may be empty.
All six annotation arrays exist but are always empty in the public test data, as
documented by the [official test dataset card](https://huggingface.co/datasets/mundo-ai/turn-benchmark-test).

The official Python loader projects only `conversation_id` and the six raw
annotation columns for scoring, adding the two full-quality audio columns only
for inference. It reads Parquet directly with PyArrow, wraps the chunked table,
casts audio to `Audio(decode=False)`, and decodes one channel lazily with
SoundFile. It deliberately avoids `datasets.load_dataset` because combining the
embedded binary audio chunks can overflow on the full test split. The exact
behavior is in [`turnbench/data.py`](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/data.py#L113-L259).
Hugging Face snapshots follow `HF_HOME`, but upstream's projected-gold cache is
separately hard-coded to `~/.cache/tt-benchmark/gold`; an isolated upstream
scorer process therefore needs both locations considered when auditing disk
state.

Implementation consequences:

1. Preserve the original `speaker_1`/`speaker_2` channel identity. Do not remap
   speakers to `user`/`agent` for scoring.
2. Assert the decoded channels have equal duration on the shared clock.
3. Run inference on the FLAC columns, never the preview columns.
4. Resample inside the model adapter and include resampling in runtime
   accounting; do not rewrite dataset audio.
5. Pin the scorer commit and dataset SHA independently in every result manifest.

## 3. Raw annotations and deterministic gold construction

No consensus-gold file is published as the normative artifact. The official
scorer rebuilds gold from the six raw tracks at scoring time, and scores are
only comparable when they use the same scorer commit. This rule is stated in
the [official evaluation methodology](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/README.md#gold-goldpy),
while [`gold.py`](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/gold.py)
is the executable source of truth.

### 3.1 Canonical taxonomy

The released fine labels map as follows. `Regular Turn` is accepted by the code
as a defensive alias but occurs in neither released split.

| Canonical label | Accepted fine labels |
| --- | --- |
| `Turn` | `Normal Turn`, `Regular Turn` (alias), `Strong Floor Hold`, `Bounded Response`, `Filler`, `Overlap` |
| `Laughter` | `Laughter` |
| `AwkwardSilence` | `Awkward Silence` |
| `Interruption` | `Floor-taking Competitive Interruption`, `Floor-taking Cooperative Interruption` |
| `NonFloorTakingInterruption` | `Non-floor Taking Competitive Interruption`, `Non-floor Taking Cooperative Interruption` |
| `Backchannel` | `Acknowledgement Backchannel`, `Continuer Backchannel`, `Reaction Backchannel` |
| `NonContent` | `Non-Speech Noise`, `Channel Bleed`, `Speech, Non-Linguistic` |

The mapping, including which labels count as INT negatives, is defined in
[`gold.py`](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/gold.py#L82-L140).
Unmapped labels are ignored for consensus. `Laughter` and `AwkwardSilence` are
canonicalized but the current EOT derivation uses `Turn` only; the upstream
source explicitly defers a `Turn ∪ Laughter` floor rule.

### 3.2 Two consensus views

Consensus is computed separately per speaker using three annotators `a/b/c`.
An event is accepted when at least two annotators agree on its canonical label
and both start and end ranges are within 0.2 seconds. Candidate events are first
matched by same-label onset within 0.5 seconds. Gold start and end are the
respective medians of the agreeing annotators, rounded to four decimals. A
dissenting event overlapping an accepted majority is resolved by the majority;
only genuinely unmatched, non-overlapping regions become excluded intervals.
The precise matching algorithm and constants are in
[`gold.py`](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/gold.py#L131-L278).

Two views must be built because the taxonomy is hierarchical:

- The **turn view** maps normal turn content, overlap, and floor-taking
  interruption fine labels to `Turn`. It answers only whether the span claims
  the floor and drives EOT gold and EOT exclusions.
- The **label view** keeps canonical labels distinct. It drives INT positives,
  INT negatives, and INT exclusions.

Do not reuse exclusions from one view for the other task.

### 3.3 EOT gold

The raw annotation is VAD-segmented, so a conversational turn can contain many
short `Turn` segments. For each speaker's consensus turn-view segment end:

- Ignore it if the end lies strictly inside another turn segment belonging to
  the same speaker.
- It is an EOT positive anchor when the other speaker starts before this
  speaker resumes, the other speaker is already speaking at the segment end,
  or this is the speaker's final segment.
- Otherwise it begins an EOT-negative mid-turn-pause span, ending no later than
  the same speaker's resumption.

If both speakers' next segments start at the same timestamp, the code treats
the event as self-resumption rather than handover because it tests
`next_other < next_self`, not `<=`.

The negative pause is truncated by the first contrary evidence: a turn-view or
label-view excluded interval on either channel, the paused speaker's own
non-turn vocalization, or an interruption event on the other channel. It is
dropped if no positive-length clean span remains. An other-speaker backchannel
does not truncate the pause because it supports the original speaker retaining
the floor. This construction is implemented in
[`build_conversation_events`](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/gold.py#L347-L436).

### 3.4 INT gold

- Positive: onset of a consensus, floor-taking `Interruption`, on the
  interrupter's channel.
- Negative span: full extent of a consensus `Backchannel` or `NonContent`
  event, on the vocalizing speaker's channel.
- Excluded: label-view no-majority regions plus consensus
  `NonFloorTakingInterruption` extents.
- `Laughter` is neither positive nor negative in the current INT task.

These sets are constructed in the same official
[`build_conversation_events`](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/gold.py#L404-L436).
For a useful parity target, the official scorer reports that the 38-conversation
dev split produces 1,904 EOT positives, 1,063 EOT negatives, 347 INT positives,
3,733 INT negatives, 545 EOT exclusions, and 902 INT exclusions
([methodology counts](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/README.md#floor-construction-also-goldpy)).
Across the full 154-conversation corpus, the paper reports 8,197 EOT positives,
4,254 EOT negatives, 1,151 INT positives, and 17,511 INT negatives
([official paper, Section IV](https://arxiv.org/pdf/2608.25218)).

## 4. Required prediction artifact

The official submission is a single strict JSON document:

```json
{
  "schema_version": 1,
  "predictions": [
    {
      "conversation_id": "20",
      "speaker_1": {
        "eot": [12.84, 58.43],
        "interruption": [31.20]
      },
      "speaker_2": {
        "eot": [27.10],
        "interruption": []
      }
    }
  ]
}
```

The schema is defined in the official
[`SUBMISSION_FORMAT.md`](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/docs/SUBMISSION_FORMAT.md#format)
and enforced by
[`submission.py`](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/submission.py).
Validation rules are:

- `schema_version` is exactly integer `1`; unknown fields at every level are
  rejected.
- Every dataset conversation appears exactly once. Missing, duplicate, or
  unknown IDs are errors. A model that finds no events must still emit the
  conversation with four empty lists.
- `speaker_1` and `speaker_2` refer directly to the corresponding audio
  columns.
- `eot` and `interruption` values are finite, non-negative float seconds on the
  shared conversation clock, not frame indices.
- Each event list is strictly increasing; duplicates are invalid.
- No time may exceed the audio duration. A time exactly at the duration is
  accepted by the implementation.

This JSON is the official scorer input. This repository may additionally store
row-wise Parquet predictions for unified reporting, but it must be able to
losslessly export the exact JSON above and validate it with the pinned official
scorer.
If a model does not support one task, its official JSON still requires an empty
list for that task; the result manifest must separately mark the task
unsupported so an empty list is not misrepresented as an evaluated detector.

## 5. Causal timestamp rule

A submitted timestamp is a **commit time**, meaning the time by which every
audio sample used for the decision has been heard. An output at time `t` may use
audio only through `t`. If a model internally identifies an event at 0.7 s but
uses audio through 1.0 s, it must submit 1.0 s. Algorithmic lookahead is thereby
charged as conversational latency. This is a term of submission documented in
the official [causality rule](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/docs/SUBMISSION_FORMAT.md#causality-is-a-term-of-submission).

The scorer validates ordering and duration but cannot infer which future audio
the model consumed. Causality is affirmed, not mechanically enforced
([official caveat](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/README.md#caveats)).
Our runner must therefore enforce `observed_until_s <= commit_time_s` at the
adapter boundary and add any declared algorithmic lookahead before exporting
the event.

For frame traces, an official frame `i` covers
`[i / frame_rate_hz, (i + 1) / frame_rate_hz)` and its committed timestamp is
the **frame end**, `(i + 1) / frame_rate_hz`, not its start or center
([official probability-grid code](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/sweep.py#L42-L104)).

## 6. Exact scoring algorithm

The constants are:

```text
tau_pre  = 0.25 seconds
tau_max  = 3.00 seconds
```

They are defined in
[`gold.py`](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/gold.py#L142-L148),
and the authoritative matcher is
[`score_task`](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/score.py#L180-L248).

For each task, speaker channels are scored independently, then all counts and TP
latencies are pooled.

1. Remove predictions falling in an excluded interval for that speaker. Both
   interval endpoints are inclusive.
2. Sort positive anchors by time.
3. For a gold anchor at `t`, form the inclusive matching window
   `[t - 0.25, min(t + 3.0, next_same_speaker_same_task_anchor)]`.
4. Claim the earliest as-yet-unclaimed prediction in that window. A claimed
   prediction is one TP with signed latency `(t_pred - t_gold) * 1000`; if none
   exists, record one FN.
5. Mark the complete positive window as FP-exempt. Extra predictions inside a
   positive window are score-neutral, even if a negative span overlaps it.
6. For each negative span `[start, end]`, look for an unclaimed, non-exempt
   prediction in the inclusive interval `[start - 0.25, end]`. Any number of
   fires produces one FP; no fire produces one TN.
7. Predictions outside every positive window and negative span are ignored.

One prediction cannot satisfy two anchors. A scorer-level edge case worth
preserving exactly is that the upper window endpoint is inclusive; therefore a
prediction exactly equal to the next same-speaker anchor is visible to the
earlier anchor first. This follows from the official implementation's
`<= window_end` comparison and time-ordered claiming
([ledger implementation](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/score.py#L146-L177)).

### 6.1 Metrics and aggregation

Each task reports:

```text
recall  = TP / (TP + FN)
fp_rate = FP / (FP + TN)
latency = p10 / p50 / p90 of signed TP latencies in milliseconds
```

Counts are micro-aggregated over all conversations and speakers; latency
percentiles are calculated over the pooled matched-TP latency list using
NumPy's percentile implementation. If the relevant denominator or TP latency
list is empty, the scorer returns NaN. See the official
[`TaskScore` and aggregation code](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/score.py#L89-L120)
and [`score_submission`](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/score.py#L272-L323).

`fp_rate` is the fraction of sampled hard-negative spans that receive a false
fire. It is not false alarms per hour, and the benchmark does not measure
arbitrary fires in otherwise unscored timeline regions.

## 7. Threshold selection, qualification, and ranking

The prediction JSON contains no probabilities or threshold. It must contain
the single operating point that would cause the deployed system to act. The
official scorer never sweeps the submitted JSON
([submission methodology](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/docs/SUBMISSION_FORMAT.md#scoring)).

For a model with continuous outputs, the official baseline procedure is:

1. Generate full dev-set probability traces.
2. Sweep thresholds on **dev independently for EOT and INT**.
3. Use the common commit rule described below.
4. For each task, select the highest-recall point satisfying dev
   `fp_rate <= 0.10`.
5. Freeze both thresholds and generate dev/test prediction JSON files.
6. Never choose or revise the threshold using test labels.

Test qualification is separate for EOT and INT. A task qualifies if test
`fp_rate <= 0.15`, then qualifying systems are ranked by descending test recall.
Over-ceiling systems remain displayed but rank below every qualifier. A system
can qualify for one task but not the other. The official rules are in the
[baseline author guide](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/baselines/README.md#rules)
and [submission policy](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/docs/SUBMISSION_FORMAT.md#submission-policy).

### 7.1 Optional continuous probability files

For reproducible dev sweeps, the upstream project defines one JSON file per
task:

```json
{
  "schema_version": 1,
  "task": "eot",
  "frame_rate_hz": 25.0,
  "probs": [
    {
      "conversation_id": "20",
      "speaker_1": {"prob": [0.01, 0.04, 0.93]},
      "speaker_2": {"prob": [0.00, 0.02, 0.10]}
    }
  ]
}
```

`task` is exactly `eot` or `int`. Probabilities must be finite and in `[0,1]`.
There must be exactly one entry per dev conversation and exactly
`floor(duration_s * frame_rate_hz)` values per speaker. The complete schema and
validator are in
[`sweep.py`](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/sweep.py#L39-L149).
There are no test probability submissions; test receives only the frozen,
committed `predictions-test.json`.

The standard threshold candidates are the union of:

- 512 quantiles of all pooled, nonzero dev scores across conversations and
  speakers; and
- the uniform grid `0.01, 0.02, ..., 0.99`.

Candidates are deduplicated and sorted. For each threshold, the common commit
rule emits an event on a strict rising edge from `<= theta` to `> theta`, uses a
2.0-second refractory period, and timestamps the event at the frame end. Only a
committed event advances the refractory; a suppressed rising edge does not.
These details are in the official
[`candidate_thetas` and `commit_events`](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/sweep.py#L152-L219).
`operating_point` applies Python `max(..., key=recall)` to the ascending
qualifying rows, so an exact recall tie keeps the lower threshold; it returns
`None` when no candidate meets the budget
([official selector](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/sweep.py#L291-L305)).

The approximately 1 GB of official baseline trace JSONs is stored separately in
[`freemanjiang/turnbench-baseline-probs`](https://huggingface.co/datasets/freemanjiang/turnbench-baseline-probs)
and pinned by upstream to revision
`e3cd4caa2a45b85ef4d6714d79d5c7c40471ae79`
([`turnbench/probs.py`](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/turnbench/probs.py#L27-L53)).

## 8. Model-adapter integration seam

The official benchmark intentionally makes model execution independent from
scoring. Each baseline loads the same conversation dataset, runs arbitrary
model-specific inference, maps native outputs to EOT/INT signals per speaker,
and emits the strict prediction JSON. The scorer consumes only discrete event
times. The [baseline author guide](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/baselines/README.md)
and [RMS VAD reference predictor](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/baselines/rms_vad/predict.py)
are the minimal reference.

For this repository, the adapter boundary should therefore be:

```text
pinned TurnBench Conversation
  -> two aligned native audio channels
  -> model adapter (causal inference, model-specific resampling/context)
  -> per-speaker EOT/INT score trace or committed events
  -> frozen policy (threshold/refractory/mapping)
  -> official predictions schema
  -> pinned official scorer
```

The benchmark's published mappings show why signal provenance must be persisted:

- VAP uses `1 - p_now[speaker]` for EOT and `p_now[speaker]` for INT at 50 Hz
  ([official VAP adapter](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/baselines/vap/predict.py)).
- ESPnet's mixed-mono, five-class model uses `P(Turn-change)` for EOT and
  `P(Interruption)` for INT, then attributes events back to a speaker using
  channel energy
  ([official ESPnet adapter notes](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/baselines/espnet_turntaking/README.md)).
- SmartTurn v3 has no native INT head; the baseline treats its completion score
  as EOT and `1 - completion` on the prospective interrupter's channel as a
  derived INT signal
  ([official SmartTurn adapter](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/baselines/smart_turn_v3/predict.py)).
- The current `dualturn` directory is explicitly a TODO/stub with model code but
  no official submission, so it cannot be used as an official reference score
  yet
  ([official baseline roster](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/baselines/README.md#roster)).

Every TurnBench manifest should record, at minimum:

- model and immutable model/checkpoint revision;
- source sample rate and model-native sample rate;
- single-channel, reconstructed-mix, or native-dual-channel mode;
- native versus derived signal type separately for EOT and INT;
- exact native-output mapping for each task;
- frame rate, context window, algorithmic lookahead, and commit-time rule;
- EOT and INT thresholds, 2 s refractory if the common rule is used, and dev
  calibration source;
- dataset repo and resolved SHA;
- official scorer repo and full commit SHA.

## 9. Recommended parity and acceptance tests

Before reporting any new model result:

1. **Dataset/gold parity:** at the pinned dev revision, reproduce the official
   event-set totals: EOT `1904+/1063-`, INT `347+/3733-`, exclusions
   `545 EOT/902 INT`.
2. **Schema parity:** run the exported JSON through the pinned upstream
   submission validator and reject unknown keys, missing IDs, duplicate times,
   non-finite values, unsorted values, and out-of-duration events.
3. **Scorer parity:** use synthetic fixtures covering early tolerance, the 3 s
   deadline, adjacent same-speaker anchors, channel isolation, excluded spans,
   overlapping positive/negative regions, and bursts in one negative span. The
   upstream tests enumerate these cases in
   [`tests/score_test.py`](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/tests/score_test.py).
4. **Oracle parity:** exporting official gold anchors as predictions must yield
   recall 1.0 and `fp_rate` 0.0; the runnable upstream example is
   [`baselines/oracle_annotator/predict.py`](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/baselines/oracle_annotator/predict.py).
5. **Baseline parity:** reproduce at least one official committed dev artifact
   using the same scorer commit before trusting a model-specific adapter.
6. **Causality:** fail locally if an event's observed-audio horizon exceeds its
   exported commit time. The official scorer cannot perform this check.
7. **Coverage and metadata:** treat limited-conversation runs as diagnostics
   only because the formal validator requires complete split coverage.
   Per-conversation-type reporting additionally needs `metadata`, which the
   official scoring-only projection omits and must be fetched separately.

The official commands are:

```bash
uv sync
uv run python -m turnbench.score predictions.json
uv run python -m turnbench.check baselines/<name>
uv run python -m turnbench.sweep baselines/<name>/probs-eot.json
uv run python -m turnbench.sweep baselines/<name>/probs-int.json
```

The upstream project's dependency versions at the audited commit are pinned in
its [`pyproject.toml`](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/pyproject.toml).
To prevent its environment from conflicting with model stacks, run the pinned
official scorer in its own `uv` environment or subprocess and treat it as a
read-only compatibility oracle.

## 10. Licenses and redistribution

The official TurnBench code and scorer are MIT-licensed
([repository license](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/LICENSE)).
The audio, transcripts, annotations, and related dataset material use Mundo
AI's **Dataset Public License v1.0**, not MIT. Its authoritative terms require
attribution, permit licensed sharing/use only for noncommercial purposes,
prohibit voice cloning, and impose license/voice-cloning conditions on
downstream redistribution
([dev dataset license at the official pin](https://huggingface.co/datasets/mundo-ai/turn-benchmark-dev/blob/8fa18a24be51528a45397b35cbcaecd84202062b/LICENSE)).

Therefore this repository may contain downloader code, adapters, manifests,
derived aggregate metrics, and scorer-compatible predictions as permitted, but
must not commit or redistribute TurnBench audio/annotations. Any release of
derived row-level artifacts should receive a separate license review rather
than assuming the scorer's MIT license applies to data-derived content.

The separate training corpus has its own
[`oto Speech License v1.0`](https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-turn-104h/blob/46f520297f434edf804389f82f9075a59d2f8268/LICENSE.md),
including noncommercial-research and voice-identity restrictions; it must be
tracked as a distinct license rather than inherited from the evaluation data.

## 11. Current blockers and unresolved upstream details

- **Dev access resolved:** access to `mundo-ai/turn-benchmark-dev` was verified
  on 2026-09-04. The pinned 38-conversation revision reproduced all official
  gold totals and the official oracle predictions scored perfectly. Access to
  the public test repository must still be checked separately if required.
- **Public-test revision is not pinned upstream:** official code pins dev and
  private golden test, but not the public label-withheld test. Our downloader
  must resolve a concrete SHA and record it at run time.
- **Scores are scorer-version-specific:** gold is derived by code, so copying
  consensus events without recording `scorer_sha` is insufficient for
  reproducibility.
- **Event-anchored blind regions:** a strong TurnBench result does not establish
  a low global false-alarm-per-hour rate; deployment robustness needs a separate
  timeline-wide metric or suite.

Official submissions currently require both dev and test prediction files plus
model and organization names, sent to the contact described in the
[official submission policy](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/docs/SUBMISSION_FORMAT.md#submission-policy).
