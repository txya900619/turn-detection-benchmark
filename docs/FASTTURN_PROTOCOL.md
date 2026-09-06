# FastTurn Test Set Protocol Contract

Status: implementation research, captured 2026-09-04 UTC. This document covers
the public Chinese FastTurn Test Set. It distinguishes facts in the official
paper/repositories from policies that `turn-eval` must add because the public
release does not include an executable scorer or a complete evaluation spec.

The canonical project repository is [`ASLP-lab/FastTurn` at commit
`0c1c03dc03c9cc0bf7a743a066fc99107c8a5c59`](https://github.com/ASLP-lab/FastTurn/tree/0c1c03dc03c9cc0bf7a743a066fc99107c8a5c59).
It has no release tags and contains only a README and two figures; it does not
publish model code, a dataset loader, an evaluation script, or reference
predictions. The accompanying source is [Wang et al., *FastTurn: Unifying
Acoustic and Streaming Semantic Cues for Low-Latency and Robust Turn Detection*,
arXiv:2604.01897v6](https://arxiv.org/abs/2604.01897v6). The public data are in
[`ASLP-lab/FastTurn-Testset`](https://huggingface.co/datasets/ASLP-lab/FastTurn-Testset).

## 1. Benchmark identity and scope

FastTurn is a **Chinese isolated-segment, four-class turn-state classification
test set**. Its labels are `complete`, `incomplete`, `backchannel`, and `wait`.
The official project describes the real portion as segments selected from
expert-annotated, dual-channel, human-to-human dialogue and the related
SmoothConv project states that FastTurn was constructed from a subset of
SmoothConv. The source conversations contain realistic overlap, pause,
backchannel, turn-transition, noise, and prosodic phenomena
([FastTurn test-set description](https://github.com/ASLP-lab/FastTurn/blob/0c1c03dc03c9cc0bf7a743a066fc99107c8a5c59/README.md#fastturn-test-set),
[SmoothConv relationship](https://github.com/qualialabsAI/SmoothConv-DuplexConv/blob/eb4297fac4f9d6f182740d79bea2df2664b4137e/README.md#fastturn-test-set)).

The **published FastTurn artifact is not itself dual-channel**. It contains one
mono WAV clip and one class label per row. It provides no paired other-speaker
channel, conversation ID, speaker ID, original timeline, start/end timestamps,
previous turns, emotion field, overlap field, or full source annotation. The
project's richer annotation description therefore characterizes the collection
source, not the schema available to a public benchmark runner. FastTurn must not
be represented as a native dual-channel or continuous-timeline evaluation.

Consequences:

- The evaluation unit is one complete released audio clip.
- `user_audio` is the clip and `agent_audio` is absent.
- This dataset can score segment-level state discrimination, but cannot by
  itself score causal event timing, interruption onset, conversational
  endpointing latency, or use of the other speaker's concurrent audio.
- There is one public test collection and no published train/dev/test partition
  inside it. Any calibration split created by `turn-eval` is a repository-defined
  derivative and must be persisted with its seed and example IDs.

## 2. Pinned dataset artifact

The audited Hugging Face revision is
[`4194458681149e2947cdb062ad61292cf9705a59`](https://huggingface.co/datasets/ASLP-lab/FastTurn-Testset/tree/4194458681149e2947cdb062ad61292cf9705a59),
which was also `main` on 2026-09-04. Every run must record the resolved full SHA,
not the string `main`. The revision contains exactly three Hub files:

| Hub file | Purpose |
| --- | --- |
| `.gitattributes` | Hugging Face/LFS attributes. |
| `README.md` | A 31-byte YAML front matter declaring only `license: apache-2.0`. |
| `fasttrun-testset.tar.gz` | The data archive; note the official filename typo `fasttrun`. |

The revision metadata and its three-file inventory are available from the
[official Hugging Face API](https://huggingface.co/api/datasets/ASLP-lab/FastTurn-Testset/revision/4194458681149e2947cdb062ad61292cf9705a59).
The archive is 3,766,503,649 bytes and its audited SHA-256 is:

```text
888cf6bfcc76df7a034060bbb11a7a1c2053c63901ab5e7f5adca48da14c2795
```

It expands to exactly this topology; there are no additional split manifests,
licenses, annotation files, or scorer files inside the archive:

```text
fasttrun-testset/
├── data.jsonl
└── wavs/
    └── {key}.wav  # 22,432 files
```

`data.jsonl` has 22,432 JSON objects, 4,232,872 bytes, and audited SHA-256
`3e0825586eb7ed36c262e3bb6feb3c9eb2d93b350fb874aad8e1e91c3723ded2`.
All `key` and `wav` values are unique, and every row satisfies
`wav == "wavs/" + key + ".wav"`.

For this repository, acquisition commands must set the required cache root and
pin the revision:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
hf download ASLP-lab/FastTurn-Testset \
  --repo-type dataset \
  --revision 4194458681149e2947cdb062ad61292cf9705a59
```

## 3. Released row schema

Every JSONL row has exactly five string fields:

```json
{
  "key": "10b50f4f-6990-4771-b6e2-fae3dfc45a82",
  "wav": "wavs/10b50f4f-6990-4771-b6e2-fae3dfc45a82.wav",
  "txt": "\u55ef<complete>",
  "state": "complete",
  "lang": "zh"
}
```

| Field | Contract |
| --- | --- |
| `key` | Unique example identifier. Treat as an opaque string; wait IDs and real-data IDs use different shapes. |
| `wav` | POSIX path relative to the extracted `fasttrun-testset/` root. |
| `txt` | Human/constructed annotated transcript, not a clean streaming-ASR transcript. |
| `state` | Gold label; exactly one of the four lowercase values below. |
| `lang` | Always `zh` in this snapshot. |

For every real-data row, `txt` ends in a target-bearing annotation matching
`state`: `<complete>`, `<incomplete>`, or `<backchannel>`. The transcript may
also contain annotations such as `<pause>`, breaths, noises, laughter, and some
inconsistently spelled tags. Wait transcripts contain no terminal `<wait>` tag.
Passing the raw `txt` value to a model would therefore leak the answer for all
21,432 real examples.

Normative loader policy:

1. Preserve raw `txt` in the dataset record for provenance only.
2. Never expose the terminal target tag to an evaluated model.
3. The default audio benchmark provides **audio only**. A transcript-requiring
   model must generate its transcript causally from audio.
4. If a supplied transcript is evaluated, strip the terminal target tag and
   mark the mode `oracle-transcript`; do not rank it with audio/ASR modes.
5. Whether other inline annotation tags are retained or stripped must be
   explicit. The production-comparable view strips all annotation tags because
   a normal ASR system would not produce them.

These safeguards are `turn-eval` policies derived from inspection of the pinned
artifact; the upstream repositories do not warn about label leakage.

## 4. Labels, sources, audio, and counts

The official counts and rounded durations are reported identically in the
[FastTurn README](https://github.com/ASLP-lab/FastTurn/blob/0c1c03dc03c9cc0bf7a743a066fc99107c8a5c59/README.md#statistics)
and [paper Table 1](https://arxiv.org/html/2604.01897v6#S2.SS3). A complete
header scan of the pinned archive gives the exact durations and encoding below.

| `state` | Source | Samples | Official hours | Audited hours | Released WAV encoding |
| --- | --- | ---: | ---: | ---: | --- |
| `complete` | Real conversation | 14,709 | 9.64 | 9.643338 | PCM signed 16-bit LE, mono, 48 kHz |
| `incomplete` | Real conversation | 3,643 | 2.15 | 2.145321 | PCM signed 16-bit LE, mono, 48 kHz |
| `backchannel` | Real conversation | 3,080 | 0.42 | 0.420014 | PCM signed 16-bit LE, mono, 48 kHz |
| `wait` | Synthetic | 1,000 | 0.71 | 0.715803 | PCM signed 16-bit LE, mono, 22.05 kHz |
| **Total** | Mixed | **22,432** | **12.92** | **12.924476** | Mixed native sample rates |

The official source states that all 1,000 wait utterances were added because
wait behavior is rare in natural conversation: DeepSeek V3 generated their
text and IndexTTS2 synthesized their audio
([paper Section 2.3](https://arxiv.org/html/2604.01897v6#S2.SS3)). The remaining
21,432 segments are real-world audio.

The FastTurn sources do not publish a formal annotation manual defining the
four states. FastTurn says its turn-detection training data include the Easy
Turn training set ([FastTurn Section
3.1](https://arxiv.org/html/2604.01897v6#S3.SS1)), whose official paper defines
the same vocabulary ([Easy Turn Section
2.1](https://arxiv.org/html/2509.23938#S2.SS1)). Combined with the released
FastTurn examples, the safest operational reading is:

- `complete`: the released segment ends at a complete user turn.
- `incomplete`: the segment ends while the user's turn remains incomplete.
- `backchannel`: brief feedback/acknowledgement that is not a normal floor claim.
- `wait`: an explicit request for the interlocutor to pause, stop speaking, or
  delay responding; it is not merely a hesitation or mid-turn silence.

The last four descriptions are inherited operational semantics, not a published
FastTurn annotation guideline. Results should retain the exact lowercase labels
and avoid inventing more detailed subtypes.

### 4.1 Required confound reporting

The wait label is perfectly confounded with synthetic source and with a
different sample rate in this release: every wait clip is 22.05 kHz, while all
real classes are 48 kHz. Resampling does not remove the different source
bandwidth or TTS artifacts. Consequently:

- report four-class scores with a prominent `wait_source=synthetic` warning;
- record original sample rate before adapter resampling;
- do not claim wait performance generalizes to real user requests;
- in addition to the four-class result, report metrics on the three real-data
  states, and optionally report wait separately;
- never let file naming, original sample rate, transcript tag, or source-specific
  metadata enter model features.

## 5. Official evaluation protocol: what exists

The paper defines three metrics from classification outcomes
([Section 3.3](https://arxiv.org/html/2604.01897v6#S3.SS3)):

```text
Accuracy   = (TP + TN) / (TP + TN + FP + FN)
Miss Rate  = FN / (TP + FN)
False Alarm Rate = FP / (FP + TN)
```

Table 2 reports all three metrics separately under each state. This presentation
is consistent with treating each state as positive against the other states,
but the paper does not explicitly specify the one-vs-rest reduction, class
weighting, prediction serialization, batching, or zero-denominator behavior.
No confusion matrices or raw predictions are published. `turn-eval` may adopt
the standard one-vs-rest construction below, but must label it as the repository
interpretation rather than an official executable oracle:

```text
TP_c = count(gold=c and prediction=c)
FN_c = count(gold=c and prediction!=c)
FP_c = count(gold!=c and prediction=c)
TN_c = count(gold!=c and prediction!=c)
```

The published Table 2 values are not a safe numerical parity target. Some
metric triples are inconsistent with the paper's formulas and the released
class prevalence under a single full-test one-vs-rest interpretation. For
example, FastTurn-Unified's Complete miss rate `14.53%` and false-alarm rate
`14.92%` imply `85.34%` accuracy on 14,709 Complete versus 7,723 non-Complete
rows, rather than the reported `81.64%`. Its Wait miss/false-alarm pair implies
`99.52%`, rather than `98.75%`. This could reflect an undocumented evaluation
subset/aggregation or table errors; author clarification is required before
paper-number reproduction can be an acceptance criterion.

The paper's Table 2 leaves backchannel and wait blank for Smart Turn and leaves
backchannel blank for Paraformer + TEN Turn, demonstrating that unsupported
states need not be forced into fabricated predictions. However, neither the
paper nor repository defines an official label mapping for binary models.

The paper's Table 3 additionally reports overall accuracy and average latency,
but it does not specify the exact latency clock, warm-up policy, hardware timing
scope, percentile distribution, or reference timestamps
([Table 3 and discussion](https://arxiv.org/html/2604.01897v6#S3.SS4)). The
public JSONL has no gold event timestamp from which conversational latency could
be reconstructed. Therefore the published latency values are reference numbers,
not reproducible scorer outputs from this artifact.

There is no upstream implementation of threshold selection, calibration,
`argmax`, continuous-signal mapping, a separate Complete/Incomplete subset, or
decision latency. The dataset has no public calibration/dev split. Optimizing a
threshold or mapping on all 22,432 labels and reporting on those same rows is
test leakage.

## 6. `turn-eval` evaluation contract

To make results deterministic without misrepresenting repository additions as
official protocol, use two distinct suites.

### 6.1 `fastturn-full` (four classes)

- Input universe: all 22,432 examples at the pinned dataset revision.
- Native four-state models: use the adapter's frozen class decision (normally
  `argmax`) with an explicit output-label mapping in the run manifest.
- Primary report: 4x4 confusion matrix; per-class precision, recall, and F1;
  macro-F1; weighted-F1; overall accuracy; official-style one-vs-rest accuracy,
  miss rate, and false alarm rate.
- Report the real-only three-state slice and synthetic wait slice separately.
- Runtime latency is measured locally and reported as initialization time plus
  inference p50/p90/p95. It must not be called the official FastTurn latency.

### 6.2 `fastturn-eot` (repository-defined binary subset)

- Input universe: only the 14,709 `complete` and 3,643 `incomplete` examples.
- This supports EOT-only models without treating backchannel or wait as a
  semantic class they can predict.
- The leaderboard is separate from `fastturn-full`; scores are never merged.
- Report confusion matrix, precision/recall/F1 for both classes, macro-F1,
  balanced accuracy, and overall accuracy.
- Any threshold or mapping learned from labels requires a deterministic,
  group-aware calibration split. Because no original conversation/group ID is
  released, true conversation-disjoint calibration cannot be proven; this is a
  dataset limitation that must remain in the manifest/report.

The production-critical error summaries are:

```text
Incomplete -> Complete error rate
Backchannel -> Complete error rate
Backchannel -> Interruption error rate  # only when an adapter defines an interruption action
```

The first two are direct confusion-matrix projections. The last is a policy
metric, not a native FastTurn class metric; its state-to-action mapping must be
stored and the result marked `derived-interruption`.

## 7. Dataset adapter and artifact requirements

Recommended normalized example:

```text
id              = key
language        = "zh"
user_audio      = wav
agent_audio     = None
transcript_raw  = txt
transcript      = None by default
annotations     = {"turn_state": state, "source": real|synthetic}
duration        = decoded WAV duration
```

The adapter must:

1. Accept either the pinned `.tar.gz` or an extracted root, without permanently
   rewriting/resampling the dataset.
2. Validate the five-field schema, unique IDs/paths, exact class vocabulary,
   `lang == "zh"`, path containment, file existence, mono PCM format, and
   decoded duration.
3. Treat `key` as opaque and reject absolute or parent-traversal `wav` paths.
4. Record native sample rate per row; resample only inside the model adapter and
   include the cost in end-to-end runtime.
5. Preserve row-level predictions with `id`, gold state, predicted state,
   optional four-state probabilities, native sample rate, duration, inference
   time, and source stratum.

Every result manifest should contain at least:

```json
{
  "benchmark": "fastturn-full",
  "dataset_repo": "ASLP-lab/FastTurn-Testset",
  "dataset_revision": "4194458681149e2947cdb062ad61292cf9705a59",
  "archive": "fasttrun-testset.tar.gz",
  "archive_sha256": "888cf6bfcc76df7a034060bbb11a7a1c2053c63901ab5e7f5adca48da14c2795",
  "language": "zh",
  "evaluation_mode": "offline-single-channel-segment",
  "input_mode": "audio-only",
  "mapping": null,
  "calibration_ids_sha256": null,
  "wait_source": "synthetic-deepseek-v3-indextts2"
}
```

## 8. Licensing and unresolved upstream gaps

Licensing is internally inconsistent and should be treated as unresolved:

- The FastTurn dataset card declares `apache-2.0`, but its README contains only
  that YAML tag and the snapshot contains no license text.
- The canonical FastTurn GitHub repository contains no license file.
- The source-corpus repository says SmoothConv and DuplexConv are released
  under [CC BY-NC 4.0](https://github.com/qualialabsAI/SmoothConv-DuplexConv/blob/eb4297fac4f9d6f182740d79bea2df2664b4137e/README.md#license),
  while also stating that FastTurn is derived from SmoothConv.

Do not infer that Apache-2.0 conclusively clears the audio for commercial use or
redistribution. Record the dataset-card declaration and the conflict, avoid
redistributing the archive, and request clarification from ASLP/QualiaLabs
before commercial use.

Other unresolved upstream gaps:

- no annotation guideline or adjudication procedure;
- no source-conversation/group mapping, so leakage-safe regrouping cannot be
  verified;
- no public split or calibration policy;
- no official evaluation code, raw predictions, or metric test vectors;
- no exact inference/preprocessing recipe for the baselines in Tables 2-3;
- no reproducible definition of the published average latency;
- no public mapping protocol for two-state models;
- no direct proof in the artifact of speaker demographics, domain balance, or
  the richer source annotations described by the project.

Until upstream supplies these, `turn-eval` must version its evaluation policy
separately from the dataset revision and describe FastTurn results as
**repository-defined scoring on the official public test artifact**, not as
bit-exact reproduction of an official scorer.
