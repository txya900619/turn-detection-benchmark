# Turn Detection Benchmark Specification

Status: Draft v0.1  
Scope: M0 and the first end-to-end EOT evaluation slice

## 1. Purpose

This repository provides reproducible inference, evaluation, runtime measurement,
and reporting for turn-detection models used in live voice agents. It answers two
questions without collapsing them into one score:

1. How well does a model decide that a user has completed or is holding a turn?
2. How well does a model distinguish a genuine interruption from a backchannel or
   other non-interruptive speech?

The repository does not train or fine-tune models. It may fit evaluation policies
on an explicitly designated calibration split, but model weights must remain
unchanged.

## 2. Normative language

The terms MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY describe requirements of the
benchmark harness. A run that violates a MUST or MUST NOT requirement is invalid
and must not appear on a primary leaderboard.

## 3. Evaluation protocols

A benchmark adapter MUST declare exactly one protocol. Protocols are different
interfaces because their prediction units and latency semantics are different.

### 3.1 Timeline protocol

The timeline protocol evaluates continuous conversations. Inputs contain a shared
conversation clock, one required user/speaker channel, an optional second channel,
positive event anchors, and scored negative intervals.

Examples: TurnBench EOT and interruption.

The model emits time-indexed signals. A frozen policy converts those signals into
committed events. Scoring uses the event commit time, not an earlier boundary that
the model estimated retrospectively.

### 3.2 Decision-point protocol

The decision-point protocol evaluates causal decisions at predefined points in a
turn. A decision point identifies a timestamp and the input prefix available at
that timestamp. Earlier silence points may be `hold`; the final silence is `eot`.

Example: LiveKit eot-bench.

The benchmark adapter MAY expose the complete stored waveform to the harness, but
the model adapter MUST receive only the causal prefix constructed by the harness.
Dataset labels and post-decision audio MUST NOT be passed to the model adapter.

### 3.3 Clip-classification protocol

The clip-classification protocol evaluates a bounded audio or text example with a
single semantic label, such as `complete`, `incomplete`, `backchannel`, or `wait`.

Examples: FastTurn Test Set and Easy Turn Testset.

Clip classification does not by itself establish conversational decision latency.
Unless a dataset supplies a continuous source clock and a gold decision time, the
report MUST label timing as inference latency rather than algorithmic or pipeline
decision latency.

### 3.4 Single-channel interruption timeline

SID-Bench provides bounded user-audio clips with a gold semantic break time but
no simultaneous agent channel. The model consumes causal prefixes and commits at
most one interruption event. Results MUST disclose the missing agent context;
dual-channel models run with a silent second channel are single-channel fallback,
not native dual-channel evaluations.

## 4. Time and causality

All dataset time is measured in seconds from the start of the example's shared
audio clock.

Every prediction records:

- `available_at_s`: audio-clock time when the prediction can be acted upon.
- `observed_until_s`: latest source timestamp consumed to make the prediction.
- `emitted_at_ns`: optional monotonic wall-clock timestamp for runtime tracing.

The following invariants are mandatory:

1. `available_at_s >= observed_until_s`.
2. No audio, transcript token, agent audio, VAD result, or message newer than
   `observed_until_s` may affect the prediction.
3. Algorithmic lookahead is charged by advancing `observed_until_s` and therefore
   `available_at_s`; it is never kept only as model metadata.
4. A policy event is scored at its commit time. A model that recognizes an event at
   0.7 s using input through 1.0 s commits at or after 1.0 s.
5. Offline execution is primary-leaderboard eligible only when it is a verified
   causal replay of the streaming computation. Otherwise it is marked diagnostic.

The harness, not a dataset or model adapter, owns prefix construction and the
shared evaluation clock. This concentrates leakage prevention at one seam.

## 5. Core domain types

The public domain model contains three example types rather than one mostly-empty
universal record:

- `TimelineExample`
- `DecisionPointExample`
- `ClipClassificationExample`

Shared fields are limited to stable identity and media metadata:

- example and conversation identifiers
- language and optional locale
- audio references with sample rate, channels, and duration
- dataset metadata required for grouping

Protocol-specific annotations remain on the corresponding example type. Model
adapters never receive annotation objects.

A raw model prediction contains time, signal probabilities, optional state
probabilities, and provenance. A committed event is a separate policy output. This
keeps model inference distinct from model-to-action mapping.

## 6. Model seams

Execution shape is the only public model seam in M0.

### 6.1 Streaming model interface

A streaming model creates an isolated session, accepts chronological user and
optional agent audio chunks, returns zero or more predictions, and resets all
state when the session closes. It MUST reject non-monotonic chunks and unsupported
channel layouts.

### 6.2 Pointwise model interface

A pointwise model accepts one causal decision request and returns one prediction.
The request contains a causal media prefix, permitted transcript or message
context, the decision timestamp, and no gold annotations.

In-process, subprocess, container, and remote transports are implementations
behind these interfaces, not additional public model interfaces.

The harness chooses an eligible execution path from the benchmark protocol and
model descriptor. An unsupported combination fails before dataset inference.

## 7. Model descriptor and support status

A model descriptor MUST declare:

- stable model name, revision, and artifact checksum when available
- native sample rate, frame interval, context window, and declared lookahead
- accepted inputs: user audio, agent audio, transcript, or message history
- supported execution shapes
- native output signals and turn-state labels
- supported languages/locales and the evidence for that support
- model and code licenses

Support is resolved per model, benchmark, task, language, and evaluation mode:

- `supported`: native documented capability
- `derived`: produced by a declared frozen mapping or policy
- `unverified_zero_shot`: executable but not documented as supported
- `unsupported`: not evaluated

Derived signals MUST identify their mapping, parameters, calibration source, and
mapping hash. Native and derived results MUST appear in separate leaderboard
groups.

## 8. Evaluation mode dimensions

Evaluation modes are orthogonal fields, not one mixed enum:

- `channel_mode`: `native_single`, `native_dual`, `single_channel_fallback`
- `execution_mode`: `pointwise`, `streaming`, `causal_offline`, `offline`
- `signal_provenance`: `native`, `derived`
- `transcript_mode`: `none`, `oracle_causal`, `streaming_asr`
- `service_mode`: `local`, `cloud`
- `language_support`: `supported`, `unverified_zero_shot`

Reports MUST group or visibly label results that differ on any material dimension.
In particular, native dual-channel and single-channel fallback scores are not
interchangeable.

## 9. Audio and incremental text

Datasets are immutable inputs and MUST NOT be permanently resampled or otherwise
rewritten by a run. The harness converts audio to a model's native sample rate and
records the resampler, version, input/output rates, and resampling wall time.

Streaming resampling MUST preserve state across chunks. Independently resampling
each chunk is invalid because it can create boundary artifacts.

Transcript-conditioned runs MUST declare one of:

- `oracle_causal`: gold words are released only after their annotated end time,
  plus any declared transcript lag.
- `streaming_asr`: actual partial ASR hypotheses and revisions are replayed.

Passing a complete reference transcript to a causal model is invalid.

## 10. Policies and calibration

Models produce signals; policies produce committed actions or events. Policies are
versioned artifacts and may include thresholds, sustain windows, action delays,
timeouts, smoothing, or a derived-signal mapping.

Policy fitting MUST use only an explicitly named calibration or development split.
The resulting policy is frozen and hashed before test inference or scoring. Test
labels MUST NOT influence thresholds, mappings, operating points, or retry choices.

Benchmark-native procedures remain authoritative:

- TurnBench selects an operating point on development data and scores committed
  events with the pinned official scorer.
- eot-bench preserves the official threshold, action-delay, and timeout sweep.
- FastTurn native state models use argmax; learned mappings require calibration.
- SID-Bench preserves official mean FIR, IRL, and APT definitions. Additional
  percentiles are diagnostic.

SID-Bench's public README, paper, and notebook disagree at edge cases. The pinned
notebook is authoritative for official numeric parity: FIR uses all evaluated
examples as its denominator, IRL averages late delays beyond the 50 ms tolerance,
and a non-break trigger within 50 ms of clip end is exempted by the notebook's
branch order. Reports MUST also show strict-paper FIR and the exemption count.

If a public dataset has no valid calibration split, a tuned test result is invalid.
The report must use an externally frozen policy or label the run diagnostic.

## 11. Metrics

### 11.1 EOT/HOLD

Primary eot-bench summaries are:

- false-cutoff rate at 300 ms mean latency
- false-cutoff rate at 600 ms mean latency
- mean latency at 5% false-cutoff rate
- mean latency at 10% false-cutoff rate

The full Pareto frontier is retained. AUC and average precision are diagnostic.

### 11.2 Continuous EOT and interruption

TurnBench reports recall, false-positive rate, and latency p10/p50/p90 separately
for EOT and interruption, using the pinned official scorer and its event windows.

### 11.3 Semantic interruption

SID-Bench reports interruption recall, FIR, official mean IRL, IRL p50 and p90/p95,
and official APT. The number of positive events contributing to latency percentiles
MUST be shown. APT is a benchmark-specific summary, not a repository-wide score.

### 11.4 Turn-state classification

Turn-state reports include per-class precision, recall, F1, support, macro-F1,
weighted-F1, and a confusion matrix. FastTurn produces separate leaderboards for:

- real three-state data: complete, incomplete, backchannel
- full four-state data including synthetic wait
- two-state EOT subset: complete, incomplete

Incomplete-to-complete and backchannel-to-complete error rates are required.
Backchannel-to-interruption error is reported only when a policy defines an
interruption action.

### 11.5 Uncertainty

Primary metrics SHOULD include 95% confidence intervals. Resampling groups MUST
preserve the highest available dependence unit, normally conversation or speaker
pair rather than individual frames or decision points. Reports show sample, event,
and group counts alongside confidence intervals.

No repository-wide overall score is defined.

## 12. Latency and runtime

Three latency families are reported separately:

- `algorithmic_latency_ms`: committed audio-clock time minus gold event time
- `inference_latency_ms`: wall time spent computing one model invocation/tick
- `pipeline_latency_ms`: actionable wall-clock delay including upstream waits,
  model compute, policy sustain/action delay, and scheduling when measured

Runtime reports include initialization time, warm and cold inference distributions,
RAM peak, GPU VRAM peak when applicable, model artifact size, thread count, batch
size, device, and hardware/software identity.

The standard real-time factor is:

`rtf = processing_wall_seconds / audio_seconds` (lower is better)

The inverse MAY also be reported as:

`realtime_multiplier = audio_seconds / processing_wall_seconds` (higher is better)

CPU utilization is meaningful only with the process scope, sampling method,
logical-core count, thread settings, and concurrency recorded.

## 13. Run artifacts

Each completed run is immutable and stored under a unique run identifier. A run
contains:

- `manifest.json`
- `predictions.parquet`
- `policy.json`
- `metrics.json`
- `runtime.json`
- a completion marker written only after validation succeeds

The manifest MUST contain at least:

- artifact schema version and run identifier
- model name, revision, artifact checksum, and known training datasets
- dataset repository, configuration, split, revision, checksums, and licenses
- benchmark and official scorer revision
- task, language, locale, protocol, and evaluation mode dimensions
- calibration source and policy hash
- git commit and dirty-worktree state
- Python and relevant runtime/library versions
- hardware, device, thread count, seed, and timestamps
- possible benchmark overlap and whether the model team authored the benchmark

Predictions are the source artifact; metrics must be reproducible from predictions,
policy, dataset annotations, and the pinned scorer. Reporting MUST NOT mutate run
artifacts.

Partial or skipped examples make a run ineligible for a primary leaderboard unless
the official benchmark explicitly permits them. By default, inference aborts on an
example error. A diagnostic `skip_errors` mode records every omission and reason.

## 14. Dataset storage

Datasets and model downloads are not committed to this repository. Development and
benchmark commands use the shared Hugging Face cache:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv sync
```

Download metadata records the repository revision, checksum where available, and
separate code/data licenses. Gated or non-redistributable datasets are represented
by download instructions and adapters only.

## 15. M0 acceptance criteria

M0 is complete when:

1. `uv run turn-eval --help` succeeds.
2. Domain types validate protocol-specific examples and reject invalid timestamps.
3. A synthetic decision-point dataset and fixed-timeout baseline run end to end.
4. The run writes and validates all required artifacts.
5. The report regenerates metrics from stored predictions rather than rerunning the
   model.
6. Tests prove that future audio and complete transcripts cannot reach a model
   adapter.
7. Tests charge declared lookahead to prediction availability.
8. Tests cover stateful streaming resampling and timestamp continuity.
9. Tests reject test-set policy fitting and unsupported capability combinations.
10. CI uses only tiny synthetic fixtures and a deterministic fake model.

## 16. First vertical slice

Implementation proceeds through one complete path rather than scaffolding every
future adapter:

1. project packaging and CLI using `uv`
2. core protocol types and validation
3. synthetic decision-point benchmark adapter
4. fixed-timeout/VAD-style baseline
5. causal runner and EOT policy sweep
6. immutable artifact writer and validator
7. Markdown/CSV/Pareto reporting
8. LiveKit eot-bench adapter
9. SmartTurn v3.2 adapter

Only after this path passes causality and artifact tests are timeline and
clip-classification protocols implemented.

## 17. Explicit non-goals for M0

M0 does not include:

- model training or fine-tuning
- whole-agent ASR-to-TTS evaluation
- a single overall turn score
- all planned model or benchmark adapters
- distributed benchmark orchestration
- permanent dataset preprocessing
- claiming that Chinese coverage is equivalent to a continuous Chinese TurnBench
