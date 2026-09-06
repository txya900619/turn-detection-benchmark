# Turn Detection Benchmark

> Continuing this work in a new session? Start with [`HANDOFF.md`](HANDOFF.md) for the current implementation state, canonical artifacts, research conclusions, pinned revisions, and next steps.

Reproducible evaluation of end-of-turn, hold, interruption, backchannel, and
turn-state models for live voice agents.

The implementation contract and milestone acceptance criteria are defined in
[docs/SPEC.md](docs/SPEC.md).

Current milestone progress and validated local baselines are recorded in
[docs/STATUS.md](docs/STATUS.md).

Interpretation of the completed open-weight baselines, including paired Chinese
error slices and deployment caveats, is in
[docs/BASELINE_ANALYSIS.md](docs/BASELINE_ANALYSIS.md).

The canonical cross-benchmark comparison is
[`results/comparison/20260906T073000Z/`](results/comparison/20260906T073000Z/):
`report.md`, `summary.csv`, `summary.json`, `manifest.json`, and figures. That
directory, the two analysis reports under `results/analysis/`, and the runtime
comparison under `results/runtime/` are committed as a provenance record; the
raw run directories they were built from are not. Restore or regenerate those
runs before reproducing numbers, and select rows through the comparison manifest
rather than by scanning run directories for a `SUCCESS` marker.

## Development

Use the shared Hugging Face cache and let `uv` manage the Python environment:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv sync
uv run turn-eval --help
uv run pytest
```

Run the deterministic M0 vertical slice:

```bash
uv run turn-eval run-synthetic --output-dir results
```

## LiveKit eot-bench

The EN/ZH adapter pins both the public dataset revision and the reference scorer
revision. Install the isolated SmartTurn extra only when that adapter is needed:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv sync --extra smart-turn
```

Run a diagnostic smoke test (limited runs are marked leaderboard-ineligible):

```bash
uv run turn-eval run-eot-bench --lang en --model fixed-timeout --limit 2
uv run turn-eval run-eot-bench --lang zh --model smart-turn-v3.2 --limit 2
```

Remove `--limit` for a complete run. Reuse an already populated cache without
network access with `--local-files-only`.

Run DualTurn's native EOT head on the same benchmark in explicit user-only
fallback mode through its pinned isolated environment:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv sync --project environments/dualturn_endpointing --frozen
CUDA_VISIBLE_DEVICES=1 uv run turn-eval run-eot-bench \
  --lang zh \
  --model dualturn \
  --provider cuda \
  --threads 1 \
  --inference-interval-ms 80 \
  --min-silence-span-ms 80 \
  --local-files-only
```

This maps `user=CH0, silence=CH1` and is reported as streaming
`single_channel_fallback`, not native dual-channel execution. The model card
supports English; Chinese is retained as an explicitly unverified zero-shot
row.

Each schema-v2 run writes an immutable manifest, prediction Parquet, frozen policy,
metrics, runtime measurements, `summary.csv`, `pareto.parquet`, and a completion
marker. Primary operating points include group-bootstrap confidence intervals.
Validate or render it without rerunning the model:

```bash
uv run turn-eval validate-run RESULTS_RUN_DIRECTORY
uv run turn-eval report RESULTS_RUN_DIRECTORY
```

Replay completed EN/ZH operating points and analyze HOLD errors without changing
any threshold, action delay, or timeout:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv run --frozen turn-eval analyze-eotbench \
  --candidate-model dualturn-endpointing \
  --run results/en/eotbench/dualturn_single_channel_fallback/RUN_ID \
  --run results/en/eotbench/realtime_vap_single_channel_fallback/RUN_ID \
  --run results/en/eotbench/smart_turn_v3.2_cpu/RUN_ID \
  --output-dir results/analysis/eotbench/ANALYSIS_ID
```

Repeat `--run` for matching Chinese artifacts. The report separates model and
forced-timeout cutoffs, slices HOLD spans by duration, and gives paired
conversation-cluster bootstrap intervals against the named candidate.

## TurnBench

TurnBench scoring is isolated because the pinned official scorer has a different
dependency stack from the core harness:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv sync --project environments/turnbench --frozen
uv run --project environments/turnbench --frozen python environments/turnbench/self_test.py
uv run turn-eval turnbench-validate predictions-dev.json
uv run turn-eval turnbench-score predictions-dev.json
```

Run the exact rule-based RMS energy-VAD floor and score the full dev split:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv run turn-eval run-turnbench-rms-vad --local-files-only
```

The adapter follows the official 20 ms / `0.01` threshold rule exactly. Use
`--limit 1 --no-score` for a partial smoke test; partial runs are never marked
leaderboard-eligible.

The scorer and dev dataset revisions are pinned independently. The dev dataset
is gated; request access to `mundo-ai/turn-benchmark-dev` before running real-data
parity. The exact protocol and license constraints are documented in
[docs/TURNBENCH_PROTOCOL.md](docs/TURNBENCH_PROTOCOL.md).

Run the pinned MaAI multilingual Realtime-VAP backend in its isolated environment:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv sync --project environments/realtime_vap --frozen
CUDA_VISIBLE_DEVICES=1 uv run turn-eval run-turnbench-vap \
  --device cuda:0 \
  --output-dir results/en/turnbench/realtime-vap-tri
```

After inference (or after merging physical Parquet shards), select EOT and
interruption thresholds independently on dev at the official 10% FPR budget:

```bash
uv run turn-eval turnbench-sweep \
  --eot-probabilities RUN_DIR/probs-eot.json \
  --interruption-probabilities RUN_DIR/probs-int.json \
  --output-dir RUN_DIR/scored \
  --fp-budget 0.10
```

This writes the complete sweeps, committed official-format predictions, and
scores. Never use a dev-selected threshold as a test-selected threshold.

The selected tri-lingual checkpoint is CC-BY-NC-SA-4.0 and therefore is a
research reference, not a permissively licensed production candidate.

Run the exact official TurnBench VapGPT OTO baseline in a separate pinned
environment:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv sync --project environments/vap_turnbench --frozen
CUDA_VISIBLE_DEVICES=1 uv run turn-eval run-turnbench-vap-gpt \
  --device cuda \
  --local-files-only \
  --output-dir results/en/turnbench/vap_gpt_oto/RUN_ID
```

This uses native dual-channel audio, 50 Hz output, and the official bounded
25-second windows (20 seconds context plus a 5-second step). It pins the MIT VAP
source and CC-BY-4.0 OTO checkpoint independently, verifies both checkpoint
hashes, and saves resumable per-conversation caches. Optional `--shard-index`
and `--shard-count` arguments allow disjoint Parquet shards to run on separate
GPUs; merge them with `environments/vap_turnbench/merge_shards.py`.

Run the Apache-2.0 DualTurn Endpointing streaming graph on TurnBench:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv sync --project environments/dualturn_endpointing --frozen
CUDA_VISIBLE_DEVICES=1 uv run turn-eval run-turnbench-dualturn \
  --provider cuda \
  --local-files-only \
  --output-dir results/en/turnbench/dualturn_endpointing/RUN_ID
```

The 80 ms ONNX graph keeps its causal KV/LSTM state on GPU through I/O binding.
Native user EOT is evaluated twice with swapped channel orientation so both
TurnBench speakers receive a native dual-channel score. Interruption is a
separate derived result based on the model's 0–240 ms FVAD horizon; it must not
be compared as a native interruption head.

Run the official ESPnet TurnTaking mixed-mono baseline in its isolated
environment:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv sync --project environments/espnet_turntaking --frozen
CUDA_VISIBLE_DEVICES=1 uv run turn-eval run-turnbench-espnet \
  --device cuda \
  --local-files-only \
  --output-dir results/en/turnbench/espnet_turntaking/RUN_ID
```

This pins the CC-BY-4.0 Switchboard checkpoint, ESPnet 202604 source, and the
ESPnet Whisper fork independently. The single-stream Whisper-medium model sees
the causal mono mix `speaker_1 + speaker_2`; native turn-change/interruption
class probabilities are assigned to speakers with the official causal energy
heuristic. The resulting EOT and interruption rows therefore use derived
speaker attribution and are not native dual-channel predictions. The default
TF32 path matches the upstream recipe and is recorded in metadata; use
`--no-tf32` only when slower FP32 reproducibility is required. GPU sharding is
available through `--shard-index`/`--shard-count`, followed by
`environments/espnet_turntaking/merge_shards.py`.

Run the official Mimi Endpointer baseline in its isolated environment:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv sync --project environments/mimi_endpointer --frozen
CUDA_VISIBLE_DEVICES=1 uv run turn-eval run-turnbench-mimi \
  --device cuda \
  --local-files-only \
  --output-dir results/en/turnbench/mimi_endpointer/RUN_ID
```

This pins the CC-BY-4.0 `oto_d1f` TurnBench checkpoint and the matching Kyutai
Mimi tokenizer independently, verifies both hashes, and emits native
dual-channel 25 Hz traces. EOT is derived as `1 - P(active speaker)` and
interruption as `P(active speaker)`. The reference implementation performs
full-sequence inference through a causal Mimi encoder and unidirectional LSTMs;
the execution mode and zero algorithmic lookahead are recorded in metadata.
Optional `--shard-index` and `--shard-count` arguments select disjoint TurnBench
Parquet shards.

Run Kyutai STT-1B Semantic VAD with its official streaming VAD-head path:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv sync --project environments/kyutai_semantic_vad --frozen
CUDA_VISIBLE_DEVICES=1 uv run turn-eval run-turnbench-kyutai \
  --device cuda \
  --local-files-only \
  --output-dir results/en/turnbench/kyutai_semantic_vad/RUN_ID
```

The CC-BY-4.0 EN/FR checkpoint is pinned by revision and four file hashes. Both
speaker channels are processed together through the causal 24 kHz Mimi encoder
and STT model, one 80 ms frame at a time. The native VAD head supplies EOT;
interruption is explicitly derived as `1 - P(turn ending)`. Use
`--shard-index`/`--shard-count` for disjoint Parquet shards and merge them with
`environments/kyutai_semantic_vad/merge_shards.py`. To reproduce the upstream
probability path, 48-to-24 kHz resampling is intentionally executed on the
selected CUDA device rather than on CPU.

Run SmartTurn v3.1 (official reference version) or v3.2 (current candidate) with
the same pinned TurnBench VAD/accumulation/settling policy:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv sync --project environments/smart_turn_turnbench --frozen
CUDA_VISIBLE_DEVICES=1 uv run turn-eval run-turnbench-smart-turn \
  --model-version 3.2 \
  --artifact gpu \
  --provider cuda \
  --workers 8 \
  --local-files-only \
  --output-dir results/en/turnbench/smart_turn_v3_2_gpu/RUN_ID
```

The backend saves a per-conversation resume cache next to the requested output
directory. Its EOT signal is native `P(complete)`; interruption is explicitly
derived as `1 - P(complete)`. The upstream Silero wrapper resets state every five
wall-clock seconds, so exact probability traces can vary with concurrency and
hardware even when all artifacts are pinned.

Run its single-channel fallback on LiveKit eot-bench by placing user audio on
channel 1 and causal silence on channel 2:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
CUDA_VISIBLE_DEVICES=1 uv run turn-eval run-eot-bench \
  --lang zh \
  --model realtime-vap \
  --device cuda:0 \
  --local-files-only
```

This mode is reported separately as `single_channel_fallback`; EOT is the
derived signal `1 - p_now[channel_1]`. It must not be compared as though it had
the agent-context benefit of native dual-channel VAP.

## FastTurn

Download, byte-preservingly extract, and strictly validate the pinned Chinese
FastTurn Test Set:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv run turn-eval prepare-fastturn
```

Run an end-to-end diagnostic vertical slice and then validate/re-render it
without invoking the model again:

```bash
RUN_DIR=$(uv run turn-eval run-fastturn \
  --suite full \
  --model fixed-state \
  --fixed-state complete \
  --limit 8 \
  --quiet \
  --local-files-only)
uv run turn-eval validate-run "$RUN_DIR"
uv run turn-eval report "$RUN_DIR"
```

`fixed-state` is a runner/CI diagnostic, not a research baseline. The available
suites are `eot` (Complete/Incomplete), `real` (three real-audio states), and
`full` (four states including synthetic Wait); their leaderboards must remain
separate.

Run the lightweight open-weight TurnSense INT8 audio baseline on the binary
FastTurn EOT subset:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv sync --project environments/turnsense --frozen
uv run turn-eval run-fastturn \
  --suite eot \
  --model turnsense \
  --threads 4 \
  --local-files-only
```

TurnSense is a 52.2 MB Apache-2.0 ONNX model with native
Complete/Incomplete/Invalid outputs. `invalid` is scored as a closed-set
abstention, not silently converted or dropped. It does not support Backchannel
or Wait, so the CLI rejects `real` and `full` suites.

The full 18,352-clip run produces Complete F1 `85.51%`, Incomplete F1 `59.48%`,
Macro-F1 `72.50%`, and Incomplete→Complete error `23.77%`. CPU inference p50 is
`35.6 ms` at RTF `0.0193` with four threads. The adapter uses the upstream
8-second tail and SoXR-HQ + Kaldi fbank/LFR/CMVN frontend; the feature tensor was
verified element-for-element against source revision
`e1ae083a361182614f15eff5c293dadb07314ec4`.

Run SmartTurn v3.2 on the same binary EOT subset with its predeclared default
threshold; do not tune this threshold on FastTurn test:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv sync --extra smart-turn --frozen
uv run --frozen --extra smart-turn turn-eval run-fastturn \
  --suite eot \
  --model smart-turn-v3.2 \
  --smart-turn-threshold 0.5 \
  --threads 4 \
  --local-files-only
```

The complete run produces Complete F1 `76.48%`, Incomplete F1 `34.14%`,
Macro-F1 `55.31%`, and Incomplete→Complete error `54.74%`. CPU inference p50 is
`164.7 ms` at RTF `0.0694`. This fixed-threshold result is a production-default
reference, not a FastTurn-calibrated operating point.

Run the Apache-2.0 Easy Turn native four-state baseline in its isolated GPU
environment:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv sync --project environments/easy_turn --frozen
EASY_TURN_GPU=2 scripts/run_easy_turn_full.sh
```

Easy Turn pins the 3.45 GB checkpoint, upstream source, and
Qwen2.5-0.5B-Instruct skeleton independently. It uses audio only and preserves
the generated transcript plus native terminal tag. A missing, repeated, or
non-terminal label tag is scored as a closed-set abstention; decoded hard labels
are not presented as calibrated probabilities. The model is evaluated as
offline clip classification, not as a streaming detector.

The complete 22,432-clip run produces Macro-F1 `79.41%` and per-class F1 of
`84.28%` Complete, `55.60%` Incomplete, `81.65%` Backchannel, and `96.10%`
Wait. Incomplete→Complete error is `25.75%`. GPU inference latency is p50
`416.5 ms`, p90 `818.1 ms`, and p95 `1027.5 ms`, with RTF `0.244`. Six clips
abstain and remain incorrect in the denominator: four exceed the upstream
30-second encoder limit and two generations contain no valid terminal tag.

The public artifact contains 22,432 isolated mono clips, not the original
dual-channel conversation timelines. Audio-only examples expose no transcript by
default because the released real-data transcripts contain terminal gold-label
tags. The dataset's Hugging Face Apache-2.0 declaration conflicts with the
CC-BY-NC-4.0 provenance stated by the source-corpus project; do not redistribute
or assume commercial clearance without upstream clarification. See
[docs/FASTTURN_PROTOCOL.md](docs/FASTTURN_PROTOCOL.md).

### X2-Turn — deferred

X2-Turn runs in a dedicated `uv` environment pinned to the official source and
checkpoint revisions. The released local Transformers path is an offline
two-pass implementation, so these results are labeled `causal_offline` and must
not be used as native-streaming latency claims. It is retained for future
high-capacity reference work but excluded from the first production ranking and
no complete baseline is currently scheduled.

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv sync --project environments/x2_turn --frozen
CUDA_VISIBLE_DEVICES=1 uv run turn-eval run-fastturn \
  --suite eot \
  --model x2-turn \
  --device cuda:0 \
  --local-files-only
```

Use `--suite real` for Complete/Incomplete/Backchannel. `--suite full` is
rejected because X2-Turn has no native Wait class. The adapter follows the
paper's last-non-idle reduction, preserves every raw six-way frame probability
and generated transcript, and records `idle`, `noidle`, or `uncertain` as an
abstention rather than inventing a class mapping. See
[docs/X2_TURN_PROTOCOL.md](docs/X2_TURN_PROTOCOL.md).

## SID-Bench

Download and validate the pinned EN, ZH, or shared noise/silence split:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv run turn-eval prepare-sidbench --split en
uv run turn-eval prepare-sidbench --split zh
uv run turn-eval prepare-sidbench --split noise
```

Metadata-only preparation is available with `--metadata-only`. Score an
official-format prediction file without loading audio:

```bash
uv run turn-eval score-sidbench predictions.jsonl --split zh --local-files-only
```

Use `--allow-partial` only for diagnostic smoke artifacts; primary scores require
exact split coverage.

The scorer reproduces the author's released notebook, including its 50 ms
boundary behavior, while also reporting strict-paper FIR and non-break FPR as
diagnostics. SID-Bench has no calibration split and releases user-only mono
clips, so policies must be frozen externally and dual-channel models must be
reported as single-channel fallback. The dataset has no declared license in its
Hub metadata or official repository. See
[docs/SIDBENCH_PROTOCOL.md](docs/SIDBENCH_PROTOCOL.md).

Run the pinned MaAI multilingual Realtime-VAP baseline with the interruption
threshold frozen at `0.82` from TurnBench EN dev:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
CUDA_VISIBLE_DEVICES=1 uv run turn-eval run-sidbench-vap \
  --split zh \
  --device cuda:0 \
  --local-files-only \
  --output-dir results/zh/sidbench/realtime_vap_single_channel_fallback/RUN_ID
```

The backend uses `user=channel_1, silence=channel_2`, commits at the first
100 ms frame whose derived `p_now[channel_1] >= 0.82`, and saves resumable
per-clip traces. This is a non-commercial CC-BY-NC-SA-4.0 research reference,
not a native dual-channel or permissively licensed production result.

Run the permissively licensed DualTurn Endpointing fallback with its derived
FVAD threshold frozen on TurnBench EN dev:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
CUDA_VISIBLE_DEVICES=2 uv run turn-eval run-sidbench-dualturn \
  --split zh \
  --provider cuda \
  --threads 1 \
  --local-files-only \
  --output-dir results/zh/sidbench/dualturn_single_channel_fallback/RUN_ID
```

This policy maps `user=CH0, silence=CH1` and commits at the first complete
80 ms frame whose `fvad[0,0]` (the user 0-240 ms future-activity horizon) is at
least `0.9896328579889585`. It is an Apache-2.0 open-weight baseline, but the
SID-Bench result remains derived single-channel fallback—not DualTurn's native
dual-channel operating mode and not a native semantic-interruption head.

Compare two sets of already completed runs without recalibrating either policy:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv run --frozen turn-eval analyze-sidbench \
  --candidate-run results/en/sidbench/dualturn_single_channel_fallback/RUN_ID \
  --reference-run results/en/sidbench/realtime_vap_single_channel_fallback/RUN_ID \
  --output-dir results/analysis/sidbench/ANALYSIS_ID
```

Repeat both run arguments for matching ZH and noise pairs. The analyzer requires
identical clip coverage and dataset revisions, preserves each run's frozen
threshold, reports category-level paired differences, and writes 10,000-resample
95% paired bootstrap intervals plus hashes of all source artifacts.

## Isolated runtime profiles

Profile a model in a fresh process with explicit warmup, raw per-tick latency,
process-scoped CPU/RAM sampling, and PID-scoped NVIDIA VRAM sampling:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
CUDA_VISIBLE_DEVICES=2 uv run turn-eval profile-runtime \
  --model dualturn \
  --backend cuda \
  --threads 1 \
  --warmup-frames 64 \
  --measurement-frames 512 \
  --local-files-only \
  --output results/runtime/dualturn/PROFILE.json
```

Supported models are `dualturn` and `realtime-vap`; supported backends are
`cpu` and `cuda`. Render and validate multiple raw profiles with:

```bash
uv run turn-eval report-runtime PROFILE_1.json PROFILE_2.json \
  --output-dir results/runtime/REPORT_ID
```

Cold first-tick latency currently has one sample per fresh process and assumes
warm OS/Hugging Face caches. It must not be presented as a cold-start
distribution without running multiple independent profiles. See
[docs/RUNTIME_PROTOCOL.md](docs/RUNTIME_PROTOCOL.md) for the comparison and
resource-scope contract.

## Unified baseline comparison

Discover only complete eligible runs and render the English/Chinese benchmark
matrix with explicit channel mode, execution mode, signal provenance, license,
and runtime columns:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv run turn-eval report-suite \
  --lang all \
  --results-root results \
  --runtime-summary results/runtime/20260905T103200Z/summary.json \
  --output-dir results/comparison/REPORT_ID
```

The command emits `report.md`, normalized `summary.csv`, structured
`summary.json`, and a `manifest.json` containing the selection policy, rejected
runs, and SHA256 of every selected input. It excludes partial runs, incomplete
coverage, missing official scores, and diagnostic FastTurn models. Duplicate
model/mode identities resolve to the latest completed run, with a canonical
`final` directory winning timestamp ties. No overall score is generated.

Render the canonical EOT Pareto curves and the SID-Bench frozen-policy
interruption comparison from that summary:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv run turn-eval plot-suite \
  --comparison-summary results/comparison/REPORT_ID/summary.json \
  --output-dir results/figures/FIGURE_ID
```

The plot artifact includes the PNG figures, exact `plot_data.json`, input
SHA256 hashes, and a short protocol note. SID-Bench is shown as a frozen-policy
trade-off scatter, not a Pareto curve: the command never sweeps a threshold on
the SID test set.
