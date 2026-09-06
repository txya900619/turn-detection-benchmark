# Implementation Status

Updated: 2026-09-06

## Complete

- M0 executable specification and causal decision-point vertical slice
- immutable run artifacts and offline metric recomputation
- pinned LiveKit eot-bench EN/ZH dataset adapter
- official 100 ms decision grid and 200–5000 ms HOLD filtering rules
- official threshold/action-delay/timeout policy grid
- fine-grained oracle-VAD timeout baseline
- SmartTurn v3.2 int8 CPU adapter with upstream Whisper preprocessing
- diagnostic AUC and average precision
- fixed-policy cluster-bootstrap 95% confidence intervals
- progress/ETA callbacks for long pointwise runs
- schema-v2 `summary.csv` and `pareto.parquet` artifacts
- local EN/ZH full-dataset runs
- pinned TurnBench protocol contract and isolated official scorer environment
- strict TurnBench submission validation and causal committed-event export
- lazy TurnBench dual-channel Parquet ingestion using full 48 kHz channels
- official-format probability traces, frame-count checks, threshold candidates,
  and rising-edge/refractory commit policy
- authenticated TurnBench dev gold-count parity and real-data oracle scoring
- official SmartTurn non-oracle prediction-artifact parity through the pinned scorer
- exact TurnBench RMS VAD adapter and full 38-conversation local inference,
  with predictions identical to the official committed dev artifact
- isolated SmartTurn TurnBench v3.1/v3.2 GPU backend with pinned Torch,
  Torchaudio, Transformers, ONNX Runtime, model weights, and Silero artifact
- full 38-conversation SmartTurn v3.1 and v3.2 inference with independently
  selected official EOT/INT operating points and resumable conversation caches
- exact official VapGPT/OTO isolated backend with pinned VAP source, Torch stack,
  checkpoint/CPC hashes, native dual-channel bounded-context inference, and
  resumable multi-GPU Parquet sharding
- full 38-conversation VapGPT inference with 5,261,292 task probabilities
  matching the quantized upstream artifacts within `5.01e-5` and exact official
  operating points/scores
- pinned DualTurn Endpointing 80 ms ONNX streaming backend with causal
  resampling, native two-orientation EOT, explicitly derived FVAD interruption,
  GPU-resident state, resume caches, and Parquet sharding
- one-conversation DualTurn CUDA smoke with bit-identical ordinary-vs-I/O-binding
  predictions, followed by a full 38-conversation three-shard run with exact
  coverage, finite probability, and independent scorer validation
- isolated ESPnet TurnTaking mixed-mono backend with pinned Whisper-medium
  checkpoint, ESPnet/Whisper source revisions, official 25 Hz causal windowing,
  resumable caches, strict probability artifacts, and multi-GPU shard merging
- full 38-conversation ESPnet TurnTaking inference, independent scorer
  validation, and fixed-official-threshold parity within the English harness
  tolerance for recall, FPR, and latency
- isolated Mimi Endpointer backend with pinned endpoint/Mimi weights, exact
  two-stream causal LSTM architecture, 24 kHz preprocessing, resumable caches,
  strict probability artifacts, and optional Parquet sharding
- full 38-conversation Mimi Endpointer inference, finite/coverage validation,
  independent scorer validation, and fixed-official-threshold parity within the
  English harness tolerance
- isolated Kyutai STT-1B Semantic VAD backend with pinned checkpoint hashes,
  native two-speaker streaming execution, resume caches, Parquet sharding, and
  strict shard-identity merging
- full 38-conversation Kyutai Semantic VAD inference and independent scorer
  validation; EOT and derived interruption both pass the fixed-threshold
  English harness tolerance after matching upstream CUDA resampling
- isolated MaAI multilingual Realtime-VAP backend with pinned code, checkpoint,
  CPC feature extractor, causal inference, and official probability artifacts
- full 38-conversation TurnBench dev execution of MaAI trilingual Realtime-VAP,
  independently selected EOT/INT operating points, and standalone scorer parity
- bounded-vectorized causal linear resampling with exact 48-to-16 kHz regression coverage
- pinned FastTurn protocol audit, safe archive preparation, strict 22,432-row
  mono clip loader, native-rate validation, and transcript label-leak prevention
- versioned four-state classification metrics including official-style one-vs-rest
  accuracy/miss/false-alarm rates and repository P/R/F1/confusion summaries
- gold-blind clip-classification model seam, resampling/runtime runner, immutable
  state artifacts, offline metric recomputation, and Markdown/CSV reporting
- FastTurn `eot`, real three-state, and full four-state CLI suites with a
  fixed-state diagnostic adapter for end-to-end CI and smoke validation
- pinned Easy Turn checkpoint, source, and Qwen skeleton; isolated `uv` GPU
  environment; exact four-state terminal-tag parser; and validated real-audio
  inference across Complete, Incomplete, Backchannel, and Wait
- pinned X2-Turn source/checkpoint protocol audit and isolated `uv` environment
- X2-Turn offline two-pass FastTurn adapter with the paper's last-non-idle
  reduction, exact three-state mapping, explicit abstentions, and raw frame/ASR
  preservation
- verified official English smoke fixture (expected transcript and 53 frames)
  and one-example FastTurn EOT artifact through the full subprocess/report path
- Realtime-VAP stateful decision-point adapter with strict per-example reset,
  causal 100 ms frame alignment, silent-channel fallback, and derived EOT signal
- verified one-example Chinese eot-bench execution through the isolated VAP
  environment and immutable artifact/report path
- pinned SID-Bench EN/ZH/noise dataset adapter with annotation checksums,
  strict schema/count validation, and explicit user-only channel semantics
- SID-Bench FIR/IRL/APT scorer reproducing the official notebook's released
  Chinese reference values, with strict-paper diagnostics for scorer conflicts
- resumable SID-Bench Realtime-VAP backend with a TurnBench-dev-frozen policy,
  causal 100 ms commits, full probability traces, and explicit
  single-channel-fallback/derived-signal labeling
- full Realtime-VAP SID-Bench EN, ZH, and Noise/Silence runs with exact artifact
  coverage and independent JSONL rescoring
- resumable DualTurn SID-Bench backend with causal 80 ms frame-end commits,
  TurnBench-dev-frozen FVAD policy, and explicit single-channel-fallback and
  derived-signal labeling
- full DualTurn SID-Bench EN, ZH, and Noise/Silence runs with exact artifact
  coverage, independent rescoring, and an Apache-2.0 open-weight checkpoint
- shared isolated runtime profiler with fresh-process construction/first-output
  clocks, raw cold/warm tick latency, process-scoped CPU/RAM, PID-scoped VRAM,
  model artifact size, and hardware/software identity
- validated CPU/CUDA runtime profiles for DualTurn and Realtime-VAP plus a
  raw-trace-recomputing Markdown/CSV/JSON comparison report
- unified EN/ZH baseline report discovery with complete-run eligibility,
  duplicate canonicalization, explicit mode/signal/license columns, input
  hashes, exclusion provenance, and no overall score
- full DualTurn LiveKit eot-bench EN/ZH runs in causal 80 ms streaming,
  single-channel-fallback mode, with validated native-EOT artifacts
- paired 18,352-clip FastTurn error analysis for TurnSense versus SmartTurn,
  including duration slices, bootstrap uncertainty, and abstention fallback risk
- reproducible SID-Bench fixed-policy paired analyzer with per-category error
  decomposition, 10,000-resample paired intervals, input hashes, and no test tuning
- eot-bench FC@300 error decomposition by HOLD duration and model-versus-timeout
  cause, showing that all current candidates remain unsafe on pauses over one second
- reproducible frozen-policy eot-bench HOLD analyzer with trace-count parity,
  duration slices, paired conversation-cluster bootstrap, and hashed inputs

Pinned external revisions:

- `livekit/eot-bench-data`: `ca9d98a9686b920a2d8c9eb984224ba9be74e4dd`
- `livekit/eot-bench` scorer reference: `6594d8b3b8af385b15f116dde310ce45af92d646`
- `pipecat-ai/smart-turn-v3`: `f766f81d3cfdf7737ac64aad813d91bbfd56bf93`
- `SesameAILabs/turnbench` scorer: `e02d547803fd0e45221afcaeb1cd24b61ae7568e`
- `mundo-ai/turn-benchmark-dev`: `8fa18a24be51528a45397b35cbcaecd84202062b`
- `ErikEkstedt/VoiceActivityProjection`: `f39a78b23a6dccdbedd106e00b48c410b8739f5d`
- `viks66/VAP_checkpoints`: `b9aa0ba1718221153e04ef343c7f3b8cf84bfbc3`
- `anyreach-ai/dualturn-endpointing`: `c3860ed71210fe0144af35af340fd7a4dec3d2d3`
- `espnet/Turn_taking_prediction_SWBD`: `766378e6119f254270f0dd4a4be13480ec6b78fc`
- `espnet/espnet`: `750e3749fc37a09187fe0fc6fb278ccb007181e8`
- `espnet/whisper`: `8a65df1760894b91857c9904410cca1123fa1bb0`
- `viks66/mimi-endpointer`: `f7459f95f43766a52c612ee1bc72b2444e86fc6b`
- `kyutai/moshiko-pytorch-bf16`: `2bfc9ae6e89079a5cc7ed2a68436010d91a3d289`
- `kyutai/stt-1b-en_fr-candle`: `095e38f6242006a93c2541149b181988397f5c7c`
- `ASLP-lab/FastTurn`: `0c1c03dc03c9cc0bf7a743a066fc99107c8a5c59`
- `ASLP-lab/FastTurn-Testset`: `4194458681149e2947cdb062ad61292cf9705a59`
- `ASLP-lab/Easy-Turn`: `7d1f8b06776b575ac22e2fd4aa7913da2f4b8b98`
- `ASLP-lab/Easy-Turn` source: `fe175eaa8442c9a0834acf29f8640ee74c2aceca`
- `Qwen/Qwen2.5-0.5B-Instruct`: `7ae557604adf67be50417f59c2c2f167def9a775`
- `X-Square-Robot/X2-Turn` source: `3609a0fd11b4a1dc2b2629e95538bbe0860da96d`
- `x-square-robot/X2-Turn-4B-0812`: `d188a8b86058ece5e293124a278a1a6256af81fa`
- `kxxia/SID-bench`: `6eb13b573ad588646ce0de513d4255e55caf858b`
- `xkx-hub/SID-bench` scorer: `0a9e5001e1fe534b3eec4f76c4455ae4bff9ccf1`

The local scorer was checked against all 5,454 combinations in the pinned
reference policy sweep. False-cutoff values matched exactly and latency differed
only by floating-point noise below `2e-15` seconds.

The complete SID-Bench EN, ZH, and Noise/Silence audio splits are cached and
validated at the pinned revision. The new scorer reproduces the author's bundled
1,600-row Chinese prediction example: FIR `0.1375`, mean IRL
`0.33837436932391524` seconds, and APT `0.6561522135416666` seconds. The stricter
paper interpretation gives FIR `0.145`; the difference is 12 non-break triggers
inside the notebook's clip-end tolerance exemption. See
`docs/SIDBENCH_PROTOCOL.md` before comparing SID-Bench rows.

The full oracle-VAD baseline reproduces the official English comparison values:

| Metric | Value |
|---|---:|
| False cutoff at 300 ms | 55.6% |
| False cutoff at 600 ms | 21.7% |
| Latency at 5% false cutoff | 1600 ms |
| Latency at 10% false cutoff | 1000 ms |

## Current local baselines

| Language | Model | FC @300 ms | FC @600 ms | Latency @5% | Latency @10% |
|---|---|---:|---:|---:|---:|
| EN | oracle-VAD timeout | 55.6% | 21.7% | 1600 ms | 1000 ms |
| ZH | oracle-VAD timeout | 51.2% | 16.3% | 1100 ms | 800 ms |
| EN | SmartTurn v3.2 int8 CPU | 37.6% | 14.8% | 1044 ms | 755 ms |
| ZH | SmartTurn v3.2 int8 CPU | 41.8% | 12.8% | 928 ms | 667 ms |
| EN | MaAI trilingual VAP, single-channel fallback | 33.3% | 13.6% | 1016 ms | 702 ms |
| ZH | MaAI trilingual VAP, single-channel fallback | 31.2% | 9.7% | 879 ms | 597 ms |
| EN | DualTurn, single-channel fallback | 22.1% | 9.6% | 991 ms | 566 ms |
| ZH | DualTurn, single-channel fallback | 23.4% | 10.2% | 910 ms | 612 ms |

The SmartTurn values use `smart-turn-v3.2-cpu.onnx`. LiveKit's published
SmartTurn row uses the distinct unquantized `smart-turn-v3.2-gpu.onnx` artifact,
so the two rows must not be treated as an exact model-reproduction comparison.
The current SmartTurn runtime measurements were collected while EN and ZH jobs ran
concurrently and are diagnostic, not runtime-leaderboard results.

The Realtime-VAP ZH row is a complete 344-turn run with 9,441 causal decision
points. It places user audio on channel 1 and silence on channel 2, and derives
EOT as `1 - p_now[channel_1]`; therefore it is a single-channel fallback result,
not a native dual-channel score. The run used 4,029.17 seconds of audio at RTF
`0.223522` (4.47× realtime) and is stored under
`results/zh/eotbench/realtime_vap_single_channel_fallback/20260904T171258618568Z-3112c7e7/`.
The checkpoint license is CC-BY-NC-SA-4.0, so this is a research baseline rather
than a permissively licensed production candidate.

The matching EN fallback run covers all 400 turns and 11,191 decision points.
It processed 5,040.90 seconds of audio at RTF `0.223765` and is stored under
`results/en/eotbench/realtime_vap_single_channel_fallback/20260904T173254860352Z-365fc311/`.

Realtime-VAP was also evaluated on all three SID-Bench test splits with its
interruption threshold fixed at `0.82`, selected previously on TurnBench EN dev.
SID-Bench labels were not used for threshold selection. Since SID-Bench provides
only user audio, this places user audio on channel 1 and silence on channel 2;
the signal is derived from `p_now[channel_1]` and belongs to a separate
single-channel-fallback leaderboard.

| Split | Recall | Official FIR | Strict FIR | Mean IRL | APT | Non-break FPR |
|---|---:|---:|---:|---:|---:|---:|
| EN | 44.00% | 67.25% | 67.31% | 124 ms | 2.521 s | 92.4% |
| ZH | 47.18% | 64.50% | 64.63% | 116 ms | 2.416 s | 90.6% |
| Noise/Silence | n/a | 91.60% | 91.60% | n/a | 12.303 s | 91.6% |

The model processed 5,991.68 s EN, 6,161.45 s ZH, and 6,487.96 s noise audio
at model-compute RTF `0.2249`, `0.2252`, and `0.2306`, respectively. These
results show that a multilingual future-activity model that is competitive on
TurnBench is not automatically a semantic interruption detector: the frozen
fallback policy fires on more than 90% of SID non-break and noise examples.
Artifacts are stored under:

- `results/en/sidbench/realtime_vap_single_channel_fallback/20260905T090900Z/`
- `results/zh/sidbench/realtime_vap_single_channel_fallback/20260905T090900Z/`
- `results/shared/sidbench/realtime_vap_single_channel_fallback/20260905T090900Z/`

DualTurn Endpointing was evaluated on the same three SID-Bench splits using
`user=CH0, silence=CH1`. The interruption signal is the derived user
`fvad[0,0]` 0-240 ms horizon, not a native semantic-interruption output. Its
threshold `0.9896328579889585` was frozen previously on TurnBench EN dev at the
FPR <= 0.10 operating point; SID-Bench labels were not used for policy tuning.
The checkpoint and code are Apache-2.0, while the results remain in the
single-channel-fallback/derived-interruption leaderboard group.

| Split | Recall | Official FIR | Strict FIR | Mean IRL | IRL p50 | IRL p90 | APT | Non-break FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EN | 69.00% | 32.25% | 32.31% | 296 ms | 178 ms | 611 ms | 1.494 s | 41.4% |
| ZH | 81.45% | 14.44% | 14.44% | 457 ms | 260 ms | 876 ms | 0.839 s | 27.8% |
| Noise/Silence | n/a | 4.20% | 4.20% | n/a | n/a | n/a | 0.424 s | 4.2% |

The CUDA backend processed 5,991.68 s EN, 6,161.45 s ZH, and 6,487.96 s noise
audio at model-compute RTF `0.1949`, `0.1934`, and `0.1592`, respectively.
Independent scoring reproduced every stored metric exactly, with
1,600/1,600/500 prediction rows and matching per-clip trace caches. Compared
with the frozen Realtime-VAP fallback, DualTurn improves recall, FIR, APT, and
noise robustness on both languages, but has worse mean response latency; the
trade-off must remain visible rather than collapsed into a single score.
Artifacts are stored under:

- `results/en/sidbench/dualturn_single_channel_fallback/20260905T095000Z/`
- `results/zh/sidbench/dualturn_single_channel_fallback/20260905T095000Z/`
- `results/shared/sidbench/dualturn_single_channel_fallback/20260905T095000Z/`

Isolated runtime profiles were collected sequentially with concurrency 1 on an
Intel Xeon E5-2630 v3 host and, for CUDA rows, one otherwise-idle RTX A5000.
Each profile used 64 warmup frames and 512 measured frames. CPU rows use four
threads; CUDA rows use one host thread. Cold first-tick values have one sample
per fresh Python process with warm OS/Hugging Face caches, so they are
diagnostic point measurements rather than cold-start distributions.

| Model | Backend | Constructed | First output | Cold tick | Warm p50 | Warm p90 | Warm p95 | RTF | CPU | RAM | VRAM | Artifacts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DualTurn | CPU, 4 threads | 2.57 s | 2.62 s | 53 ms | 66.8 ms | 85.6 ms | 88.2 ms | 0.867 | 386.6% | 487 MiB | n/a | 154 MiB |
| DualTurn | A5000 CUDA | 5.41 s | 6.04 s | 628 ms | 13.5 ms | 18.5 ms | 20.0 ms | 0.177 | 98.2% | 1,177 MiB | 742 MiB | 154 MiB |
| Realtime-VAP | CPU, 4 threads | 2.89 s | 2.98 s | 91 ms | 22.1 ms | 29.3 ms | 30.9 ms | 0.236 | 392.2% | 653 MiB | n/a | 30 MiB |
| Realtime-VAP | A5000 CUDA | 3.45 s | 4.29 s | 844 ms | 21.9 ms | 28.3 ms | 30.0 ms | 0.232 | 97.7% | 1,249 MiB | 322 MiB | 30 MiB |

DualTurn's four-thread CPU p90 exceeds its 80 ms streaming interval, while the
CUDA row has substantial scheduling headroom at the cost of slower cold start
and 742 MiB process VRAM. Realtime-VAP gains effectively no warm-tick or RTF
benefit from CUDA on this host, so its CPU backend is the more economical
deployment choice. The validated comparison is stored under
`results/runtime/20260905T103200Z/`.

The current unified EN/ZH comparison report selects 39 normalized rows: eight
LiveKit eot-bench rows, 18 TurnBench task rows, six SID-Bench rows, two FastTurn
binary-EOT rows, one native four-state row, and four isolated runtime rows. It
hashes 65 source artifacts and records every excluded partial, diagnostic,
intermediate, or superseded run. The report keeps native/derived and
single/dual-channel modes visible and explicitly refuses a single overall
score. Canonical outputs are stored under
`results/comparison/20260906T111447Z/`.

TurnBench dev access was verified for the pinned 38-conversation revision. The
pinned official scorer reproduced `1904+/1063-` EOT events, `347+/3733-` INT
events, and `545/902` exclusions. Official oracle predictions scored recall 1.0,
FPR 0.0, and zero latency on both tasks.

The official committed dev artifacts also reproduce through the bridge:

| Model | EOT recall | EOT FPR | INT recall | INT FPR |
|---|---:|---:|---:|---:|
| SmartTurn v3 | 75.37% | 9.97% | 11.82% | 9.59% |
| VapGPT OTO | 84.09% | 4.52% | 95.68% | 9.99% |

The SmartTurn row is an upstream artifact parity check. The VapGPT values are
also reproduced by the new full local model inference described below.

The official RMS VAD floor was also executed locally over all 38 conversations.
Its generated prediction JSON is exactly equal to the upstream committed artifact:

| Model / source | Task | Recall | FPR | Latency p50 | Latency p90 |
|---|---|---:|---:|---:|---:|
| RMS VAD (new exact local run) | EOT | 59.45% | 54.66% | -98 ms | 2161 ms |
| RMS VAD (new exact local run) | INT | 99.42% | 38.98% | 137 ms | 602 ms |

The run processed 26,306.87 seconds of dual-channel conversation audio at model
compute RTF `0.000637` and is stored under
`results/en/turnbench/rms_vad/20260904T182738279470Z-9d4327e3/`.

SmartTurn v3.1 and v3.2 were then run with the same upstream-style 12.5 Hz
Silero-gated timeline policy and the unquantized GPU ONNX artifacts:

| Model | Task | Threshold | Recall | FPR | Latency p50 | Latency p90 |
|---|---|---:|---:|---:|---:|---:|
| SmartTurn v3.1 GPU (new run) | EOT | 0.009208 | 73.48% | 9.97% | 1012 ms | 1225 ms |
| SmartTurn v3.1 GPU (new run) | INT derived | 0.015734 | 12.10% | 9.99% | 173 ms | 703 ms |
| SmartTurn v3.2 GPU (new run) | EOT | 0.007573 | 76.63% | 9.88% | 1017 ms | 1353 ms |
| SmartTurn v3.2 GPU (new run) | INT derived | 0.017108 | 10.09% | 8.95% | 122 ms | 289 ms |

The v3.1 EOT recall is 1.90 percentage points below the committed upstream
artifact's 75.37%, while INT differs by 0.28 points. The official SmartTurn
pipeline resets each Silero instance every five wall-clock seconds, so its VAD
boundaries depend on worker scheduling. In addition, concurrent calls into the
shared CUDA endpoint session produced one invalid negative value locally; this
backend serializes endpoint calls and records that execution policy. Therefore
the v3.1 result is a new pinned local run, not an exact upstream reproduction.
The two comparable local runs show that v3.2 improves EOT recall by 3.15 points
at the same dev FPR budget, while neither version has a native interruption head.

Artifacts:

- `results/en/turnbench/smart_turn_v3_1_gpu/20260904T184500Z/`
- `results/en/turnbench/smart_turn_v3_2_gpu/20260904T190100Z/`

The exact official VapGPT OTO path was then run locally over all 38 conversations
using the pinned 50 Hz CPC/VAP model and official 20-second-context, 5-second-step
windowing policy:

| Model | Task | Threshold | Recall | FPR | Latency p50 | Latency p90 |
|---|---|---:|---:|---:|---:|---:|
| VapGPT OTO (new exact run) | EOT derived | 0.916147 | 84.09% | 4.52% | 463 ms | 1657 ms |
| VapGPT OTO (new exact run) | INT floor-state | 0.859117 | 95.68% | 9.99% | 896 ms | 2002 ms |

The two task artifacts contain 2,630,646 probabilities each. Every value is
within `5.01e-5` of the upstream four-decimal reference, and the selected
thresholds, event counts, recall, FPR, and latency quantiles reproduce exactly.
Two physical GPUs processed disjoint Parquet shards; aggregate model-compute RTF
was `0.02008`. The merged run, scores, and parity record are under
`results/en/turnbench/vap_gpt_oto/20260904T192700Z/`.

DualTurn Endpointing was run over the same 38 conversations using the pinned
80 ms causal streaming ONNX graph. Because its EOT head only targets channel 0,
each conversation was evaluated twice with swapped channel orientation. The
interruption row is not a native interruption head; it sweeps the explicitly
derived `FVAD[speaker, 0-240 ms]` signal.

| Model | Task | Threshold | Recall | FPR | Latency p50 | Latency p90 |
|---|---|---:|---:|---:|---:|---:|
| DualTurn Endpointing | EOT native | 0.976409 | 49.53% | 9.69% | 539 ms | 1282 ms |
| DualTurn Endpointing | INT derived FVAD | 0.989633 | 96.25% | 9.99% | 548 ms | 1681 ms |

The native EOT result is substantially weaker on TurnBench human-human dialogue
than the model card's published single-channel eot-bench result; those protocols
must not be conflated. The strong interruption row is useful architecture
evidence but remains derived. Three GPU shards processed 26,306.87 seconds of
audio at aggregate two-orientation compute RTF `0.29065`. The final artifacts
and independent exact rescore are under
`results/en/turnbench/dualturn_endpointing/20260904T210300Z/`.

ESPnet TurnTaking mixed-mono was run over all 38 conversations using the pinned
Whisper-medium Switchboard checkpoint and the official causal prefix/30-second
sliding-window recipe. Its native five-way turn-state probabilities come from a
single mono mix; both task signals use the official causal energy heuristic to
attribute events back to a TurnBench speaker.

| Model | Task | Threshold | Recall | FPR | Latency p50 | Latency p90 |
|---|---|---:|---:|---:|---:|---:|
| ESPnet TurnTaking mixed-mono | EOT, derived speaker attribution | 0.000869 | 83.72% | 8.84% | 922 ms | 2135 ms |
| ESPnet TurnTaking mixed-mono | INT, derived speaker attribution | 0.20 | 64.27% | 9.08% | 227 ms | 1428 ms |

The local TF32 sweep selected a different EOT quantile candidate from the
committed upstream artifact. At the upstream fixed threshold `0.0007176334`,
local versus upstream dev EOT is `83.51%/7.53%` versus `83.61%/7.43%`
(recall/FPR), with identical 895 ms p50 latency and only 0.9 ms p90 difference.
At the shared interruption threshold `0.20`, recall differs by 0.58 percentage
points, FPR by 0.05 points, p50 is identical, and p90 differs by 2.4 ms. Thus
the inference/timing harness passes the stated tolerance, but this TF32 run is
not claimed as bit-exact probability parity.

Each task artifact contains 1,315,302 validated speaker-frame probabilities.
Three shards processed 26,306.87 seconds of audio at aggregate compute RTF
`1.39788`; the 1.23 GB checkpoint occupied about 14.1 GB VRAM per A5000 during
inference. Artifacts and the independently reproduced scores are under
`results/en/turnbench/espnet_turntaking/20260904T223000Z-final/`.

Mimi Endpointer was run over all 38 conversations using the official
TurnBench-fine-tuned `oto_d1f` checkpoint. It encodes the two 24 kHz speaker
channels independently with Mimi, then applies shared unidirectional LSTMs to
produce a five-way floor-state distribution at 25 Hz.

| Model | Task | Threshold | Recall | FPR | Latency p50 | Latency p90 |
|---|---|---:|---:|---:|---:|---:|
| Mimi Endpointer | EOT derived | 0.923188 | 75.95% | 4.52% | 752 ms | 1862 ms |
| Mimi Endpointer | INT floor-state | 0.923886 | 87.61% | 9.78% | 821 ms | 2101 ms |

The table uses independently selected local dev operating points. At the
advertised upstream thresholds (`0.92` EOT, `0.9224` INT), interruption metrics
reproduce exactly. EOT FPR is exact and differs by only one true positive at a
floating-point threshold boundary; recall and all latency quantiles pass the
planned `±0.01` / `±30 ms` harness tolerance. Each task artifact contains
1,315,302 finite probabilities. Model inference processed 26,306.87 seconds of
audio in 147.44 seconds (RTF `0.00560`, about 178× realtime), excluding 62.45
seconds of resampling. Artifacts, independent rescore, and parity evidence are
under `results/en/turnbench/mimi_endpointer/20260905T061000Z/`.

Kyutai Semantic VAD was run over all 38 conversations through the official
native streaming path. Both speaker channels are batched together through the
24 kHz Mimi encoder and the STT-1B model, one 80 ms frame at a time. Its VAD
head is native EOT; interruption is explicitly derived as `1 - P(turn ending)`.

| Model | Task | Threshold | Recall | FPR | Latency p50 | Latency p90 |
|---|---|---:|---:|---:|---:|---:|
| Kyutai Semantic VAD | EOT native head | 0.832031 | 79.67% | 9.97% | 1099 ms | 1767 ms |
| Kyutai Semantic VAD | INT derived | 0.973511 | 93.08% | 8.52% | 579 ms | 1410 ms |

The table uses independently selected local dev operating points. At the
advertised upstream thresholds, EOT and derived INT both pass the planned
recall/FPR ±0.01 and latency ±30 ms tolerances. Matching the official CUDA
resampling order was material: an earlier CPU-resampling diagnostic run missed
INT p90 tolerance by 29.8 ms, while the corrected run differs by 27.2 ms and
passes. The upstream artifact did not pin its HF model revision, so no bit-exact
claim is made.

Each task artifact contains 657,630 finite probabilities. Three GPU shards
processed 26,306.87 seconds of audio at aggregate compute RTF `0.29992` (about
3.33× realtime), with 2.97–3.07 GB peak allocated VRAM per shard. Artifacts and
validation evidence are under
`results/en/turnbench/kyutai_semantic_vad/20260905T073700Z-final/`. The earlier
`20260905T062300Z-final` CPU-resampling artifact is diagnostic only.

## Newly executed TurnBench baseline

The pinned MaAI trilingual Realtime-VAP checkpoint was run over all 38 dev
conversations (7.31 hours) in native dual-channel streaming mode. EOT and
interruption are derived signals (`1 - p_now[speaker]` and `p_now[speaker]`),
not native task-specific classifier heads. Thresholds were selected independently
on dev using the official highest-recall operating point with FPR at or below 10%.

| Model / source | Task | Threshold | Recall | FPR | Latency p50 | Latency p90 |
|---|---|---:|---:|---:|---:|---:|
| MaAI trilingual CPC 10 Hz (new run) | EOT | 0.850835 | 82.41% | 9.97% | 333 ms | 1628 ms |
| MaAI trilingual CPC 10 Hz (new run) | INT | 0.82 | 90.78% | 9.48% | 1075 ms | 2210 ms |
| TurnBench VapGPT OTO (new exact run) | EOT | 0.916147 | 84.09% | 4.52% | 463 ms | 1657 ms |
| TurnBench VapGPT OTO (new exact run) | INT | 0.859117 | 95.68% | 9.99% | 896 ms | 2002 ms |

The MaAI and TurnBench VapGPT rows are different model/checkpoint pipelines.
The VapGPT row is the exact local official reproduction above; the comparison
to MaAI is architectural context. The MaAI run uses version `0.2.18` and checkpoint revision
`1a7721ad12a26ac8cd6025bec5543f9c005b9f51`, a 5-second context, and 10 Hz
frame-end timestamps. Its weights are CC-BY-NC-SA-4.0, so it is a research
reference rather than a permissively licensed production candidate.

The three inference shards produced identical schema/frame validation over
526,108 speaker-frames per task. Aggregate model-compute RTF was `0.2274`.
The committed predictions were rescored independently and matched the sweep
summary exactly. Local artifacts are under
`results/en/turnbench/realtime_vap_tri/20260904T142500Z/final/`.

The recorded resampling time belongs to the original scalar implementation used
for this run. The implementation is now vectorized; a 60-second 48-to-16 kHz
regression input takes about 0.046 seconds locally and produces exactly the same
integer-ratio samples. Model inference RTF above is unaffected by that change.

## Canonical benchmark figures

The comparison report now has a deterministic plotting layer. It renders the
complete LiveKit eot-bench policy Pareto frontier separately for EN and ZH and
marks the prescribed 300/600 ms operating points. Native, single-channel
fallback, and diagnostic oracle-VAD rows remain visibly identified.

SID-Bench is intentionally rendered as a frozen-policy recall/FIR scatter with
IRL p50 encoded by bubble size, not as a Pareto frontier. The plot command reads
only the selected summary point and never recalibrates a threshold on SID test
data. The immutable artifact contains the three PNG files, exact plot data, and
SHA256 hashes for all seven plotting source inputs under
`results/figures/20260905T113616Z/`.

## FastTurn TurnSense baseline

The first real open-weight Chinese FastTurn EOT baseline is complete. TurnSense
INT8 (`92924ae57f61b59cc6195d1367f8c705b2ba06b4`) ran all 18,352
Complete/Incomplete clips using audio only and the upstream-parity SoXR-HQ +
Kaldi frontend.

| Metric | Result |
|---|---:|
| Complete precision / recall / F1 | 93.07% / 79.08% / 85.51% |
| Incomplete precision / recall / F1 | 50.07% / 73.26% / 59.48% |
| Macro-F1 / weighted-F1 | 72.50% / 80.34% |
| Incomplete → Complete | 23.77% |
| Invalid abstention | 2.85% |
| CPU p50 / p90 / p95 | 35.6 / 84.5 / 110.3 ms |
| RTF | 0.0193 |

The immutable run is under
`results/zh/fastturn/eot/turnsense-int8/20260905T113151624921Z-a5893f4c/`.
It closes binary EOT coverage only: TurnSense does not support Backchannel or
Wait, so FastTurn real/full remain unsupported rather than heuristically mapped.
The clip runner now streams waveforms lazily and permits adapter-owned resampling
while accounting for it separately; this removes the earlier ~8 GB waveform
preload (a 16-clip verification peaked at 332 MB) and lets TurnSense match the
upstream SoXR-HQ path.

## FastTurn Easy Turn baseline

The native four-state Easy Turn baseline is complete over all 22,432 FastTurn
clips. It uses the pinned upstream Wenet/LLM-ASR generation path at batch size
one; a four-example parity check reproduces the previous golden raw generations
exactly. Four inputs beyond Easy Turn's 30-second positional-encoding limit are
retained as incorrect abstentions rather than truncated or removed. Two more
generations have no valid terminal tag and are also retained as abstentions.

| Metric | Result |
|---|---:|
| Complete / Incomplete F1 | 84.28% / 55.60% |
| Backchannel / Wait F1 | 81.65% / 96.10% |
| Macro-F1 / weighted-F1 | 79.41% / 79.79% |
| Incomplete → Complete | 25.75% |
| Backchannel → Complete | 2.82% |
| Abstention | 0.027% (6 / 22,432) |
| GPU p50 / p90 / p95 | 416.5 / 818.1 / 1027.5 ms |
| RTF | 0.244 |

The immutable run is under
`results/zh/fastturn/full/easy-turn/20260905T220700516857Z-a9d86d2a/`.
Easy Turn and FastTurn are authored by the same research team and train/test
overlap cannot be audited from the public artifacts, so this row is not treated
as an independent generalization estimate.

## FastTurn EOT comparison

SmartTurn v3.2 CPU is complete over the same 18,352 Complete/Incomplete clips as
TurnSense. Its EOT probability is mapped with the predeclared upstream-default
threshold `0.5`; no FastTurn test labels are used for threshold selection.

| Model | Complete F1 | Incomplete F1 | Macro-F1 | Incomplete→Complete | CPU p50 | RTF | Size |
|---|---:|---:|---:|---:|---:|---:|---:|
| TurnSense INT8 | 85.51% | 59.48% | 72.50% | 23.77% | 35.6 ms | 0.0193 | 52.2 MB |
| SmartTurn v3.2 CPU, threshold 0.5 | 76.48% | 34.14% | 55.31% | 54.74% | 164.7 ms | 0.0694 | 8.68 MB |

SmartTurn has the smaller artifact but is slower and substantially less accurate
at its fixed default threshold on isolated FastTurn clips. This does not justify
post-hoc threshold tuning on FastTurn test; any calibrated row must use an
external calibration set and remain separately labeled.

The immutable SmartTurn run is under
`results/zh/fastturn/eot/smart-turn-v3.2-cpu/20260905T235530638173Z-71b3330e/`.

## DualTurn eot-bench and paired Chinese analysis

DualTurn completed the full 400-turn English and 344-turn Chinese eot-bench
subsets through its native 80 ms streaming graph. English processed 5,040.90 s
of audio at RTF `0.168250`; Chinese processed 4,029.17 s at RTF `0.168408`.
The English row is supported by the model card; Chinese is explicitly marked
`unverified_zero_shot`. Both use `user=CH0, silence=CH1`, so neither is presented
as native-dual-channel evaluation.

The paired FastTurn analysis finds a +12.59-point TurnSense accuracy advantage
over SmartTurn (95% paired bootstrap interval +11.72 to +13.49), driven by a
+28.00-point advantage on Incomplete clips. TurnSense is 0.71 points worse on
clips at most one second long. A naive SmartTurn fallback for TurnSense's 523
abstentions improves aggregate accuracy to 79.47% but worsens
Incomplete-to-Complete from 23.77% to 25.56%, so it is not adopted as a primary
policy. Full interpretation is recorded in `docs/BASELINE_ANALYSIS.md`.

## Next

The schema-v2 unified comparison at
`results/comparison/20260906T111447Z/` contains 40 selected rows. FastTurn EOT
has two rows (TurnSense and SmartTurn) and FastTurn Full has one native
four-state row (Easy Turn); they render in separate tables and normalized CSV
columns. DualTurn EN/ZH eot-bench rows and refreshed EOT Pareto plots are now
included. The manifest hashes every selected input, and the coverage audit
reports both suites complete without implying that their scores are comparable.

1. Audit whether an external non-FastTurn Chinese calibration split exists for
   SmartTurn; otherwise keep `0.5` as the only primary FastTurn row.
2. Define an externally calibrated two-stage interruption policy: fast acoustic
   acceptance for interruption-at-beginning and conservative semantic confirmation
   for middle-turn speech.
3. Evaluate a HOLD-safe TurnSense invalid/short-utterance policy on an external
   calibration split before considering a hybrid production row.

X2-Turn is deferred by project decision because its 10.47 GB BF16 checkpoint
and two-pass reference inference are too expensive for the first production
candidate set. Its verified adapter remains available but no full baseline run
is planned in this phase.
