# Turn Detection Benchmark — Session Handoff

Last updated: 2026-09-06 UTC (baseline committed to `main`)

This is the primary handoff document for the next working session. Read this file first, then use the linked protocol and analysis documents for implementation details.

## 1. Current status at a glance

- The benchmark harness, dataset adapters, model adapters, metrics, reports, runtime probes, and CI fixtures are implemented.
- The complete test suite passes: **128 passed** with the `smart-turn` extra installed. A plain `uv sync --frozen`, which is what CI does, gives 127 passed and 1 skipped.
- No benchmark or model process was running at the time of handoff.
- The canonical cross-benchmark report is in [`results/comparison/20260906T161125Z/`](results/comparison/20260906T161125Z/).
- The current research interpretation is in [`docs/BASELINE_ANALYSIS.md`](docs/BASELINE_ANALYSIS.md).
- English and Chinese component suites have been run for the selected open-weight baselines.
- **X2-Turn is explicitly deferred** because the released checkpoint is very large. Do not resume a full X2-Turn run unless the user reverses this decision.
- Whole-agent benchmarks and model training are intentionally outside the current core scope.

Important repository-state warning:

- The implementation is committed on `main`. `origin/main` is still at the original commit `cb22646`; nothing has been pushed.
- Raw benchmark run directories under `results/` are **not** committed and exist only on this machine. Preserve `results/` separately if moving or cloning the work.
- [`results/ARCHIVE_MANIFEST.json`](results/ARCHIVE_MANIFEST.json) indexes that tree: 12,587 files, 0.963 GiB, 53 completed runs, every file hashed. Regenerate it with `uv run python scripts/archive_manifest.py --output results/ARCHIVE_MANIFEST.json` after a restore to prove the copy is identical.
- Only the canonical comparison, analysis, and runtime artifacts listed in section 8 are tracked in Git, as a provenance record.
- Do **not** run `git reset --hard`, `git checkout --`, `git clean`, or otherwise discard the working tree; untracked run artifacts are not recoverable from Git.

## 2. Required environment conventions

Use `uv` for every Python environment and command. Before Hugging Face downloads or model execution, set:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
```

The project deliberately isolates incompatible model dependencies under `environments/`. Do not try to install SmartTurn, VAP, ESPnet, Mimi, DualTurn, Easy Turn, and other model stacks into one shared environment.

Common verification commands:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
uv sync --extra dev
uv run pytest -q
uv run turn-eval --help
```

## 3. Research goal and scope

The repository evaluates turn detection and turn management components for live voice agents deployed through LiveKit Agents, Pipecat, or other streaming ASR → LLM → TTS pipelines.

The research questions are:

1. Which available model is the best fit for a Chinese live voice agent?
2. If current models are insufficient, which architecture is the best candidate for Chinese fine-tuning or retraining?

This repository covers:

- inference;
- dataset normalization;
- causal evaluation;
- calibration and threshold policy;
- metrics;
- runtime measurement;
- artifact provenance;
- reporting.

This repository does **not** cover model training or fine-tuning. Any future training work should live in a separate repository and use this repository as the fixed evaluation harness.

## 4. Non-negotiable benchmark principles

### 4.1 Separate EOT from interruption

Turn management contains two different problems and must not collapse them into one score:

- **EOT / HOLD:** while the agent listens, determine whether the user completed the turn or is pausing mid-turn.
- **Interruption / backchannel:** while the agent speaks, distinguish a genuine barge-in from acknowledgements such as “嗯嗯”.

### 4.2 No single overall score

Do not create a single `TurnScore`. Maintain separate leaderboards for:

- EOT false-cutoff versus conversational latency;
- interruption recall, FIR, response latency, and APT;
- turn-state per-class and macro metrics;
- runtime and resource use.

### 4.3 Strict causality

A prediction timestamp is the time by which the model has observed all information required to make the prediction. Audio, transcript, agent audio, lookahead, or future VAD after timestamp `t` must not be used for a decision reported at `t`.

Algorithmic lookahead is part of conversational latency. See [`docs/TURNBENCH_PROTOCOL.md`](docs/TURNBENCH_PROTOCOL.md) and the causality tests.

### 4.4 Separate latency concepts

Always distinguish:

- conversational or algorithmic latency;
- inference wall-clock latency;
- end-to-end detector latency, including VAD delay and policy delay.

### 4.5 Honest capability and evaluation labels

Never equate a native model output with a heuristic mapping. Results must preserve:

- `SUPPORTED`, `DERIVED`, or `UNSUPPORTED` capability status;
- native single-channel or native dual-channel mode;
- single-channel fallback;
- derived interruption or derived state;
- streaming, offline, or cloud/API execution;
- supported language versus unverified zero-shot use.

For example, DualTurn and VAP SID-Bench results are derived from future-activity signals in single-channel fallback mode; they are not native semantic interruption heads.

### 4.6 Immutable datasets and artifacts

- Never permanently preprocess or overwrite a dataset.
- Resample inside the model adapter and record the cost.
- Freeze thresholds using a development or calibration split before test evaluation.
- Never tune a threshold or mapping on the reported test split.
- Completed run directories are immutable. Produce a new run instead of editing an old artifact.

## 5. Intended benchmark suites

### English core suite

- TurnBench: architecture validation for EOT and interruption.
- LiveKit eot-bench EN: production-oriented EOT/HOLD trade-off.
- SID-Bench EN: genuine interruption versus non-interruptive speech.

### Chinese core suite

- LiveKit eot-bench ZH: EOT/HOLD.
- FastTurn Test Set: complete, incomplete, backchannel, and wait states.
- SID-Bench ZH: interruption and backchannel rejection.

### Secondary and later work

- Easy Turn Testset is a secondary/legacy Chinese dataset. The Easy Turn model has been evaluated on FastTurn, but the separate Easy Turn Testset benchmark has not been made part of the canonical report.
- HumDial-FDBench and Full-Duplex-Bench are whole-agent benchmarks for phase 2. They measure the combined ASR, LLM, TTS, detector, and agent policy and must not be mixed into the component-model leaderboard.

## 6. Model selection decisions

The user narrowed the current baseline scope to models with openly available weights.

Primary open-weight coverage implemented or evaluated:

- SmartTurn v3.2: compact semantic endpointing reference.
- Realtime-VAP multilingual: future voice-activity prediction.
- DualTurn Endpointing: causal dual-channel endpointing, evaluated in single-channel fallback where necessary.
- TurnSense INT8: Chinese/English semantic EOT model.
- Easy Turn: Chinese four-state reference.
- TurnBench architecture validation also includes exact VapGPT OTO, ESPnet TurnTaking, Mimi Endpointer, and Kyutai Semantic VAD.

Explicit decisions:

- **X2-Turn is deferred.** Although called “4B”, the downloaded release contains about 5.235B stored parameters and a 10,470,070,544-byte BF16 weight file. The released local path is two-pass and needs substantial GPU memory; it is not currently worth the execution cost.
- A verified X2 adapter, protocol, and partial smoke artifact remain in the repository for future use. The smoke artifact is excluded from canonical comparison.
- Cloud-only OpenAI and Gemini baselines were not run because the current scope is open weights.
- Moshi is not prioritized because it is a much heavier whole-agent-like execution path.
- WavLM variants remain optional. The public checkpoints can be added if architecture completeness becomes more important, but their checkpoint licensing needs explicit audit.
- ESPnet per-channel evaluation is a lower-priority ablation, not required for the current baseline report.

The complete TurnBench coverage audit is in [`docs/TURNBENCH_BASELINE_AUDIT.md`](docs/TURNBENCH_BASELINE_AUDIT.md).

## 7. Implemented repository areas

The implementation includes:

- typed examples, predictions, capabilities, manifests, timing, and audio utilities;
- streaming and pointwise adapter interfaces;
- eot-bench, TurnBench, FastTurn, Easy Turn, and SID-Bench dataset/metric logic;
- official-style threshold selection and Pareto summaries;
- isolated model runners under `environments/`;
- immutable run artifacts containing `manifest.json`, `predictions.parquet`, `metrics.json`, and `runtime.json` where applicable;
- report generation without an overall score;
- EOT and SID error-analysis commands;
- synthetic CI fixtures for timing, causality, matching, latency, resampling, manifests, and adapters.

Important documentation:

- [`docs/SPEC.md`](docs/SPEC.md): repository and interface specification.
- [`docs/STATUS.md`](docs/STATUS.md): detailed implementation and run status.
- [`docs/BASELINE_ANALYSIS.md`](docs/BASELINE_ANALYSIS.md): cross-benchmark interpretation and recommendations.
- [`docs/BENCHMARK_COVERAGE_AUDIT.md`](docs/BENCHMARK_COVERAGE_AUDIT.md): which model/benchmark cells are measured, blocked, decided, or unrun.
- [`docs/TURNBENCH_PROTOCOL.md`](docs/TURNBENCH_PROTOCOL.md): official TurnBench matching and timing rules.
- [`docs/TURNBENCH_BASELINE_AUDIT.md`](docs/TURNBENCH_BASELINE_AUDIT.md): evaluated and missing TurnBench models, audited against the official roster.
- [`docs/FASTTURN_PROTOCOL.md`](docs/FASTTURN_PROTOCOL.md): FastTurn state mappings and metrics.
- [`docs/SIDBENCH_PROTOCOL.md`](docs/SIDBENCH_PROTOCOL.md): SID scoring and mapping caveats.
- [`docs/RUNTIME_PROTOCOL.md`](docs/RUNTIME_PROTOCOL.md): runtime measurement protocol.
- [`docs/X2_TURN_PROTOCOL.md`](docs/X2_TURN_PROTOCOL.md): deferred X2-Turn integration details.

New analysis code:

- [`src/turn_eval/reports/eot_analysis.py`](src/turn_eval/reports/eot_analysis.py)
- [`src/turn_eval/reports/sid_analysis.py`](src/turn_eval/reports/sid_analysis.py)
- CLI commands: `turn-eval analyze-eotbench` and `turn-eval analyze-sidbench`.

## 8. Canonical artifacts

### 8.1 Unified comparison report

Canonical directory: [`results/comparison/20260906T161125Z/`](results/comparison/20260906T161125Z/)

It contains:

- `report.md`;
- `summary.csv`;
- `summary.json`;
- `manifest.json`;
- English and Chinese EOT Pareto figures;
- SID trade-off figures;
- 39 normalized comparison rows;
- hashes for 65 selected input artifacts.

The comparison manifest intentionally excludes partial and smoke runs. Do not select a result merely because its directory contains `SUCCESS`; use the canonical comparison manifest or inspect completeness and protocol conformance.

### 8.2 Analysis artifacts

- EOT analysis: [`results/analysis/eotbench/20260906T083000Z/`](results/analysis/eotbench/20260906T083000Z/)
- SID analysis: [`results/analysis/sidbench/20260906T080000Z/`](results/analysis/sidbench/20260906T080000Z/)
- Runtime comparison: [`results/runtime/20260905T103200Z/`](results/runtime/20260905T103200Z/)

### 8.3 Selected EOT runs

| Language | Model | Run |
|---|---|---|
| EN | DualTurn fallback | `results/en/eotbench/dualturn_single_channel_fallback/20260906T065528310380Z-1af1ceaa` |
| EN | Realtime-VAP fallback | `results/en/eotbench/realtime_vap_single_channel_fallback/20260904T173254860352Z-365fc311` |
| EN | SmartTurn v3.2 | `results/en/eotbench/smart_turn_v3.2_cpu/20260904T132526687730Z-d6672f41` |
| EN | Fixed-timeout oracle-VAD diagnostic | `results/en/eotbench/fixed_timeout_oracle_vad/20260904T131955441085Z-dc2d6a7b` |
| ZH | DualTurn fallback | `results/zh/eotbench/dualturn_single_channel_fallback/20260906T065234992726Z-9511ff53` |
| ZH | Realtime-VAP fallback | `results/zh/eotbench/realtime_vap_single_channel_fallback/20260904T171258618568Z-3112c7e7` |
| ZH | SmartTurn v3.2 | `results/zh/eotbench/smart_turn_v3.2_cpu/20260904T132512978680Z-b98439c4` |
| ZH | Fixed-timeout oracle-VAD diagnostic | `results/zh/eotbench/fixed_timeout_oracle_vad/20260904T131954888548Z-25979a77` |

### 8.4 Selected TurnBench runs

| Model | Run |
|---|---|
| DualTurn Endpointing | `results/en/turnbench/dualturn_endpointing/20260904T210300Z` |
| ESPnet TurnTaking | `results/en/turnbench/espnet_turntaking/20260904T223000Z-final` |
| Kyutai Semantic VAD | `results/en/turnbench/kyutai_semantic_vad/20260905T073700Z-final` |
| Mimi Endpointer | `results/en/turnbench/mimi_endpointer/20260905T061000Z` |
| Realtime-VAP multilingual | `results/en/turnbench/realtime_vap_tri/20260904T142500Z/final` |
| SmartTurn v3.1 | `results/en/turnbench/smart_turn_v3_1_gpu/20260904T184500Z` |
| SmartTurn v3.2 | `results/en/turnbench/smart_turn_v3_2_gpu/20260904T190100Z` |
| Exact VapGPT OTO | `results/en/turnbench/vap_gpt_oto/20260904T192700Z` |
| RMS VAD diagnostic | `results/en/turnbench/rms_vad/20260904T182738279470Z-9d4327e3` |

### 8.5 Selected FastTurn runs

| Task | Model | Run |
|---|---|---|
| 2-state EOT | SmartTurn v3.2 | `results/zh/fastturn/eot/smart-turn-v3.2-cpu/20260905T235530638173Z-71b3330e` |
| 2-state EOT | TurnSense INT8 | `results/zh/fastturn/eot/turnsense-int8/20260905T113151624921Z-a5893f4c` |
| 4-state | Easy Turn | `results/zh/fastturn/full/easy-turn/20260905T220700516857Z-a9d86d2a` |

### 8.6 Selected SID-Bench runs

| Split | Model | Run |
|---|---|---|
| EN | DualTurn fallback | `results/en/sidbench/dualturn_single_channel_fallback/20260905T095000Z` |
| EN | Realtime-VAP fallback | `results/en/sidbench/realtime_vap_single_channel_fallback/20260905T090900Z` |
| ZH | DualTurn fallback | `results/zh/sidbench/dualturn_single_channel_fallback/20260905T095000Z` |
| ZH | Realtime-VAP fallback | `results/zh/sidbench/realtime_vap_single_channel_fallback/20260905T090900Z` |
| Noise | DualTurn fallback | `results/shared/sidbench/dualturn_single_channel_fallback/20260905T095000Z` |
| Noise | Realtime-VAP fallback | `results/shared/sidbench/realtime_vap_single_channel_fallback/20260905T090900Z` |

### 8.7 Selected runtime files

- `results/runtime/dualturn/20260905T103200Z-cpu.json`
- `results/runtime/dualturn/20260905T103200Z-cuda-a5000.json`
- `results/runtime/realtime_vap/20260905T103200Z-cpu.json`
- `results/runtime/realtime_vap/20260905T103200Z-cuda-a5000.json`

## 9. Main empirical results

All percentages below are percentage points. Consult the immutable JSON and Parquet artifacts before quoting additional precision.

### 9.1 LiveKit eot-bench

| Language | Model | False cutoff @300 ms | False cutoff @600 ms | Mean latency @5% cutoff | Mean latency @10% cutoff |
|---|---:|---:|---:|---:|---:|
| EN | DualTurn fallback | 22.13% | 9.65% | 991 ms | 566 ms |
| EN | Realtime-VAP fallback | 33.33% | 13.62% | 1016 ms | 702 ms |
| EN | SmartTurn v3.2 | 37.59% | 14.75% | 1044 ms | 755 ms |
| ZH | DualTurn fallback | 23.36% | 10.20% | 910 ms | 612 ms |
| ZH | Realtime-VAP fallback | 31.25% | 9.70% | 879 ms | 597 ms |
| ZH | SmartTurn v3.2 | 41.78% | 12.83% | 928 ms | 667 ms |
| EN | Realtime-VAP, 20 s context | 30.35% | 14.04% | 1184 ms | 753 ms |
| ZH | Realtime-VAP, 20 s context | 25.66% | 9.87% | 910 ms | 590 ms |
| EN | TurnSense INT8 (zero-shot) | 55.60% | 21.70% | 1608 ms | 942 ms |
| ZH | TurnSense INT8 | 49.18% | 15.79% | 990 ms | 719 ms |

The 20 s rows are a controlled checkpoint ablation on the same pinned revision,
frame rate held at 10 Hz. It was run to test whether the long-HOLD failures
below are a context-length problem. **They are not:** both 2-5 s slices stay at
100% and the Chinese 1-2 s slice worsens. The gain is on ordinary 300-500 ms
pauses, is significant only in Chinese (-5.59 pp, CI -9.53 to -1.84), and costs
up to 168 ms of latency at the 5% operating point. On Chinese it does make VAP
statistically indistinguishable from DualTurn at FC@300 (+2.30 pp, CI -1.97 to
+6.67). See [`docs/BASELINE_ANALYSIS.md`](docs/BASELINE_ANALYSIS.md).

Interpretation:

- DualTurn is the strongest current production EOT candidate among the compared models.
- The Chinese DualTurn result is an **unverified zero-shot** result; the model officially supports the English configuration used for the reference comparison.
- The local English DualTurn result is close to, but not identical to, its published model-card result: local minus card is approximately +0.33 pp at 300 ms, -0.55 pp at 600 ms, -124 ms at 5%, and -44 ms at 10%. Scorer or inference-path differences should be investigated before claiming exact reproduction.
- Timeout policies are not identical. EN DualTurn used a 3000 ms timeout while VAP and SmartTurn used 1000 ms, so part of its English advantage is policy-dependent. In ZH, DualTurn used 1000 ms versus VAP at 1500 ms and still performed well.

HOLD error analysis at the 300 ms operating point:

- EN false HOLD cutoffs: DualTurn 156, VAP 235, SmartTurn 265.
- ZH false HOLD cutoffs: DualTurn 142, VAP 190, SmartTurn 254.
- Paired clustered difference, DualTurn minus VAP: EN -11.21 pp, 95% CI [-14.71, -7.57]; ZH -7.89 pp, CI [-11.60, -4.35].
- DualTurn minus SmartTurn: EN -15.46 pp, CI [-19.47, -11.61]; ZH -18.42 pp, CI [-22.76, -14.03].
- All three models remain unsafe on long 1–2 second HOLD pauses. EN false-cutoff rates were 62.3%, 90.2%, and 90.2% for DualTurn, VAP, and SmartTurn; ZH rates were 77.8%, 80.0%, and 82.2%.

### 9.2 TurnBench English development set

The official gated development set contains 38 conversations in the validated run. Test labels are withheld, so these are development scores with thresholds selected on development data, not public test claims.

| Model/task | Recall | FPR | Latency p50 | Latency p90 |
|---|---:|---:|---:|---:|
| DualTurn EOT | 49.53% | 9.69% | 539 ms | 1282 ms |
| DualTurn interruption, derived FVAD | 96.25% | 9.99% | 548 ms | 1681 ms |
| ESPnet EOT | 83.72% | 8.84% | 922 ms | 2135 ms |
| ESPnet interruption | 64.27% | 9.08% | 227 ms | 1428 ms |
| Kyutai EOT | 79.67% | 9.97% | 1099 ms | 1767 ms |
| Kyutai interruption | 93.08% | 8.52% | 579 ms | 1410 ms |
| Mimi EOT | 75.95% | 4.52% | 752 ms | 1862 ms |
| Mimi interruption | 87.61% | 9.78% | 821 ms | 2101 ms |
| MaAI VAP EOT | 82.41% | 9.97% | 333 ms | 1628 ms |
| MaAI VAP interruption | 90.78% | 9.48% | 1075 ms | 2210 ms |
| SmartTurn v3.1 EOT | 73.48% | 9.97% | 1012 ms | 1225 ms |
| SmartTurn v3.1 interruption, derived | 12.10% | 9.99% | 173 ms | 703 ms |
| SmartTurn v3.2 EOT | 76.63% | 9.88% | 1017 ms | 1353 ms |
| SmartTurn v3.2 interruption, derived | 10.09% | 8.95% | 122 ms | 289 ms |
| Exact VapGPT OTO EOT | 84.09% | 4.52% | 463 ms | 1657 ms |
| Exact VapGPT OTO interruption | 95.68% | 9.99% | 896 ms | 2002 ms |

The exact VapGPT OTO reproduction matches the official score. Detailed tolerance checks and exceptions are in [`docs/STATUS.md`](docs/STATUS.md).

### 9.3 FastTurn Chinese test set

Two-state EOT, on the identical 18,352 Complete/Incomplete clips:

| Model | Complete F1 | Incomplete F1 | Macro-F1 | Incomplete → Complete | Abstain | p50 inference | RTF | Size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TurnSense INT8 | 85.51% | 59.48% | 72.50% | 23.77% | 2.85% | 35.6 ms | 0.0193 | 52.2 MB |
| Easy Turn | 84.67% | 56.34% | 70.50% | 25.75% | 6.22% | 455 ms | 0.244 | 3.45 GB |
| SmartTurn v3.2 | 76.48% | 34.14% | 55.31% | 54.74% | 0.00% | 164.7 ms | 0.0694 | 8.68 MB |

The Easy Turn row is derived from its completed four-state run by restriction,
without rerunning inference; its RTF is inherited from that run rather than
measured on the subset. TurnSense leads it on every two-state metric at about
one sixty-sixth of the size. Do not compare Easy Turn's four-state Macro-F1 of
79.41% against a two-state Macro-F1: different clip sets, different label
vocabularies, and the ordering inverts on a shared axis.

TurnSense produced 523 abstentions, or 2.85% of examples. On 18,352 paired valid predictions, TurnSense accuracy was 77.93% versus SmartTurn at 65.33%, a +12.59 pp difference with 95% CI [11.72, 13.49]. The gain is mainly on longer examples and incomplete turns; TurnSense was 0.71 pp worse on examples up to one second.

A naive fallback from TurnSense abstention to SmartTurn increases accuracy to 79.47% and macro-F1 to 72.66%, but worsens Incomplete → Complete to 25.56%. Do not adopt this hybrid without an independent calibration set.

Four-state Easy Turn result on FastTurn:

| Complete F1 | Incomplete F1 | Backchannel F1 | Wait F1 | Macro-F1 | Incomplete → Complete | Backchannel → Complete |
|---:|---:|---:|---:|---:|---:|---:|
| 84.28% | 55.60% | 81.65% | 96.10% | 79.41% | 25.75% | 2.82% |

Easy Turn inference p50/p90/p95 was 416.5/818.1/1027.5 ms with RTF 0.244. Treat it as a research reference, not a clean independent production claim: Easy Turn and FastTurn come from the same research team, train/evaluation overlap has not been fully audited, the Wait class is synthetic, sample rate is a confound, and dataset licensing needs reconciliation.

### 9.4 SID-Bench

| Language | Model | Interruption recall | FIR | Strict FIR | IRL p50 | IRL p90 | APT | Non-break trigger rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| EN | DualTurn fallback | 69.00% | 32.25% | 32.31% | 178 ms | 611 ms | 1.494 s | 41.4% |
| EN | VAP fallback | 44.00% | 67.25% | 67.31% | 62 ms | 150 ms | 2.521 s | 92.4% |
| ZH | DualTurn fallback | 81.45% | 14.44% | 14.44% | 260 ms | 876 ms | 0.839 s | 27.8% |
| ZH | VAP fallback | 47.18% | 64.50% | 64.62% | 88 ms | 162 ms | 2.416 s | 90.6% |

| EN | Kyutai native single | 37.27% | 74.38% | 74.38% | 0 ms | — | 2.698 s | 100.0% |
| EN | VapGPT OTO fallback | 31.45% | 78.38% | 78.38% | 0 ms | 57 ms | 2.843 s | 100.0% |

Noise trigger rates were 4.2% for DualTurn, 91.6% for VAP, 100.0% for VapGPT OTO, and 100.0% for Kyutai.

**SID-Bench can currently evaluate exactly one of the five candidate
architectures.** A usable signal must satisfy two conditions: it must be
independent of the partner channel, which SID-Bench does not provide, and it
must localize interruption onset in time. Floor-state signals fail the first;
VapGPT saturates above 0.958 on all 1,600 English clips. Kyutai passes the first
condition, verified bit-for-bit, and fails the second: `1 - P(turn ending)` is
true for nearly all of an utterance, its per-clip maximum is exactly 1.0000 in
every category, and all 1,600 clips commit at the first frame. Only DualTurn's
0-240 ms forward horizon satisfies both, and 58.6% of English non-break clips
never trigger it. Its lead is therefore close to uncontested rather than close
to fair, and the VAP rows are degenerate artifacts. Mimi and ESPnet are blocked
by the same mechanism as VapGPT. See
[`docs/BENCHMARK_COVERAGE_AUDIT.md`](docs/BENCHMARK_COVERAGE_AUDIT.md).

Paired comparisons:

- EN DualTurn versus VAP: recall +25.00 pp, CI [22.27, 27.64]; strict FIR -35.00 pp, CI [-37.44, -32.62]; APT about 1.03 seconds lower.
- ZH: recall +34.27 pp, CI [30.73, 37.91]; strict FIR -50.19 pp, CI [-52.62, -47.69]; APT about 1.58 seconds lower.
- Middle-position early triggers: EN 49.83% DualTurn versus 96.67% VAP; ZH 15.17% versus 90.33%.
- On ZH beginning interruptions, DualTurn recall was 87.0% versus VAP 92.2%, a -5.2 pp difference, CI [-9.0, -1.4]. DualTurn is more conservative and misses some immediate barge-ins.

These are derived policies and single-channel fallbacks. SID-Bench does not provide the agent channel or an independent calibration split. The current numbers do not establish a production-ready native interruption detector.

### 9.5 Runtime

| Model/device | Init/first-call notes | p50 | p90 | p95 | RTF | RAM | VRAM |
|---|---|---:|---:|---:|---:|---:|---:|
| DualTurn CPU, 4 threads | constructed 2.57 s; first 2.62 s; cold call 53.4 ms | 66.8 ms | 85.6 ms | 88.2 ms | 0.867 | 487 MB | — |
| DualTurn CUDA, A5000 | constructed 5.41 s; first 6.04 s; cold call 628 ms | 13.5 ms | 18.5 ms | 20.0 ms | 0.177 | 1177 MB | 742 MB |
| VAP CPU | — | 22.1 ms | 29.3 ms | — | 0.236 | 653 MB | — |
| VAP CUDA, A5000 | no useful speedup | 21.9 ms | 28.3 ms | — | 0.232 | 1249 MB | 322 MB |

On the tested older CPU, DualTurn misses an 80 ms p90 frame budget, so GPU execution is preferred if strict frame headroom is required. VAP should run on CPU on this hardware because CUDA adds memory cost without meaningful latency gain.

## 10. Current research conclusions

- **Production EOT candidate:** DualTurn, with SmartTurn retained as a very small CPU-friendly reference.
- **Chinese clip-classification candidate:** TurnSense INT8, but **for isolated clips only**. Its FastTurn lead does not transfer to streaming: on eot-bench ZH it is 49.18% FC@300 against DualTurn's 23.36%, a paired +25.82 pp with CI [+20.31, +31.42], and it fails hardest on ordinary 300-500 ms pauses. An abstention and short-utterance policy is still needed for the clip role.
- **Four-state reference:** Easy Turn, with the independence and licensing caveats above.
- **Coverage caveat:** SID-Bench can evaluate exactly one of the five candidate interruption architectures. Every other evaluated signal commits at the first frame of every clip, either because it is defined relative to an agent channel the benchmark does not provide, or because it does not localize onset. DualTurn's lead is uncontested rather than earned, and both Realtime-VAP rows are degenerate artifacts. See [`docs/BENCHMARK_COVERAGE_AUDIT.md`](docs/BENCHMARK_COVERAGE_AUDIT.md).
- **Interruption interim baseline:** DualTurn-derived policy. It is far safer than the current VAP heuristic on middle speech and noise, but its non-break trigger rate and immediate-barge-in trade-off are still not production quality.
- **VAP role:** useful research architecture and TurnBench reference, but the evaluated checkpoint/policy is not a production recommendation. Preserve its non-commercial or share-alike license constraints when applicable.
- **Training decision:** existing models are reasonably promising for EOT, but long HOLD pauses and semantic interruption/backchannel rejection remain weak. If training begins later, prioritize Chinese dual-channel/future-activity or native semantic interruption architectures rather than a broad retraining effort inside this repo.

A promising interruption design is a two-stage policy:

1. fast acoustic acceptance for obvious immediate barge-in;
2. conservative semantic confirmation for speech occurring in the middle of an agent turn.

This is only a hypothesis. It must be calibrated on a separate development set and evaluated once on a held-out test set. Do not tune it directly on SID-Bench test examples.

## 11. Known issues and traps

1. **Raw results are not in Git.** The implementation is committed on `main`, but the raw run directories under `results/` are ignored and exist only on this machine. Preserve them before moving or re-cloning.
2. **Existing DualTurn eot artifacts contain a device metadata inconsistency.** Their manifests record `execution.provider=cuda`, but their `runtime.json` may say `device=cpu` because of an old default. The runner was fixed in `environments/dualturn_endpointing/run_eotbench.py` to write `"device": args.provider`. Immutable old artifacts were intentionally not edited. Use the isolated runtime report for device claims.
3. **DualTurn ZH is zero-shot and unverified.** Do not describe it as officially supported Chinese performance.
4. **Timeout policy can confound eot-bench comparisons.** Quote action-delay and timeout settings with results.
5. **TurnBench is development-only here.** Test labels are withheld; do not call the current results test leaderboard results.
6. **SID outputs are derived.** Do not call them native interruption probabilities.
7. **FastTurn Wait is synthetic.** Keep real/synthetic provenance visible in per-class interpretation.
8. **TurnSense abstention handling is unresolved.** Do not silently coerce abstentions into a class.
9. **Easy Turn independence and license need audit.** Do not overstate generalization.
10. **X2 smoke output is partial.** It is excluded from the comparison manifest and must remain so.
11. **No job is currently running.** A successor should not wait for a background process.

## 12. What remains to do

Priority order for the next session:

### P0 — Preserve and release the current baseline

Done on 2026-09-06:

1. This handoff and `docs/BASELINE_ANALYSIS.md` were cross-checked against the canonical comparison report; all shared numbers agree. The two missing ZH strict-FIR cells in section 9.4 were filled in from the report.
2. All 65 input hashes recorded in `results/comparison/20260906T073000Z/manifest.json` were re-verified against the artifacts on disk: 65 match, 0 missing, 0 mismatched.
3. `ruff format` had drifted on 11 files, which would have failed CI on the first push. It was applied; `ruff check`, `ruff format --check`, `pytest`, and `turn-eval --help` all pass. Note that 128 tests pass only with the `smart-turn` extra installed; a plain `uv sync --frozen` (what CI does) gives 127 passed and 1 skipped.
4. The README now links the canonical comparison artifact and states which parts of `results/` are tracked.
5. The implementation was committed to `main`, together with the curated comparison, analysis, and runtime artifacts (about 992 KB). Raw run directories remain ignored.

Still open:

- Nothing has been pushed to `origin`. Decide whether this work should be published, and whether the raw `results/` tree needs external storage or an archive manifest.

### P1 — Build a valid calibration path for production policies

Done on 2026-09-06:

1. `src/turn_eval/calibration.py` supplies seeded, stratified, grouped calibration splits and frozen policy artifacts. `require_heldout_evaluation` fails closed on a policy frozen on a different split, a policy selected on the held-out side, and an evaluation set that reaches into calibration ids.
2. `turn-eval calibrate-sidbench` fits the two-stage interruption policy by replaying a completed run's cached traces. At the incumbent parameters the replay reproduces that run's published metrics exactly, which is what licenses the comparison.
3. Both languages independently select 8 frames (640 ms) of confirmation. Held-out ZH: recall 82.60% to 86.23%, non-break trigger 26.57% to 9.43%. Held-out EN: recall 67.53% to 82.99%, non-break trigger 41.14% to 6.00%.
4. The gain is a trade, and the aggregate APT hides it: response latency p50 roughly triples, and immediate-barge-in recall falls 3 to 8 points while mid-turn recall rises 10 to 35. See [`docs/BASELINE_ANALYSIS.md`](docs/BASELINE_ANALYSIS.md).

Blocked, with the reason recorded:

- **The long-HOLD guard cannot be calibrated on eot-bench.** ZH holds 45 spans of 1 s or more across 38 conversations and exactly one span at 2-5 s; a 30% split leaves about 13 spans to fit on and 32 to test on. Closing this needs more long-pause data, not more models. See [`docs/BENCHMARK_COVERAGE_AUDIT.md`](docs/BENCHMARK_COVERAGE_AUDIT.md).

Still open:

1. Decide whether the two-stage latency cost is acceptable for the target product, or fit a variant that keeps a fast path for immediate barge-in.
2. Report end-to-end decision latency including VAD and policy delay, not only the model's response latency.
3. Re-run the frozen policy on the noise split, which the current artifact does not cover.

Do not use the SID test set or FastTurn test set to choose final production thresholds.

### P1 — Resolve evidence gaps

0. Close the prioritized coverage gaps in [`docs/BENCHMARK_COVERAGE_AUDIT.md`](docs/BENCHMARK_COVERAGE_AUDIT.md). The audit classifies all 88 model/benchmark cells; 24 are unrun with no recorded reason. G1, G3 and G4 are closed; G8 was withdrawn. G2 is closed too, with a negative result that settles Chinese model selection: DualTurn stands, TurnSense does not transfer to streaming. No high-value coverage gap remains. Interruption coverage is now limited by SID-Bench itself rather than by missing runners: it needs a dual-channel Chinese interruption set, or a fallback protocol supplying a realistic agent channel instead of digital silence.
1. Investigate the small local-versus-model-card DualTurn eot-bench discrepancy.
2. Audit Easy Turn training data, FastTurn overlap, and dataset/model licenses.
3. Define and evaluate TurnSense abstention behavior on independent calibration data.
4. Add paired uncertainty estimates to any new headline comparisons.

### P2 — Optional benchmark/model extensions

- Run the separate Easy Turn Testset only as a secondary/legacy reproduction.
- Add WavLM Causal/Anchor only if complete open-weight architecture coverage is required and license status is acceptable.
- Add ESPnet per-channel as an ablation if needed.
- Add native dual-channel Chinese data/evaluation when available.
- Add LiveKit cloud/local models only if the scope expands beyond open-weight baselines.
- Add whole-agent HumDial-FDBench or Full-Duplex-Bench under a separate `agent_bench` area, not the component leaderboard.
- Keep X2-Turn deferred unless the user explicitly requests it and suitable hardware/storage are available.

### Future training, in a separate repository

If the fixed evaluation shows EOT is insufficient, consider SmartTurn-ZH or a DualTurn-style Chinese endpointing model. If interruption and backchannel rejection remain insufficient, prioritize VAP-ZH, DualTurn-ZH, causal WavLM, Mimi/NAC-LD, or a native semantic interruption head. Use this repository without changing test protocols to measure improvement.

## 13. Milestone status relative to the original plan

| Milestone | Status | Notes |
|---|---|---|
| M0 — Skeleton | Complete | CLI, schemas, adapters, manifests, tests, and CI exist. |
| M1 — eot-bench | Complete | EN/ZH, SmartTurn, VAP, DualTurn, Pareto summaries, and error analysis. |
| M2 — TurnBench English | Complete for selected open-weight architectures | Official dev validation and several exact/reference adapters. Not every public baseline was run. |
| M3 — Chinese turn state | Substantially complete | FastTurn two-state and four-state results with TurnSense, SmartTurn, and Easy Turn. X2 deferred. Separate Easy Turn Testset remains optional. |
| M4 — Interruption | Complete as a derived baseline | SID EN/ZH/noise for DualTurn and VAP, plus paired analysis. Native semantic interruption remains future work. |
| M5 — Chinese baseline report | Complete under narrowed open-weight scope | Canonical comparison and research analysis exist. This is not yet a production-policy sign-off. |

## 14. Pinned revisions and external references

Use pinned revisions whenever reproducing results.

### Benchmarks

- TurnBench scorer/source: `e02d547803fd0e45221afcaeb1cd24b61ae7568e` — [GitHub](https://github.com/SesameAILabs/turnbench/tree/e02d547803fd0e45221afcaeb1cd24b61ae7568e), [leaderboard](https://turnbench.sesame.com/), [paper](https://arxiv.org/abs/2608.25218), [development data](https://huggingface.co/datasets/mundo-ai/turn-benchmark-dev)
- TurnBench development dataset revision: `8fa18a24be51528a45397b35cbcaecd84202062b`
- LiveKit eot-bench scorer: `6594d8b3b8af385b15f116dde310ce45af92d646` — [GitHub](https://github.com/livekit/eot-bench)
- LiveKit eot-bench data: `ca9d98a9686b920a2d8c9eb984224ba9be74e4dd` — [Hugging Face](https://huggingface.co/datasets/livekit/eot-bench-data)
- FastTurn source: `0c1c03dc03c9cc0bf7a743a066fc99107c8a5c59` — [GitHub](https://github.com/ASLP-lab/FastTurn/tree/0c1c03dc03c9cc0bf7a743a066fc99107c8a5c59), [paper](https://arxiv.org/abs/2604.01897v6)
- FastTurn Testset: `4194458681149e2947cdb062ad61292cf9705a59` — [Hugging Face](https://huggingface.co/datasets/ASLP-lab/FastTurn-Testset)
- SID-Bench dataset: `6eb13b573ad588646ce0de513d4255e55caf858b` — [Hugging Face](https://huggingface.co/datasets/kxxia/SID-bench)
- SID-Bench scorer: `0a9e5001e1fe534b3eec4f76c4455ae4bff9ccf1`

### Models and implementations

- SmartTurn v3 source/model revision: `f766f81d3cfdf7737ac64aad813d91bbfd56bf93` — [model](https://huggingface.co/pipecat-ai/smart-turn-v3), [source](https://github.com/pipecat-ai/smart-turn)
- Voice Activity Projection source: `f39a78b23a6dccdbedd106e00b48c410b8739f5d` — [GitHub](https://github.com/ErikEkstedt/VoiceActivityProjection)
- VAP multilingual checkpoints: `b9aa0ba1718221153e04ef343c7f3b8cf84bfbc3` — [Hugging Face](https://huggingface.co/viks66/VAP_checkpoints)
- DualTurn Endpointing: `c3860ed71210fe0144af35af340fd7a4dec3d2d3` — [Hugging Face](https://huggingface.co/anyreach-ai/dualturn-endpointing)
- ESPnet TurnTaking model: `766378e6119f254270f0dd4a4be13480ec6b78fc` — [Hugging Face](https://huggingface.co/espnet/Turn_taking_prediction_SWBD)
- ESPnet source: `750e3749fc37a09187fe0fc6fb278ccb007181e8`
- ESPnet Whisper dependency: `8a65df1760894b91857c9904410cca1123fa1bb0`
- Mimi Endpointer: `f7459f95f43766a52c612ee1bc72b2444e86fc6b` — [Hugging Face](https://huggingface.co/viks66/mimi-endpointer)
- Kyutai Moshiko source/model dependency: `2bfc9ae6e89079a5cc7ed2a68436010d91a3d289`
- Kyutai STT 1B: `095e38f6242006a93c2541149b181988397f5c7c` — [Hugging Face](https://huggingface.co/kyutai/stt-1b-en_fr-candle)
- Easy Turn model: `7d1f8b06776b575ac22e2fd4aa7913da2f4b8b98` — [Hugging Face](https://huggingface.co/ASLP-lab/Easy-Turn)
- Easy Turn source: `fe175eaa8442c9a0834acf29f8640ee74c2aceca` — [GitHub](https://github.com/ASLP-lab/Easy-Turn)
- Easy Turn Qwen dependency: `7ae557604adf67be50417f59c2c2f167def9a775`
- TurnSense model: `92924ae57f61b59cc6195d1367f8c705b2ba06b4` — [Hugging Face](https://huggingface.co/Baiji-Team/TurnSense)
- TurnSense source revision: `e1ae083a361182614f15eff5c293dadb07314ec4`
- Optional WavLM turn-taking checkpoint: [Hugging Face](https://huggingface.co/ZhuoyanTao/causal-wavlm-turn-taking)

### Deferred X2-Turn

- Source: `3609a0fd11b4a1dc2b2629e95538bbe0860da96d` — [GitHub](https://github.com/X-Square-Robot/X2-Turn/tree/3609a0fd11b4a1dc2b2629e95538bbe0860da96d)
- Model: `d188a8b86058ece5e293124a278a1a6256af81fa` — [Hugging Face](https://huggingface.co/x-square-robot/X2-Turn-4B-0812/tree/d188a8b86058ece5e293124a278a1a6256af81fa)

## 15. Dataset/cache locations

- Hugging Face cache: `/mnt/md0/user_wayne/.hf_cache`
- FastTurn extracted data: `$HF_HOME/turn-eval-datasets/fastturn/<revision>`
- TurnBench is gated. Access was already obtained for this workspace, but a new machine/session may still need an authenticated Hugging Face token.
- Datasets are not committed to this repository. Use the download/preparation commands documented in the README and preserve revision/checksum metadata.

## 16. Recommended first action for the next session

Start with:

```bash
cd /mnt/md0/user_wayne/turn-detection-benchmark
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
git status --short
uv run pytest -q
sed -n '1,240p' HANDOFF.md
sed -n '1,260p' docs/BASELINE_ANALYSIS.md
```

Then confirm the next goal with the user. The most useful next research step is not another expensive baseline run; it is establishing an independent calibration dataset and evaluating a HOLD-safe EOT policy plus a two-stage interruption policy without test-set tuning.
