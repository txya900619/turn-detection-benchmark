# Cross-Benchmark Coverage Audit

Audited: 2026-09-06 UTC

## Purpose

[`TURNBENCH_BASELINE_AUDIT.md`](TURNBENCH_BASELINE_AUDIT.md) audits this
repository against TurnBench's official roster. No equivalent audit existed for
LiveKit eot-bench, FastTurn, or SID-Bench, because those benchmarks publish no
roster to compare against. As a result their coverage gaps were never stated,
and an empty cell could not be distinguished from a deliberate exclusion.

This document classifies every model/benchmark cell so that a reader can tell
which absences are protocol limits, which are recorded decisions, and which are
simply unrun work.

## Verdict

Across 11 models and 8 benchmark splits there are 88 cells. The canonical
comparison fills **23**. Of the 65 empty cells, **28 are blocked** by a
capability or protocol limit, **13 are recorded decisions**, and **24 are gaps
with no recorded reason**.

Seven of the ten evaluated models appear in exactly one benchmark. Only DualTurn
and Realtime-VAP have cross-benchmark coverage, and both of the interruption
rows in the canonical report belong to those two models. Conclusions that rank
models across benchmarks therefore rest on a much narrower sample than the
39-row report suggests.

## Method

A cell is `(model, benchmark-split)`. TurnBench's EOT and interruption tasks are
collapsed into one cell because they are produced by a single run.

Classification is decided by comparing what a benchmark requires against what a
model declares in its `ModelDescriptor`
([`src/turn_eval/core/types.py:276`](../src/turn_eval/core/types.py)) and in the
`channel_mode` / `execution_mode` / `signal_provenance` fields recorded in
[`results/comparison/20260906T073000Z/summary.csv`](../results/comparison/20260906T073000Z/summary.csv).

| Code | Meaning |
| --- | --- |
| `DONE` | A complete, leaderboard-eligible run is selected in the canonical comparison manifest. |
| `BLOCKED` | The model cannot produce what the benchmark requires without inventing a mapping. Filling the cell would violate the honest-labelling principle in HANDOFF section 4.5, or would produce a degenerate metric. These cells should stay empty. |
| `DECIDED` | Absence is a recorded decision with a citable reason. |
| `GAP` | Technically feasible with an existing precedent in this repository, and no recorded reason for the absence. |

### What each benchmark requires

| Benchmark | Input | Required model signal | Threshold source |
| --- | --- | --- | --- |
| LiveKit eot-bench EN/ZH | Mono user-audio prefix at prescribed decision points | A **continuous** `p_eot` in `[0, 1]`. `eot_bench_policy_grid` sweeps 101 thresholds and `_evaluate_policy` filters on `row.p_eot > threshold` ([`src/turn_eval/metrics/eot.py:64`](../src/turn_eval/metrics/eot.py)); a hard 0/1 label collapses the sweep to a single point and yields no Pareto curve. | Prescribed official sweep |
| TurnBench EN dev | Dual-channel conversation timeline | Continuous per-task scores for EOT **and** interruption | Selected on dev at a 10% FPR budget |
| FastTurn ZH, 2-state | Mono isolated clip | A discrete `complete` / `incomplete` decision | Predeclared fixed, no calibration split exists |
| FastTurn ZH, 4-state | Mono isolated clip | Native `complete` / `incomplete` / `backchannel` / `wait` | None |
| SID-Bench EN/ZH/noise | Mono user-only clip with a timeline | A **time-of-trigger** decision during agent speech | Frozen externally before scoring |

### What each model declares

| Model | `output_signals` | `execution_modes` | Supported languages | License |
| --- | --- | --- | --- | --- |
| DualTurn Endpointing | `eot` | `streaming` | `en` | Apache-2.0 |
| Realtime-VAP trilingual | `eot` | `streaming` | `en`, `zh`, `ja` | CC-BY-NC-SA-4.0 |
| SmartTurn v3.2 | `eot` | `pointwise` | `en`, `zh` | BSD-2-Clause |
| SmartTurn v3.1 | `eot` | `pointwise` | `en`, `zh` | BSD-2-Clause |
| VapGPT OTO | floor state, native interruption | dual-channel bounded windows | `en` | CC-BY-4.0 |
| ESPnet TurnTaking | native turn-change classes | mixed-mono causal | `en` | CC-BY-4.0 |
| Mimi Endpointer | floor state | dual-channel causal | `en` | CC-BY-4.0 |
| Kyutai Semantic VAD | native VAD head | dual-channel streaming | `en`, `fr` | CC-BY-4.0 |
| TurnSense INT8 | `turn_state` | `offline` | `zh` | Apache-2.0 |
| Easy Turn | `turn_state` | `offline` | `zh` | Apache-2.0 |
| X2-Turn 4B | `turn_state` | `causal_offline` | `en`, `zh` | Apache-2.0 |

Note the discriminator: models declaring `output_signals={"eot"}` produce a
continuous endpointing probability, while models declaring
`output_signals={"turn_state"}` produce a class decision. That single field
explains most of the `BLOCKED` cells in both directions.

## Coverage matrix

| Model | eot-EN | eot-ZH | TurnBench-EN | FastTurn-2 | FastTurn-4 | SID-EN | SID-ZH | SID-noise |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| DualTurn Endpointing | DONE | DONE | DONE | GAP | BLOCKED | DONE | DONE | DONE |
| Realtime-VAP | DONE | DONE | DONE | GAP | BLOCKED | DONE | DONE | DONE |
| SmartTurn v3.2 | DONE | DONE | DONE | DONE | BLOCKED | GAP | GAP | GAP |
| SmartTurn v3.1 | DECIDED | DECIDED | DONE | DECIDED | BLOCKED | DECIDED | DECIDED | DECIDED |
| VapGPT OTO | GAP | GAP | DONE | BLOCKED | BLOCKED | **GAP** | DECIDED | GAP |
| ESPnet TurnTaking | GAP | GAP | DONE | BLOCKED | BLOCKED | GAP | DECIDED | GAP |
| Mimi Endpointer | GAP | GAP | DONE | BLOCKED | BLOCKED | GAP | DECIDED | GAP |
| Kyutai Semantic VAD | GAP | GAP | DONE | BLOCKED | BLOCKED | **GAP** | DECIDED | GAP |
| TurnSense INT8 | **GAP** | **GAP** | DECIDED | DONE | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| Easy Turn | BLOCKED | BLOCKED | BLOCKED | **GAP** | DONE | BLOCKED | BLOCKED | BLOCKED |
| X2-Turn 4B | BLOCKED | BLOCKED | BLOCKED | DECIDED | DECIDED | BLOCKED | BLOCKED | BLOCKED |

Bold cells are the prioritized gaps listed below.

## Blocked cells, with reasons

These should remain empty. Filling them would require inventing a mapping and
would breach HANDOFF section 4.5.

| Cells | Reason |
| --- | --- |
| Easy Turn and X2-Turn on eot-bench | Both emit a decoded terminal label, not a calibrated probability. HANDOFF section 9.3 records that Easy Turn's "decoded hard labels are not presented as calibrated probabilities". The prescribed 101-threshold sweep degenerates to one point, so no false-cutoff/latency trade-off curve exists. |
| Easy Turn, TurnSense, X2-Turn on SID-Bench | SID requires a time-of-trigger decision during agent speech. All three are declared `offline` or `causal_offline` clip classifiers with no timeline output. Running them would mean designing a new sliding-window streaming protocol, not writing a runner. |
| Easy Turn, X2-Turn on TurnBench | Same continuous-score requirement as eot-bench, plus no dual-channel input path. |
| DualTurn, Realtime-VAP, VapGPT, ESPnet, Mimi, Kyutai on FastTurn 4-state | None has a native Backchannel or Wait class. All declare `output_signals={"eot"}` or a floor-state head. |
| VapGPT, ESPnet, Mimi, Kyutai on FastTurn 2-state | These are dual-channel conversation models. FastTurn releases isolated mono clips with no other-speaker channel, no conversation ID, and no timeline (`FASTTURN_PROTOCOL.md` section 1), so their defining input is absent. |
| SmartTurn v3.1/v3.2 and TurnSense on FastTurn 4-state | SmartTurn is binary EOT only. TurnSense's native labels are `complete` / `incomplete` / `invalid` with no Backchannel or Wait, as recorded in `STATUS.md:505`. |

## Recorded decisions

| Cells | Reason |
| --- | --- |
| X2-Turn, FastTurn 2-state and 4-state | Deferred. The release stores about 5.235B parameters in a 10,470,070,544-byte BF16 file and the local path is two-pass. See HANDOFF section 6 and [`X2_TURN_PROTOCOL.md`](X2_TURN_PROTOCOL.md). A verified adapter and a partial smoke artifact are retained. |
| SmartTurn v3.1, everything except TurnBench | v3.1 is carried only as the official TurnBench reference version; v3.2 is the production candidate. The role split is stated in the README's TurnBench section. |
| VapGPT, ESPnet, Mimi, Kyutai on SID-ZH | All four are English-only checkpoints (Kyutai is EN/FR). A Chinese row would be unverified zero-shot, which HANDOFF section 4.5 requires to be labelled as such. Deprioritized below their SID-EN cells rather than blocked. |
| TurnSense on TurnBench | TurnSense declares `supported_languages={"zh"}` and TurnBench dev is English, so the row would be unverified zero-shot. Deprioritized below its eot-bench cells. |

## Gaps, prioritized

Every gap below has an existing precedent in this repository, so the work is a
new runner or a re-scoring pass, not new protocol design. The seven groups below
account for all 24 gap cells.

Each SID-Bench run also produces the shared noise split under the same frozen
policy, so a SID gap group always covers its model's `SID-EN` and `SID-noise`
cells together.

| Group | Cells | Cost |
| --- | --- | --- |
| G1 | VapGPT OTO on SID-EN, SID-noise | GPU inference |
| G2 | TurnSense on eot-ZH, eot-EN | New adapter, then CPU inference |
| G3 | Kyutai on SID-EN, SID-noise | GPU inference |
| G4 | Easy Turn on FastTurn 2-state | Re-scoring only, no inference |
| G5 | DualTurn and Realtime-VAP on FastTurn 2-state | Blocked on P1 calibration |
| G6 | VapGPT, ESPnet, Mimi, Kyutai on eot-EN and eot-ZH | 8 cells, low value |
| G7 | SmartTurn v3.2 on SID-EN, SID-ZH, SID-noise | Cheap, CPU |
| G8 | ESPnet and Mimi on SID-EN, SID-noise | GPU inference |

### G1. VapGPT OTO on SID-Bench EN

The strongest reason to run this is signal provenance, not headline recall.
On TurnBench dev, VapGPT OTO reaches 95.68% interruption recall at 9.99% FPR
from a **native floor-state** head, whereas DualTurn's comparable 96.25% is
`derived-fvad`. SID-Bench is the dedicated interruption benchmark, and both of
its current rows are derived single-channel fallbacks. A native-signal row would
test whether the SID conclusion survives a change of signal provenance.
The checkpoint is CC-BY-4.0, so unlike Realtime-VAP it is not license-blocked.

Precedent: `run-sidbench-vap` and `run-sidbench-dualturn` already implement the
`user=CH0, silence=CH1` fallback. Requires a frozen threshold from TurnBench EN
dev, matching the existing procedure.

### G2. TurnSense on eot-bench ZH and EN

TurnSense is the only natively Chinese-trained candidate in the roster, and it
has the best Chinese two-state numbers by a wide margin (+12.59 pp paired
accuracy over SmartTurn). It has never been measured in a streaming setting, so
it cannot be placed on the same axes as DualTurn, and the Chinese production
recommendation currently rests on a zero-shot English model.

Feasible: TurnSense emits a three-way softmax, so `P(complete)` is continuous
and satisfies the sweep requirement. Only `TurnSenseFastTurnAdapter`
(`ClipClassificationModelAdapter`) exists; a `PointwiseModelAdapter` must be
added, mirroring `SmartTurnV32Adapter`
([`src/turn_eval/models/smart_turn.py:39`](../src/turn_eval/models/smart_turn.py)).

One design decision must be made and recorded first: what the `invalid` class
means at a streaming decision point. Folding it into `P(complete)` dilutes the
score; treating it as "do not decide, keep waiting" is more defensible but is a
policy choice that must be frozen on calibration data, not selected on
eot-bench.

### G3. Kyutai Semantic VAD on SID-Bench EN

93.08% TurnBench interruption recall at 8.52% FPR, CC-BY-4.0, native streaming
at 80 ms. Same precedent and threshold procedure as G1. Lower priority than G1
because its interruption signal is already `derived`, so it does not add the
native-provenance evidence that G1 does.

### G4. Easy Turn restricted to the FastTurn 2-state subset

**Analysis only; requires no new inference.** The completed Easy Turn run covers
all 22,432 clips, which already includes the 18,352 Complete/Incomplete clips
used by the 2-state leaderboard. Restricting the existing predictions to that
subset would place Easy Turn, TurnSense, and SmartTurn on one axis.

A mapping decision must be recorded first: Easy Turn may predict `backchannel`
or `wait` on a 2-state clip. That is a closed-set confusion, not an abstention,
and must be scored as an error rather than dropped.

### G5. DualTurn and Realtime-VAP on the FastTurn 2-state subset

This is the cleanest way to put DualTurn and TurnSense in one coordinate system
on Chinese data. Both models emit a continuous `p_eot` that can be thresholded
into Complete/Incomplete on an isolated clip.

**Blocked on calibration, not on capability.** FastTurn publishes no
train/dev/test partition (`FASTTURN_PROTOCOL.md` section 1), so there is nowhere
legitimate to select the threshold. Any derivative split must be persisted with
its seed and example IDs. This gap closes only after the P1 calibration work.

### G6. ESPnet, Mimi, VapGPT, Kyutai on eot-bench EN/ZH

Lower value. Each would be a single-channel fallback row for an English-only
dual-channel research checkpoint, on a benchmark where the production question
is already answered by DualTurn. ESPnet is the mildest case because it already
consumes a mono mix.

### G7. SmartTurn v3.2 on SID-Bench

Expected to be poor: its derived TurnBench interruption recall is 10.09%. The
value is in converting "expected to be poor" into "measured", since that
expectation is currently recorded nowhere. Cheap on CPU. Unlike the four
English-only checkpoints, SmartTurn declares Chinese support, so its `SID-ZH`
cell is a gap rather than a zero-shot deprioritization.

### G8. ESPnet TurnTaking and Mimi Endpointer on SID-Bench EN

Completes the interruption picture across all TurnBench architectures. Both are
CC-BY-4.0. Lower priority than G1 and G3 on their TurnBench interruption recall
(ESPnet 64.27%, Mimi 87.61%), and Mimi's runner performs full-sequence
inference, so a streaming SID fallback would need its execution mode restated
in metadata.

## Effect on the current conclusions

Two statements in [`BASELINE_ANALYSIS.md`](BASELINE_ANALYSIS.md) should be read
with this coverage in mind.

1. "DualTurn's frozen derived-FVAD policy is the strongest current SID-Bench
   fallback" is true, but the comparison set is two models, both derived and
   both single-channel fallback. It is not evidence that DualTurn beats the
   native-signal interruption models, which have never been run on SID.
2. "Isolated Chinese Complete/Incomplete candidate: TurnSense" and "EOT
   production candidate: DualTurn" are drawn from different benchmarks with
   different evaluation units. The two models have never been compared on any
   shared axis. G2 and G5 exist to close that.

Nothing in this audit contradicts a published number. The canonical comparison
report remains valid for what it measures; this document records what it does
not measure.

## Maintenance

Update this file whenever a model or benchmark is added, and re-derive the
matrix from the canonical comparison manifest rather than from memory. The
matrix above was generated against
`results/comparison/20260906T073000Z/summary.csv`, whose 65 input hashes were
re-verified on 2026-09-06.
