# Cross-Benchmark Coverage Audit

Audited: 2026-09-06 UTC. Last updated after closing gaps G4 and G1.

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
comparison fills **26**. Of the 62 empty cells, **34 are blocked** by a
capability or protocol limit, **11 are recorded decisions**, and **17 are gaps
with no recorded reason**.

Closing G1 produced a structural result that reclassified six further cells and
placed a caveat on two rows that were already published. It is stated in full
under "The single-channel fallback criterion" below.

Seven of the ten evaluated models appear in exactly one benchmark. Only DualTurn
and Realtime-VAP have cross-benchmark coverage, and both of the interruption
rows in the canonical report belong to those two models. Conclusions that rank
models across benchmarks therefore rest on a much narrower sample than the
40-row report suggests.

## Method

A cell is `(model, benchmark-split)`. TurnBench's EOT and interruption tasks are
collapsed into one cell because they are produced by a single run.

Classification is decided by comparing what a benchmark requires against what a
model declares in its `ModelDescriptor`
([`src/turn_eval/core/types.py:276`](../src/turn_eval/core/types.py)) and in the
`channel_mode` / `execution_mode` / `signal_provenance` fields recorded in
[`results/comparison/20260906T111447Z/summary.csv`](../results/comparison/20260906T111447Z/summary.csv).

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
| Realtime-VAP | DONE | DONE | DONE | GAP | BLOCKED | DONE* | DONE* | DONE* |
| SmartTurn v3.2 | DONE | DONE | DONE | DONE | BLOCKED | GAP | GAP | GAP |
| SmartTurn v3.1 | DECIDED | DECIDED | DONE | DECIDED | BLOCKED | DECIDED | DECIDED | DECIDED |
| VapGPT OTO | GAP | GAP | DONE | BLOCKED | BLOCKED | DONE | DECIDED | DONE |
| ESPnet TurnTaking | GAP | GAP | DONE | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| Mimi Endpointer | GAP | GAP | DONE | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| Kyutai Semantic VAD | GAP | GAP | DONE | BLOCKED | BLOCKED | **GAP** | DECIDED | GAP |
| TurnSense INT8 | **GAP** | **GAP** | DECIDED | DONE | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| Easy Turn | BLOCKED | BLOCKED | BLOCKED | DONE | DONE | BLOCKED | BLOCKED | BLOCKED |
| X2-Turn 4B | BLOCKED | BLOCKED | BLOCKED | DECIDED | DECIDED | BLOCKED | BLOCKED | BLOCKED |

Bold cells are the prioritized gaps listed below. `DONE*` marks a completed
row whose signal is degenerate under the criterion in the next section; the
run is valid and the number is real, but it does not measure model quality.

## The single-channel fallback criterion

Running G1 produced a negative result that generalizes, so it is recorded here
rather than only in the gap entry.

SID-Bench releases user-only mono clips and provides no agent channel. Every
dual-channel model must therefore be run in single-channel fallback, with the
user on channel 1 and causal silence on channel 2. That configuration is only
sound for a signal that is an **absolute property of the user's own stream**. A
signal defined **relative to the partner channel** collapses to a constant,
because a silent partner trivially never holds the floor.

VapGPT OTO makes this measurable. Its interruption signal is `p_now[speaker]`,
the native floor state. Comparing the same checkpoint across the two
configurations, on frames rather than on labels:

| Configuration | q10 | Median | Frames at or above the frozen 0.859 threshold |
| --- | ---: | ---: | ---: |
| TurnBench EN dev, native dual channel | 0.007 | 0.500 | 40.12% |
| SID-Bench EN, single-channel fallback | — | ~0.99 | effectively all |

In fallback the signal saturates within the first frames of every clip. The
per-clip maximum is above 0.958 for **all 1,600** English clips, and the
interruption and non-break distributions almost completely overlap:

| Category | n | Min of per-clip max | Median | Max |
| --- | ---: | ---: | ---: | ---: |
| interruption_beginning | 500 | 0.9602 | 0.9987 | 0.9997 |
| interruption_middle | 600 | 0.9806 | 0.9991 | 0.9997 |
| nonbreak | 500 | 0.9583 | 0.9913 | 0.9990 |

No threshold separates them, so this is not a calibration problem that a
different operating point would fix. Re-selecting a threshold on SID would in
any case be test-set tuning, which the protocol forbids.

### Which signals the criterion admits

| Model | Committed SID signal | Relative to partner channel? | Consequence |
| --- | --- | --- | --- |
| DualTurn Endpointing | `fvad[0,0]`, user 0-240 ms future activity | No | Evaluable; the completed row is meaningful |
| SmartTurn v3.2 | `P(complete)` on user audio | No | Evaluable; gap G7 stands |
| Kyutai Semantic VAD | `1 - vad_heads[2]`, P(turn ending) per stream | No | Evaluable; gap G3 stands |
| Realtime-VAP | `p_now[channel_1]` | **Yes**, floor state | Degenerate; see the caveat below |
| VapGPT OTO | `p_now[speaker]` | **Yes**, floor state | Degenerate; measured by G1 |
| Mimi Endpointer | `P(active speaker)` | **Yes**, floor state | Blocked, same mechanism |
| ESPnet TurnTaking | class probability times an `energy_1/(energy_1+energy_2)` onset share | **Yes**, energy-share attribution | Blocked; with channel 2 silent the share is identically 1 |

This is why DualTurn leads SID-Bench while losing to VapGPT's native head on
TurnBench. DualTurn's signal asks "is this user about to speak", which survives
the fallback. The floor-state models ask "which of the two is speaking", which
does not.

### Caveat on two already-published rows

The Realtime-VAP SID-Bench rows use the same floor-state signal and are
therefore degenerate in the same way. Their 92.4% English and 90.6% Chinese
non-break trigger rates are largely an artifact of the fallback, not a measure
of that checkpoint's interruption ability. The runs remain valid and are kept,
but they must not be read as evidence that Realtime-VAP is poor at interruption.
The matrix marks them `DONE*`.

### What this means for the benchmark, not just the models

SID-Bench cannot currently evaluate the native dual-channel interruption
architectures at all. Answering "which architecture is best for Chinese
interruption" therefore needs either a dual-channel Chinese interruption set, or
a fallback protocol that supplies a realistic agent channel instead of digital
silence. Both are larger pieces of work than a missing runner, and both belong
with the P1 calibration effort rather than with coverage filling.

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
| VapGPT, Mimi, ESPnet on SID-Bench EN, ZH, and noise | Their committed interruption signal is defined relative to the partner channel, which SID-Bench does not provide. See "The single-channel fallback criterion". Measured for VapGPT, inferred for Mimi and ESPnet from identical signal definitions. |
| SmartTurn v3.1/v3.2 and TurnSense on FastTurn 4-state | SmartTurn is binary EOT only. TurnSense's native labels are `complete` / `incomplete` / `invalid` with no Backchannel or Wait, as recorded in `STATUS.md:505`. |

## Recorded decisions

| Cells | Reason |
| --- | --- |
| X2-Turn, FastTurn 2-state and 4-state | Deferred. The release stores about 5.235B parameters in a 10,470,070,544-byte BF16 file and the local path is two-pass. See HANDOFF section 6 and [`X2_TURN_PROTOCOL.md`](X2_TURN_PROTOCOL.md). A verified adapter and a partial smoke artifact are retained. |
| SmartTurn v3.1, everything except TurnBench | v3.1 is carried only as the official TurnBench reference version; v3.2 is the production candidate. The role split is stated in the README's TurnBench section. |
| VapGPT and Kyutai on SID-ZH | Both are English-only checkpoints (Kyutai is EN/FR). A Chinese row would be unverified zero-shot, which HANDOFF section 4.5 requires to be labelled as such. Deprioritized below their SID-EN cells rather than blocked. ESPnet and Mimi were previously listed here; they are now blocked outright by the fallback criterion. |
| TurnSense on TurnBench | TurnSense declares `supported_languages={"zh"}` and TurnBench dev is English, so the row would be unverified zero-shot. Deprioritized below its eot-bench cells. |

## Gaps, prioritized

Every gap below has an existing precedent in this repository, so the work is a
new runner or a re-scoring pass, not new protocol design. The seven groups below
account for all 17 remaining gap cells.

Each SID-Bench run also produces the shared noise split under the same frozen
policy, so a SID gap group always covers its model's `SID-EN` and `SID-noise`
cells together.

| Group | Cells | Cost |
| --- | --- | --- |
| G1 | VapGPT OTO on SID-EN, SID-noise | Closed 2026-09-06; negative result |
| G2 | TurnSense on eot-ZH, eot-EN | New adapter, then CPU inference |
| G3 | Kyutai on SID-EN, SID-noise | GPU inference; signal survives the criterion |
| G4 | Easy Turn on FastTurn 2-state | Closed 2026-09-06, re-scoring only |
| G5 | DualTurn and Realtime-VAP on FastTurn 2-state | Blocked on P1 calibration |
| G6 | VapGPT, ESPnet, Mimi, Kyutai on eot-EN and eot-ZH | 8 cells, low value |
| G7 | SmartTurn v3.2 on SID-EN, SID-ZH, SID-noise | Cheap, CPU |
| G8 | ESPnet and Mimi on SID-EN, SID-noise | Withdrawn; blocked by the criterion |

### G1. VapGPT OTO on SID-Bench EN — CLOSED, negative result

Closed on 2026-09-06 with `turn-eval run-sidbench-vap-gpt`, using the
interruption threshold 0.8591171121643947 frozen from
`results/en/turnbench/vap_gpt_oto/20260904T192700Z/scored/scores.json`. No
SID-Bench label was used to select it.

| Split | Model | Recall | FIR | IRL p50 | APT | Non-break FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| EN | DualTurn, derived FVAD | 69.00% | 32.25% | 178 ms | 1.494 s | 41.40% |
| EN | Realtime-VAP, floor state | 44.00% | 67.25% | 62 ms | 2.521 s | 92.40% |
| EN | VapGPT OTO, native floor state | 31.45% | 78.38% | 0 ms | 2.843 s | 100.00% |
| NOISE | VapGPT OTO, native floor state | n/a | 100.00% | n/a | 12.976 s | 100.00% |

By category, the failure is total on exactly the cases that matter: on
`interruption_middle` recall is 0.17% with a 99.83% false-interruption rate,
because the policy commits at the first 20 ms frame, long before the annotated
break. On `interruption_beginning` recall is 69.00%, which is what committing
immediately happens to score when the interruption really is at the start.

The model with the strongest native interruption head on TurnBench, at 95.68%
recall, is thus the worst row on the dedicated interruption benchmark. That is
not a statement about VapGPT. It is a property of the fallback, explained by the
criterion above, and it is the reason this cell and five others are now blocked
rather than open.

Two implementation notes for anyone reproducing this:

- `VapGPT.probs` computes `p_now` and then unconditionally computes a training
  loss whose label extractor needs about 101 future frames. Its `vad` argument
  cannot switch that off, because the body reassigns `vad` before the branch
  that tests it. 746 of the 1,600 English SID clips are shorter than the
  resulting 2.02-second floor. The runner therefore recomputes `p_now` from the
  forward pass using upstream's own objective, rather than padding the audio and
  changing the model input. `environments/vap_turnbench/verify_p_now_equivalence.py`
  asserts the two paths agree; they are bit-for-bit identical, with a maximum
  absolute difference of 0.
- Runtime is not the constraint: RTF is 0.015, about 67 times real time.

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

### G3. Kyutai Semantic VAD on SID-Bench EN — now the highest-value open gap

Promoted on 2026-09-06. After G1, Kyutai is the **only** remaining dual-channel
TurnBench architecture whose committed SID signal survives the fallback
criterion: `1 - vad_heads[2]` is a per-stream turn-ending probability, not a
comparison against the partner channel. It scores 93.08% TurnBench interruption
recall at 8.52% FPR, is CC-BY-4.0, and runs natively streaming at 80 ms.

It is now the only way left to test whether DualTurn's SID-Bench lead reflects
its architecture or merely the fact that its signal is one of the few that the
fallback does not destroy. Use the same frozen-threshold procedure as G1, and
check the committed signal's saturation before trusting the row.

### G4. Easy Turn restricted to the FastTurn 2-state subset — CLOSED

Closed on 2026-09-06 by `turn-eval rescore-fastturn-eot`, which restricts a
completed multi-state run to the native Complete/Incomplete subset without
rerunning inference. The derived run pins the source run's manifest and
prediction hashes and records `inference_rerun: false`.

Easy Turn predicted `backchannel` or `wait` on **1,135** of the 18,352 two-state
clips. Following the source run's own declared `out_of_suite_labels` policy,
those become abstentions that stay in the denominator as closed-set errors; the
raw four-way scores are preserved in the derived artifact. Total abstentions are
1,141, or 6.22%, against TurnSense's 2.85%.

Result, on the identical 18,352 clips:

| Model | Complete F1 | Incomplete F1 | Macro-F1 | Inc→Com | Abstain | p50 | Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TurnSense INT8 | 85.51% | 59.48% | **72.50%** | **23.77%** | 2.85% | 36 ms | 52 MB |
| Easy Turn | 84.67% | 56.34% | 70.50% | 25.75% | 6.22% | 455 ms | 3.45 GB |
| SmartTurn v3.2 | 76.48% | 34.14% | 55.31% | 54.74% | 0.00% | 165 ms | 8.7 MB |

TurnSense is ahead of Easy Turn on every two-state metric while being about 66
times smaller and an order of magnitude faster. This is the first comparison of
the two on a shared axis, and it reverses the impression left by placing Easy
Turn's four-state Macro-F1 of 79.41% next to TurnSense's two-state 72.50%: those
two numbers were never comparable, because they score different clip sets under
different label vocabularies.

Runtime caveat: the derived run recomputes latency percentiles on the subset,
which raises Easy Turn's p50 from 417 ms to 455 ms because the two-state clips
are longer on average than the backchannel clips excluded from the subset. The
wall-clock and audio totals, and therefore RTF, are inherited from the source
run and are not subset measurements; the manifest records this.

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

### G8. ESPnet TurnTaking and Mimi Endpointer on SID-Bench EN — WITHDRAWN

Withdrawn on 2026-09-06. Mimi commits `P(active speaker)` and ESPnet weights its
class probability by an `energy_1/(energy_1+energy_2)` onset share. Both are
defined relative to the partner channel, so both degenerate exactly as VapGPT
did. These cells are now blocked, not open.

## Effect on the current conclusions

1. "DualTurn's frozen derived-FVAD policy is the strongest current SID-Bench
   fallback" survives G1, but the reason has changed. DualTurn does not lead
   because it discriminates interruptions better than the native architectures;
   it leads because its signal is one of the few that the single-channel
   fallback does not destroy. Four of the seven candidate signals cannot be
   measured on SID-Bench at all. Until G3 runs, the ranking should be read as
   "best among the signals this benchmark can measure", not "best available".
2. The Realtime-VAP SID rows are degenerate under the same criterion. Their
   non-break trigger rates are an artifact of the fallback and were previously
   presented as a model-behaviour contrast with DualTurn.
3. "Isolated Chinese Complete/Incomplete candidate: TurnSense" and "EOT
   production candidate: DualTurn" are drawn from different benchmarks with
   different evaluation units. G4 closed the TurnSense-versus-Easy-Turn half of
   this; TurnSense and DualTurn still share no axis, which is what G2 and G5
   exist to fix.

Nothing in this audit contradicts a published number. Every run remains valid
for what it measures; this document records what those measurements do and do
not support.

## Maintenance

Update this file whenever a model or benchmark is added, and re-derive the
matrix from the canonical comparison manifest rather than from memory. The
matrix above was generated against
`results/comparison/20260906T111447Z/summary.csv`, which selects 40 rows from 69
hashed input artifacts.
