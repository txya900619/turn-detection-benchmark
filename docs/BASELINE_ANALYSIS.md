# Baseline Analysis

Updated: 2026-09-06

This document interprets the completed open-weight baseline runs. It does not
define a single overall score, and it does not compare native and derived
signals as though they were equivalent.

## Production EOT: LiveKit eot-bench

| Language | Model | Signal / mode | FC @300 ms | FC @600 ms | Latency @5% | Latency @10% |
|---|---|---|---:|---:|---:|---:|
| EN | DualTurn | native / single-channel fallback | 22.13% | 9.65% | 991 ms | 566 ms |
| EN | Realtime-VAP | derived / single-channel fallback | 33.33% | 13.62% | 1016 ms | 702 ms |
| EN | SmartTurn v3.2 CPU | native / single channel | 37.59% | 14.75% | 1044 ms | 755 ms |
| ZH | DualTurn | native / single-channel fallback | 23.36% | 10.20% | 910 ms | 612 ms |
| ZH | Realtime-VAP | derived / single-channel fallback | 31.25% | 9.70% | 879 ms | 597 ms |
| ZH | SmartTurn v3.2 CPU | native / single channel | 41.78% | 12.83% | 928 ms | 667 ms |

DualTurn dominates the English rows at all four prescribed operating points.
On Chinese it has the best cutoff rate at 300 ms, while Realtime-VAP is only
0.50 percentage points better at 600 ms and 15--31 ms faster at the constrained
cutoff operating points. For a production candidate, DualTurn has two material
advantages over this VAP checkpoint: its EOT value is a native output and its
weights are Apache-2.0; the VAP row is derived and CC-BY-NC-SA-4.0.

The DualTurn English result is close to, but not an exact reproduction of, its
model-card row. Local minus published values are +0.33 percentage points at
300 ms, -0.55 points at 600 ms, -124 ms at 5% cutoff, and -44 ms at 10% cutoff.
The 5% latency difference is too large for an exact-parity claim and should be
treated as a scorer/inference-version investigation item.

Chinese versus English is not a controlled language ablation because the two
eot-bench subsets contain different conversations. Still, the zero-shot Chinese
run shows no catastrophic shift: FC @300/600 differs by only +1.23/+0.55 points.
AUC falls from 0.966 to 0.933 and AP from 0.922 to 0.818, which is evidence that
the score distribution and calibration do change across languages.

### Context length: a checkpoint ablation

The pinned `maai-kyoto/vap_tri` revision publishes one checkpoint per (frame
rate, context length) pair. Holding the frame rate at 10 Hz and moving the
context from 5 s to 20 s changes nothing else, which makes it a controlled test
of the hypothesis that the long-HOLD failures below are a context-length
problem.

**The hypothesis is not supported.** False-cutoff rate by HOLD duration:

| Language | HOLD duration | N | VAP 5 s | VAP 20 s | DualTurn |
|---|---|---:|---:|---:|---:|
| EN | 300--<500 ms | 246 | 29.3% | 22.4% | 14.2% |
| EN | 500--<1000 ms | 207 | 43.5% | 44.9% | 33.8% |
| EN | 1--<2 s | 61 | 90.2% | 78.7% | 62.3% |
| EN | 2--5 s | 18 | 100.0% | 100.0% | 72.2% |
| ZH | 300--<500 ms | 212 | 34.9% | 21.2% | 21.2% |
| ZH | 500--<1000 ms | 171 | 46.2% | 40.9% | 35.7% |
| ZH | 1--<2 s | 45 | 80.0% | 88.9% | 77.8% |
| ZH | 2--5 s | 1 | 100.0% | 100.0% | 100.0% |

Four times the context leaves both 2--5 s slices at 100%, and the Chinese
1--2 s slice gets worse rather than better. Those long slices hold 61, 45, 18
and 1 spans, so none of their movements is well supported on its own; the
defensible reading is that context length does not address long hesitation.

The gain is real but sits somewhere else, on ordinary 300--500 ms pauses:
-6.9 points in English and -13.7 points in Chinese. Aggregated over all HOLD
spans, the paired conversation-cluster comparison against the 5 s checkpoint is
-2.98 points in English with a 95% interval of -7.09 to +1.13, which includes
zero, and -5.59 points in Chinese with an interval of -9.53 to -1.84, which does
not.

The Chinese result matters for model selection. Against DualTurn, the 20 s
checkpoint is +2.30 points with an interval of -1.97 to +6.67: on Chinese
eot-bench the two are now statistically indistinguishable at the 300 ms
operating point, where the 5 s checkpoint had been 7.89 points worse. DualTurn
remains clearly ahead in English, at +8.23 points with an interval of +4.23 to
+12.16.

The improvement is not free. At the 5% cutoff operating point mean latency rises
by 168 ms in English and 32 ms in Chinese, and FC@600 is marginally worse in
both. The license also does not change: this is still the CC-BY-NC-SA-4.0
checkpoint, so it is a research reference rather than a production candidate.

### False-cutoff timing slices

At the prescribed 300 ms operating point, a false cutoff can come from the
model crossing its threshold or from the policy timeout. Replaying each frozen
policy over the stored causal traces gives:

| Language | Model | False cutoffs | Model-triggered | Timeout-triggered |
|---|---|---:|---:|---:|
| EN | DualTurn | 156 | 155 | 1 |
| EN | Realtime-VAP | 235 | 200 | 35 |
| EN | SmartTurn | 265 | 220 | 45 |
| ZH | DualTurn | 142 | 136 | 6 |
| ZH | Realtime-VAP | 190 | 189 | 1 |
| ZH | SmartTurn | 254 | 243 | 11 |

DualTurn's English policy uses a 3,000 ms timeout, while VAP and SmartTurn use
1,000 ms. Some of the English advantage therefore belongs to policy selection,
not the neural score alone. This does not explain Chinese: DualTurn uses a
shorter 1,000 ms timeout than VAP's 1,500 ms and still makes 48 fewer cutoffs.

False-cutoff rates rise sharply with mid-turn pause duration:

| Language | HOLD duration | N | DualTurn | VAP | SmartTurn |
|---|---:|---:|---:|---:|---:|
| EN | 200--<300 ms | 173 | 0.0% | 0.0% | 0.0% |
| EN | 300--<500 ms | 246 | 14.2% | 29.3% | 41.9% |
| EN | 500--<1000 ms | 207 | 33.8% | 43.5% | 43.0% |
| EN | 1--<2 s | 61 | 62.3% | 90.2% | 90.2% |
| EN | 2--5 s | 18 | 72.2% | 100.0% | 100.0% |
| ZH | 200--<300 ms | 179 | 0.0% | 0.0% | 0.0% |
| ZH | 300--<500 ms | 212 | 21.2% | 34.9% | 55.2% |
| ZH | 500--<1000 ms | 171 | 35.7% | 46.2% | 57.9% |
| ZH | 1--<2 s | 45 | 77.8% | 80.0% | 82.2% |
| ZH | 2--5 s | 1 | 100.0% | 100.0% | 100.0% |

The current candidates are useful for ordinary 300--1000 ms pauses but none is
safe for long hesitation. This argues for reporting a long-HOLD slice in
addition to the aggregate Pareto points and for retaining a separate policy
guard when the application expects users to pause for more than one second.

An exploratory word-timestamp slice finds only 11 Chinese HOLD spans whose
preceding token has question punctuation. DualTurn cuts off 36.4% versus 72.7%
for VAP and SmartTurn, but this sample is too small for a linguistic claim. The
duration result above has much stronger support and should drive the immediate
deployment decision.

The paired conversation-cluster analysis confirms that DualTurn's FC@300
advantage is not driven by a few conversations. Relative to VAP, its cutoff rate
is 11.21 points lower in English (95% CI -14.71 to -7.57) and 7.89 points lower
in Chinese (-11.60 to -4.35). Relative to SmartTurn, the differences are -15.46
points (-19.47 to -11.61) and -18.42 points (-22.76 to -14.03), respectively.
The reproducible trace-replay artifact is under
`results/analysis/eotbench/20260906T083000Z/`.

### TurnSense in a streaming setting

TurnSense's FastTurn advantage does not transfer. Measured on eot-bench with a
predeclared `invalid` policy, on the same 100 ms decision grid as VAP:

| Language | Model | FC @300 ms | FC @600 ms | Latency @5% |
|---|---|---:|---:|---:|
| ZH | DualTurn | 23.36% | 10.20% | 910 ms |
| ZH | Realtime-VAP 20 s | 25.66% | 9.87% | 910 ms |
| ZH | **TurnSense** | **49.18%** | 15.79% | 990 ms |
| EN | DualTurn | 22.13% | 9.65% | 991 ms |
| EN | Realtime-VAP 20 s | 30.35% | 14.04% | 1184 ms |
| EN | **TurnSense** | **55.60%** | 21.70% | 1608 ms |

The paired conversation-cluster comparison puts TurnSense +25.82 points behind
DualTurn in Chinese, 95% CI [+20.31, +31.42], and +23.52 behind Realtime-VAP,
CI [+18.34, +28.69]. It fails hardest on ordinary 300--500 ms pauses, at 66.0%
against 21.2% for both references.

This settles the Chinese model-selection question that the FastTurn result had
left open. Being the best isolated Complete/Incomplete classifier does not make
a model a usable streaming endpointer: FastTurn asks what a finished segment
was, while eot-bench asks when to cut, and TurnSense is strong at the first and
weak at the second.

The English row deserves one clarification. It is not a degenerate timeout
fallback: all 392 English false cutoffs are model-triggered and none are
timeout-triggered. Under zero-shot English its score distribution collapses
toward zero, with median `p_eot` 0.0054 against 0.7353 in Chinese, so the
prescribed sweep selects a very low threshold at which the model crosses on
essentially every pause beyond 500 ms.

## FastTurn paired EOT analysis

TurnSense and SmartTurn were compared on the identical 18,352 Complete and
Incomplete clips. The paired analysis uses fixed, predeclared model decisions;
no threshold was selected on FastTurn test.

| Slice | N | SmartTurn accuracy | TurnSense accuracy | TurnSense delta |
|---|---:|---:|---:|---:|
| All | 18,352 | 65.33% | 77.93% | +12.59 pp |
| Complete | 14,709 | 70.30% | 79.08% | +8.78 pp |
| Incomplete | 3,643 | 45.26% | 73.26% | +28.00 pp |
| Duration <=1 s | 5,239 | 68.58% | 67.88% | -0.71 pp |
| Duration 1--2 s | 5,569 | 65.43% | 80.28% | +14.85 pp |
| Duration 2--4 s | 4,923 | 62.69% | 82.45% | +19.76 pp |
| Duration 4--8 s | 2,116 | 62.67% | 84.78% | +22.12 pp |
| Duration >8 s | 505 | 67.52% | 83.37% | +15.84 pp |

The paired accuracy delta is +12.59 percentage points with a 95% paired
bootstrap interval of +11.72 to +13.49 points. Of all clips, both models are
correct on 9,627; only TurnSense is correct on 4,674; only SmartTurn is correct
on 2,363; and both are wrong on 1,688. The result is therefore not an artifact
of independent aggregate scoring.

TurnSense's weakness is concentrated in very short utterances. It emits 523
invalid/abstain decisions, including 223 in the <=1-second slice. Short-clip
routing is worth studying, but a naive fallback is not automatically safe.
Replacing TurnSense abstentions with SmartTurn raises accuracy from 77.93% to
79.47% and macro-F1 from 72.50% to 72.66%, but worsens the production-critical
Incomplete-to-Complete error from 23.77% to 25.56%. Of 108 abstained Incomplete
clips, SmartTurn cuts off 65. Such a hybrid must be selected on an external
calibration set with an explicit false-cutoff budget, not on FastTurn test.

### Easy Turn on the same two-state axis

Easy Turn's completed four-state run already covers the 18,352 Complete and
Incomplete clips, so its two-state row is derived by restriction rather than by
rerunning inference (`turn-eval rescore-fastturn-eot`). Predictions of
`backchannel` or `wait` on a two-state clip are abstentions scored as closed-set
errors, which is the policy the source run already declared.

| Model | Complete F1 | Incomplete F1 | Macro-F1 | Inc-to-Com | Abstain | p50 | Size |
|---|---:|---:|---:|---:|---:|---:|---:|
| TurnSense INT8 | 85.51% | 59.48% | 72.50% | 23.77% | 2.85% | 36 ms | 52 MB |
| Easy Turn | 84.67% | 56.34% | 70.50% | 25.75% | 6.22% | 455 ms | 3.45 GB |
| SmartTurn v3.2 | 76.48% | 34.14% | 55.31% | 54.74% | 0.00% | 165 ms | 8.7 MB |

TurnSense leads Easy Turn on every two-state metric at roughly one sixty-sixth
of the size. Easy Turn misroutes 1,135 two-state clips into Backchannel or Wait,
which is 6.19 percentage points of pure abstention loss that the four-state view
hides.

This does not make Easy Turn a weak model; it makes the two views
non-interchangeable. Reading its four-state Macro-F1 of 79.41% against
TurnSense's two-state 72.50% compares different clip sets under different label
vocabularies, and inverts the ordering that holds on a shared axis.

## Interruption

Read this section against
[`BENCHMARK_COVERAGE_AUDIT.md`](BENCHMARK_COVERAGE_AUDIT.md). SID-Bench provides
no agent channel, so a dual-channel model can only be run with causal silence on
channel 2, and that is sound only for a signal that is an absolute property of
the user's own stream. A signal defined relative to the partner channel
collapses, because a silent partner never holds the floor.

Running VapGPT OTO measured this. Its native floor state has median 0.500 on
TurnBench dual-channel audio and saturates above 0.958 on all 1,600 SID English
clips, with the interruption and non-break distributions overlapping almost
completely. With its externally frozen threshold it commits at the first 20 ms
frame of every clip. Mimi and ESPnet commit signals with the same relative
definition and are blocked for the same reason.

Partner independence is necessary but not sufficient. Kyutai's streams are batch
items, verified bit-for-bit independent of the partner slot, so the fallback
costs it nothing, and it still fails: `1 - P(turn ending)` asks "is this speaker
not about to finish", which is true for nearly all of an utterance. Its per-clip
maximum is exactly 1.0000 in every category and all 1,600 clips commit at the
first 80 ms frame.

| EN | Signal | Recall | FIR | Non-break FPR | Middle recall |
|---|---|---:|---:|---:|---:|
| DualTurn | user 0-240 ms future activity | 69.00% | 32.25% | 41.40% | — |
| Realtime-VAP | floor state | 44.00% | 67.25% | 92.40% | — |
| Kyutai | 1 - P(turn ending) | 37.27% | 74.38% | 100.00% | 0.17% |
| VapGPT OTO | native floor state | 31.45% | 78.38% | 100.00% | 0.17% |

A usable SID signal must be low by default and rise at the onset. DualTurn's
0-240 ms forward horizon is the only evaluated signal that is both partner
independent and onset localizing, and 58.6% of English non-break clips never
trigger it. Every other architecture commits at the first frame of every clip.

The consequence is that DualTurn's SID-Bench lead is close to uncontested rather
than close to fair. It is not evidence that its architecture discriminates
interruptions better than the alternatives; it is evidence that the alternatives
cannot be measured on this benchmark. The Realtime-VAP rows below are degenerate
in the same way, so their non-break trigger rates are largely an artifact.

DualTurn's frozen derived-FVAD policy is the strongest current SID-Bench
fallback: EN/ZH recall is 69.00%/81.45%, FIR is 32.25%/14.44%, and APT is
1.494/0.839 seconds. It also reduces the noise false-trigger rate from VAP's
91.6% to 4.2%. This is strong architecture evidence, not proof of a native
semantic interruption detector: both evaluated SID policies use user-only
audio with a silent agent-channel fallback, and DualTurn's interruption signal
is derived from future VAD.

The paired fixed-policy analysis localizes the improvement:

| Split / category | DualTurn | VAP | Paired delta [95% CI] |
|---|---:|---:|---:|
| EN overall recall | 69.00% | 44.00% | +25.00 pp [+22.27, +27.64] |
| EN strict FIR | 32.31% | 67.31% | -35.00 pp [-37.44, -32.62] |
| EN middle-break early trigger | 49.83% | 96.67% | -46.83 pp [-51.00, -42.83] |
| EN non-break trigger | 41.40% | 92.40% | -51.00 pp [-55.40, -46.60] |
| ZH overall recall | 81.45% | 47.18% | +34.27 pp [+30.73, +37.91] |
| ZH strict FIR | 14.44% | 64.62% | -50.19 pp [-52.62, -47.69] |
| ZH middle-break early trigger | 15.17% | 90.33% | -75.17 pp [-78.50, -71.67] |
| ZH non-break trigger | 27.80% | 90.60% | -62.80 pp [-67.00, -58.60] |
| Noise trigger | 4.20% | 91.60% | -87.40 pp [-90.20, -84.40] |

DualTurn's gain is therefore semantic/floor-timing behavior rather than merely
better voice activity detection. VAP fires before the annotated break on more
than 90% of middle-break examples in both languages. DualTurn reduces this
failure substantially, although its 41.4% English and 27.8% Chinese non-break
trigger rates are still too high for an unguarded production policy.

There is also a real conservatism trade-off. On Chinese interruption-at-beginning
examples, DualTurn recall is 87.0% versus VAP's 92.2%, a paired -5.2-point
difference with a 95% interval of -9.0 to -1.4 points. A production policy may
need a fast acoustic path for obvious immediate barge-in and a more conservative
semantic path for middle-turn speech.

## Current decision

- EOT production candidate: DualTurn first, with SmartTurn as the small-model
  reference. Realtime-VAP remains a research baseline because of signal
  provenance and license.
- Isolated Chinese Complete/Incomplete candidate: TurnSense, **for clip
  classification only**. It is the worst streaming endpointer measured, +25.82
  points behind DualTurn on Chinese eot-bench, so it must not be carried over
  into a streaming role on the strength of its FastTurn score.
- Native four-state reference: Easy Turn, with its same-team benchmark and
  possible overlap caveat kept visible.
- Interruption candidate: DualTurn-derived policy only as an interim baseline;
  a native Chinese semantic-interruption model is still an open gap.

The canonical normalized report and plots are under
`results/comparison/20260906T161125Z/`.

The reproducible SID paired report is under
`results/analysis/sidbench/20260906T080000Z/`. It hashes all manifests, frozen
policies, and prediction Parquets used by the analysis.

The completed DualTurn eot-bench artifacts have a metadata erratum:
`manifest.execution.provider` and the backend's fail-closed session check record
CUDA, but `runtime.json.device` inherited the harness's CPU default. The writer
is fixed for future runs. Existing immutable artifacts were not edited, and the
comparison report uses the separately validated isolated CUDA runtime profile.
