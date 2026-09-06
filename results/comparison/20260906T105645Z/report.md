# Turn Detection Baseline Comparison

No overall score is computed. Rows with different channel modes or signal provenance
are not directly interchangeable. TurnBench thresholds are selected and scored on dev;
SID-Bench policies are frozen externally; eot-bench reports its prescribed policy sweep.

## EOT / HOLD — EN

| Model | Role | Mode | Cutoff @300 | Cutoff @600 | Latency @5% | Latency @10% | License |
|---|---|---|---:|---:|---:|---:|---|
| dualturn-endpointing | baseline | single_channel_fallback / streaming / native | 22.13% | 9.65% | 991 ms | 566 ms | Apache-2.0 |
| fixed-timeout-oracle-vad | diagnostic | native_single / pointwise / derived | 55.60% | 21.70% | 1600 ms | 1000 ms | N/A |
| maai-realtime-vap-trilingual-cpc-10hz-5s | baseline | single_channel_fallback / streaming / derived | 33.33% | 13.62% | 1016 ms | 702 ms | CC-BY-NC-SA-4.0 |
| smart-turn-v3.2-cpu | baseline | native_single / pointwise / native | 37.59% | 14.75% | 1044 ms | 755 ms | BSD-2-Clause |

## EOT / HOLD — ZH

| Model | Role | Mode | Cutoff @300 | Cutoff @600 | Latency @5% | Latency @10% | License |
|---|---|---|---:|---:|---:|---:|---|
| dualturn-endpointing | baseline | single_channel_fallback / streaming / native | 23.36% | 10.20% | 910 ms | 612 ms | Apache-2.0 |
| fixed-timeout-oracle-vad | diagnostic | native_single / pointwise / derived | 51.15% | 16.28% | 1100 ms | 800 ms | N/A |
| maai-realtime-vap-trilingual-cpc-10hz-5s | baseline | single_channel_fallback / streaming / derived | 31.25% | 9.70% | 879 ms | 597 ms | CC-BY-NC-SA-4.0 |
| smart-turn-v3.2-cpu | baseline | native_single / pointwise / native | 41.78% | 12.83% | 928 ms | 667 ms | BSD-2-Clause |

## TurnBench EN dev

| Model | Task | Role | Mode / signal | Recall | FPR | Latency p50 | Latency p90 | License |
|---|---|---|---|---:|---:|---:|---:|---|
| dualturn-endpointing | eot | baseline | native-dual-channel-two-orientations / streaming / native | 49.53% | 9.69% | 539 ms | 1282 ms | Apache-2.0 |
| dualturn-endpointing | interruption | baseline | native-dual-channel-two-orientations / streaming / derived-fvad | 96.25% | 9.99% | 548 ms | 1681 ms | Apache-2.0 |
| espnet-turntaking-swbd-mixed-mono | eot | baseline | mixed-mono-single-stream-derived-speaker-attribution / causal-prefix-then-30s-sliding-window / native-class-probability-derived-speaker-attribution | 83.72% | 8.84% | 922 ms | 2135 ms | CC-BY-4.0 |
| espnet-turntaking-swbd-mixed-mono | interruption | baseline | mixed-mono-single-stream-derived-speaker-attribution / causal-prefix-then-30s-sliding-window / native-class-probability-derived-speaker-attribution | 64.27% | 9.08% | 227 ms | 1428 ms | CC-BY-4.0 |
| kyutai-semantic-vad-stt-1b-en-fr-candle | eot | baseline | native-dual-channel / streaming-batched-two-speaker / native-vad-head | 79.67% | 9.97% | 1099 ms | 1767 ms | CC-BY-4.0 |
| kyutai-semantic-vad-stt-1b-en-fr-candle | interruption | baseline | native-dual-channel / streaming-batched-two-speaker / derived | 93.08% | 8.52% | 579 ms | 1410 ms | CC-BY-4.0 |
| maai-realtime-vap-trilingual-cpc-10hz-5s | eot | baseline | native-dual-channel / streaming / derived | 82.41% | 9.97% | 333 ms | 1628 ms | CC-BY-NC-SA-4.0 |
| maai-realtime-vap-trilingual-cpc-10hz-5s | interruption | baseline | native-dual-channel / streaming / derived | 90.78% | 9.48% | 1075 ms | 2210 ms | CC-BY-NC-SA-4.0 |
| mimi-endpointer-oto-d1f | eot | baseline | native-dual-channel / causal-model-full-sequence-inference / derived | 75.95% | 4.52% | 752 ms | 1862 ms | CC-BY-4.0 |
| mimi-endpointer-oto-d1f | interruption | baseline | native-dual-channel / causal-model-full-sequence-inference / derived-floor-state | 87.61% | 9.78% | 821 ms | 2101 ms | CC-BY-4.0 |
| smart-turn-v3.1-gpu | eot | baseline | native-single-channel-per-speaker / streaming-vad-gated-pointwise / native | 73.48% | 9.97% | 1012 ms | 1225 ms | BSD-2-Clause |
| smart-turn-v3.1-gpu | interruption | baseline | native-single-channel-per-speaker / streaming-vad-gated-pointwise / derived | 12.10% | 9.99% | 173 ms | 703 ms | BSD-2-Clause |
| smart-turn-v3.2-gpu | eot | baseline | native-single-channel-per-speaker / streaming-vad-gated-pointwise / native | 76.63% | 9.88% | 1017 ms | 1353 ms | BSD-2-Clause |
| smart-turn-v3.2-gpu | interruption | baseline | native-single-channel-per-speaker / streaming-vad-gated-pointwise / derived | 10.09% | 8.95% | 122 ms | 289 ms | BSD-2-Clause |
| turnbench-rms-vad | eot | diagnostic | native-dual-channel / streaming / derived | 59.45% | 54.66% | -98 ms | 2161 ms | N/A (rule-based) |
| turnbench-rms-vad | interruption | diagnostic | native-dual-channel / streaming / derived | 99.42% | 38.98% | 137 ms | 602 ms | N/A (rule-based) |
| vap-gpt-oto | eot | baseline | native-dual-channel / bounded-context-overlapping-windows / derived | 84.09% | 4.52% | 463 ms | 1657 ms | CC-BY-4.0 |
| vap-gpt-oto | interruption | baseline | native-dual-channel / bounded-context-overlapping-windows / native-floor-state | 95.68% | 9.99% | 896 ms | 2002 ms | CC-BY-4.0 |

## SID-Bench interruption

| Split | Model | Mode / signal | Recall | Official FIR | Strict FIR | IRL p50 | IRL p90 | APT | Non-break FPR | License |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| EN | dualturn-endpointing | single_channel_fallback / streaming / derived-fvad | 69.00% | 32.25% | 32.31% | 178 ms | 611 ms | 1.494 s | 41.40% | Apache-2.0 |
| EN | maai-realtime-vap-trilingual-cpc-10hz-5s | single_channel_fallback / streaming / derived | 44.00% | 67.25% | 67.31% | 62 ms | 150 ms | 2.521 s | 92.40% | CC-BY-NC-SA-4.0 |
| NOISE | dualturn-endpointing | single_channel_fallback / streaming / derived-fvad | n/a | 4.20% | 4.20% | n/a | n/a | 0.424 s | 4.20% | Apache-2.0 |
| NOISE | maai-realtime-vap-trilingual-cpc-10hz-5s | single_channel_fallback / streaming / derived | n/a | 91.60% | 91.60% | n/a | n/a | 12.303 s | 91.60% | CC-BY-NC-SA-4.0 |
| ZH | dualturn-endpointing | single_channel_fallback / streaming / derived-fvad | 81.45% | 14.44% | 14.44% | 260 ms | 876 ms | 0.839 s | 27.80% | Apache-2.0 |
| ZH | maai-realtime-vap-trilingual-cpc-10hz-5s | single_channel_fallback / streaming / derived | 47.18% | 64.50% | 64.62% | 88 ms | 162 ms | 2.416 s | 90.60% | CC-BY-NC-SA-4.0 |

## FastTurn ZH — binary EOT subset

| Model | Mode | Threshold | Complete P/R/F1 | Incomplete P/R/F1 | Macro-F1 | Inc→Com | Abstain | p50 | RTF | License |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| easy-turn | native_single / offline / native | n/a | 92.45% / 78.10% / 84.67% | 49.61% / 65.17% / 56.34% | 70.50% | 25.75% | 6.22% | 455 ms | 0.244 | Apache-2.0 |
| smart-turn-v3.2-cpu | native_single / offline / native | 0.5 | 83.83% / 70.30% / 76.48% | 27.41% / 45.26% / 34.14% | 55.31% | 54.74% | 0.00% | 165 ms | 0.069 | BSD-2-Clause |
| turnsense-int8 | native_single / offline / native | n/a | 93.07% / 79.08% / 85.51% | 50.07% / 73.26% / 59.48% | 72.50% | 23.77% | 2.85% | 36 ms | 0.019 | Apache-2.0 |

## FastTurn ZH — native four-state

This table is not comparable to the binary EOT subset above.

| Model | Mode | Complete F1 | Incomplete F1 | Backchannel F1 | Wait F1 | Macro-F1 | Inc→Com | BC→Com | Abstain | p50 | RTF | License |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| easy-turn | native_single / offline / native | 84.28% | 55.60% | 81.65% | 96.10% | 79.41% | 25.75% | 2.82% | 0.03% | 417 ms | 0.244 | Apache-2.0 |

## Isolated runtime

| Model | Backend | Warm p50 | Warm p90 | RTF | CPU | RAM | VRAM | Artifacts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dualturn-endpointing | cpu | 67 ms | 86 ms | 0.867 | 386.6% | 487 MiB | n/a | 154 MiB |
| dualturn-endpointing | cuda | 14 ms | 19 ms | 0.177 | 98.2% | 1177 MiB | 742 MiB | 154 MiB |
| maai-realtime-vap-trilingual-cpc-10hz-5s | cpu | 22 ms | 29 ms | 0.236 | 392.2% | 653 MiB | n/a | 30 MiB |
| maai-realtime-vap-trilingual-cpc-10hz-5s | cuda:0 | 22 ms | 28 ms | 0.232 | 97.7% | 1249 MiB | 322 MiB | 30 MiB |

## Coverage gaps

FastTurn ZH: binary EOT and native four-state coverage complete in separate tables. X2-Turn remains deferred; partial smoke runs and
fixed-state diagnostics are excluded from baseline tables.
