# EOT-Bench HOLD Error Analysis

Operating point: `false_cutoff_at_300ms`. Every threshold, action delay, and
timeout is replayed from its immutable input metrics. No threshold or policy is
selected by this analysis. Paired intervals use conversation-cluster bootstrap.

## EN

| Model | Threshold | Delay | Timeout | False cutoffs | Model | Timeout | Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| dualturn-endpointing | 0.86 | 200 ms | 3000 ms | 156 / 705 | 155 | 1 | 22.13% |
| maai-realtime-vap-trilingual-cpc-10hz-5s | 0.14 | 200 ms | 1000 ms | 235 / 705 | 200 | 35 | 33.33% |
| smart-turn-v3.2-cpu | 0.07 | 200 ms | 1000 ms | 265 / 705 | 220 | 45 | 37.59% |

Candidate: `dualturn-endpointing`.

| Reference | Candidate-only | Reference-only | Both false | Both safe | Delta [95% CI] |
|---|---:|---:|---:|---:|---:|
| maai-realtime-vap-trilingual-cpc-10hz-5s | 39 | 118 | 117 | 431 | -11.21 pp [-14.71, -7.57] |
| smart-turn-v3.2-cpu | 33 | 142 | 123 | 407 | -15.46 pp [-19.47, -11.61] |

| HOLD duration | N | dualturn-endpointing | maai-realtime-vap-trilingual-cpc-10hz-5s | smart-turn-v3.2-cpu |
|---|---:|---:|---:|---:|
| 200--<300 ms | 173 | 0.0% | 0.0% | 0.0% |
| 300--<500 ms | 246 | 14.2% | 29.3% | 41.9% |
| 500--<1000 ms | 207 | 33.8% | 43.5% | 43.0% |
| 1--<2 s | 61 | 62.3% | 90.2% | 90.2% |
| 2--5 s | 18 | 72.2% | 100.0% | 100.0% |

## ZH

| Model | Threshold | Delay | Timeout | False cutoffs | Model | Timeout | Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| dualturn-endpointing | 0.85 | 200 ms | 1000 ms | 142 / 608 | 136 | 6 | 23.36% |
| maai-realtime-vap-trilingual-cpc-10hz-5s | 0.06 | 200 ms | 1500 ms | 190 / 608 | 189 | 1 | 31.25% |
| smart-turn-v3.2-cpu | 0.41 | 200 ms | 1000 ms | 254 / 608 | 243 | 11 | 41.78% |

Candidate: `dualturn-endpointing`.

| Reference | Candidate-only | Reference-only | Both false | Both safe | Delta [95% CI] |
|---|---:|---:|---:|---:|---:|
| maai-realtime-vap-trilingual-cpc-10hz-5s | 26 | 74 | 116 | 392 | -7.89 pp [-11.60, -4.35] |
| smart-turn-v3.2-cpu | 23 | 135 | 119 | 331 | -18.42 pp [-22.76, -14.03] |

| HOLD duration | N | dualturn-endpointing | maai-realtime-vap-trilingual-cpc-10hz-5s | smart-turn-v3.2-cpu |
|---|---:|---:|---:|---:|
| 200--<300 ms | 179 | 0.0% | 0.0% | 0.0% |
| 300--<500 ms | 212 | 21.2% | 34.9% | 55.2% |
| 500--<1000 ms | 171 | 35.7% | 46.2% | 57.9% |
| 1--<2 s | 45 | 77.8% | 80.0% | 82.2% |
| 2--5 s | 1 | 100.0% | 100.0% | 100.0% |
