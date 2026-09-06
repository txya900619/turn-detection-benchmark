# SID-Bench Paired Fixed-Policy Analysis

Thresholds remain frozen at the values recorded in each input run. The 95%
intervals use paired clip bootstrap resampling; no SID-Bench label is used
for calibration. Negative strict-FIR and APT deltas favor the candidate.

## EN

Candidate: `dualturn-endpointing`; reference: `maai-realtime-vap-trilingual-cpc-10hz-5s`.

| Metric | Candidate | Reference | Paired delta [95% CI] |
|---|---:|---:|---:|
| Recall | 69.00% | 44.00% | +25.00 pp [+22.27, +27.64] |
| Strict FIR | 32.31% | 67.31% | -35.00 pp [-37.44, -32.62] |
| APT | 1.494 s | 2.521 s | -1.03 s [-1.15, -0.91] |

| Category | Metric | Candidate | Reference | Paired delta [95% CI] |
|---|---|---:|---:|---:|
| interruption_beginning | interruption_recall | 94.60% | 92.80% | +1.80 pp [-0.40, +4.20] |
| interruption_beginning | early interruption | 2.20% | 7.00% | -4.80 pp [-6.80, -3.00] |
| interruption_middle | interruption_recall | 47.67% | 3.33% | +44.33 pp [+40.33, +48.17] |
| interruption_middle | early interruption | 49.83% | 96.67% | -46.83 pp [-51.00, -42.83] |
| nonbreak | strict_trigger_rate | 41.40% | 92.40% | -51.00 pp [-55.40, -46.60] |

## ZH

Candidate: `dualturn-endpointing`; reference: `maai-realtime-vap-trilingual-cpc-10hz-5s`.

| Metric | Candidate | Reference | Paired delta [95% CI] |
|---|---:|---:|---:|
| Recall | 81.45% | 47.18% | +34.27 pp [+30.73, +37.91] |
| Strict FIR | 14.44% | 64.62% | -50.19 pp [-52.62, -47.69] |
| APT | 0.839 s | 2.416 s | -1.58 s [-1.73, -1.42] |

| Category | Metric | Candidate | Reference | Paired delta [95% CI] |
|---|---|---:|---:|---:|
| interruption_beginning | interruption_recall | 87.00% | 92.20% | -5.20 pp [-9.00, -1.40] |
| interruption_beginning | early interruption | 0.20% | 7.80% | -7.60 pp [-10.00, -5.40] |
| interruption_middle | interruption_recall | 76.83% | 9.67% | +67.17 pp [+63.00, +71.17] |
| interruption_middle | early interruption | 15.17% | 90.33% | -75.17 pp [-78.50, -71.67] |
| nonbreak | strict_trigger_rate | 27.80% | 90.60% | -62.80 pp [-67.00, -58.60] |

## NOISE

Candidate: `dualturn-endpointing`; reference: `maai-realtime-vap-trilingual-cpc-10hz-5s`.

| Metric | Candidate | Reference | Paired delta [95% CI] |
|---|---:|---:|---:|
| Strict FIR | 4.20% | 91.60% | -87.40 pp [-90.20, -84.40] |
| APT | 0.424 s | 12.303 s | -11.88 s [-13.69, -10.25] |

| Category | Metric | Candidate | Reference | Paired delta [95% CI] |
|---|---|---:|---:|---:|
| noise_or_silence | strict_trigger_rate | 4.20% | 91.60% | -87.40 pp [-90.20, -84.40] |
