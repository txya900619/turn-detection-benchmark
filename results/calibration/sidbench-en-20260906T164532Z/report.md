# SID-Bench two-stage interruption policy

The policy was selected on the calibration side of a seeded split and scored
once on the held-out side. The incumbent is scored on the same held-out ids,
so the comparison is like-for-like. No held-out label took part in selection.

## Split

- Dataset: `sid-bench-en` at revision `6eb13b573ad588646ce0de513d4255e55caf858b`
- Seed 0, calibration fraction 0.30, grouped by `example_id`
- Calibration 480 clips, held-out 1120
- Split identity `17c047ab6f616217`

| Category | Groups | Calibration | Held-out |
| --- | ---: | ---: | ---: |
| interruption_beginning | 500 | 150 | 350 |
| interruption_middle | 600 | 180 | 420 |
| nonbreak | 500 | 150 | 350 |

## Selected policy

- Objective: maximise interruption recall subject to non-break trigger rate <= 0.10, ties broken by lower average penalty time
- Search space: 184 configurations
- Threshold `0.97`, persistence `8` frames of 80.0 ms
- Incumbent for comparison: threshold `0.9896328579889585`, persistence `1`

## Results

| Policy and split | Recall | FIR | Non-break trigger | IRL p50 | IRL p90 | APT | Clips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen policy, calibration | 87.27% | 5.42% | 4.00% | 670 ms | 1313 ms | 0.866 s | 480 |
| Incumbent, calibration | 72.42% | 30.42% | 42.00% | 178 ms | 534 ms | 1.400 s | 480 |
| **Frozen policy, held-out** | 82.99% | 7.14% | 6.00% | 683 ms | 1490 ms | 0.984 s | 1120 |
| **Incumbent, held-out** | 67.53% | 33.04% | 41.14% | 178 ms | 660 ms | 1.535 s | 1120 |

Non-break budget was 10.00% on calibration.

The aggregate APT falls because non-break false triggers dominate it, so it
hides the response-latency cost of the confirmation delay. Read IRL p50 and
p90 alongside it, and the per-category rows below, before treating this as a
uniform improvement.

### Held-out, by category

| Policy | Category | Recall | Early trigger | APT |
| --- | --- | ---: | ---: | ---: |
| Frozen | interruption_beginning | 85.71% | 0.00% | 0.876 s |
| Frozen | interruption_middle | 80.71% | 14.05% | 1.817 s |
| Frozen | nonbreak | n/a | n/a | 0.092 s |
| Incumbent | interruption_beginning | 94.00% | 2.00% | 0.320 s |
| Incumbent | interruption_middle | 45.48% | 52.14% | 3.496 s |
| Incumbent | nonbreak | n/a | n/a | 0.395 s |

## Provenance

- Replayed from `/mnt/md0/user_wayne/turn-detection-benchmark/results/en/sidbench/dualturn_single_channel_fallback/20260905T095000Z` without rerunning the model.
- The replay reproduces that run's published metrics exactly at the
  incumbent parameters, which is what licenses the comparison.
