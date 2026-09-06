# SID-Bench two-stage interruption policy

The policy was selected on the calibration side of a seeded split and scored
once on the held-out side. The incumbent is scored on the same held-out ids,
so the comparison is like-for-like. No held-out label took part in selection.

## Split

- Dataset: `sid-bench-zh` at revision `6eb13b573ad588646ce0de513d4255e55caf858b`
- Seed 0, calibration fraction 0.30, grouped by `example_id`
- Calibration 480 clips, held-out 1120
- Split identity `37ba74229cc96b16`

| Category | Groups | Calibration | Held-out |
| --- | ---: | ---: | ---: |
| interruption_beginning | 500 | 150 | 350 |
| interruption_middle | 600 | 180 | 420 |
| nonbreak | 500 | 150 | 350 |

## Selected policy

- Objective: maximise interruption recall subject to non-break trigger rate <= 0.10, ties broken by lower average penalty time
- Search space: 1800 configurations
- Slow path: threshold `0.92`, persistence `8` frames of 80.0 ms
- Fast path: commit immediately at `0.999`
- Incumbent for comparison: threshold `0.9896328579889585`, persistence `1`

## Results

| Policy and split | Recall | FIR | Non-break trigger | IRL p50 | IRL p90 | APT | Clips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen policy, calibration | 85.15% | 5.00% | 8.67% | 695 ms | 1402 ms | 0.778 s | 480 |
| Incumbent, calibration | 78.79% | 17.08% | 30.67% | 267 ms | 924 ms | 0.999 s | 480 |
| **Frozen policy, held-out** | 86.36% | 4.29% | 9.43% | 682 ms | 1322 ms | 0.712 s | 1120 |
| **Incumbent, held-out** | 82.60% | 13.30% | 26.57% | 252 ms | 869 ms | 0.771 s | 1120 |

Non-break budget was 10.00% on calibration.

The aggregate APT falls because non-break false triggers dominate it, so it
hides the response-latency cost of the confirmation delay. Read IRL p50 and
p90 alongside it, and the per-category rows below, before treating this as a
uniform improvement.

### Held-out, by category

| Policy | Category | Recall | Early trigger | APT |
| --- | --- | ---: | ---: | ---: |
| Frozen | interruption_beginning | 84.00% | 0.00% | 0.881 s |
| Frozen | interruption_middle | 88.33% | 3.57% | 1.036 s |
| Frozen | nonbreak | n/a | n/a | 0.154 s |
| Incumbent | interruption_beginning | 87.14% | 0.29% | 0.568 s |
| Incumbent | interruption_middle | 78.81% | 13.10% | 1.259 s |
| Incumbent | nonbreak | n/a | n/a | 0.388 s |

## Provenance

- Replayed from `/mnt/md0/user_wayne/turn-detection-benchmark/results/zh/sidbench/dualturn_single_channel_fallback/20260905T095000Z` without rerunning the model.
- The replay reproduces that run's published metrics exactly at the
  incumbent parameters, which is what licenses the comparison.
