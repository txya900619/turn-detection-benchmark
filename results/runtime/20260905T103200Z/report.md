# Isolated Runtime Profiles

Each row is a fresh Python process with concurrency 1. Cold first-tick values have
one sample per row and use warm OS/Hugging Face caches; warm latency statistics are
recomputed from the raw per-tick trace. CPU utilization is process CPU time divided
by wall time and may exceed 100% for multi-threaded runs.

| Model | Backend | Threads | Constructed | First output | Cold tick | Warm p50 | Warm p90 | Warm p95 | RTF | CPU | RAM peak | VRAM peak | Artifacts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dualturn-endpointing | cpu | 4 | 2.570s | 2.620s | 53.4ms | 66.8ms | 85.6ms | 88.2ms | 0.867 | 386.6% | 487.3 MiB | n/a | 154.2 MiB |
| dualturn-endpointing | cuda | 1 | 5.410s | 6.040s | 628.3ms | 13.5ms | 18.5ms | 20.0ms | 0.177 | 98.2% | 1176.8 MiB | 742.0 MiB | 154.2 MiB |
| maai-realtime-vap-trilingual-cpc-10hz-5s | cpu | 4 | 2.890s | 2.980s | 90.8ms | 22.1ms | 29.3ms | 30.9ms | 0.236 | 392.2% | 652.9 MiB | n/a | 30.4 MiB |
| maai-realtime-vap-trilingual-cpc-10hz-5s | cuda:0 | 1 | 3.450s | 4.290s | 844.3ms | 21.9ms | 28.3ms | 30.0ms | 0.232 | 97.7% | 1249.0 MiB | 322.0 MiB | 30.4 MiB |
