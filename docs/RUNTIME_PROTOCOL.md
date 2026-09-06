# Runtime Profiling Protocol

Status: canonical local profiling contract

## Scope

Runtime profiling is separate from conversational latency. It measures model
construction, first output, steady-state inference ticks, process resources,
and artifact size. It does not replace benchmark algorithmic latency or full
voice-agent pipeline latency.

## Isolation and timing

Each profile runs in a fresh Python process with concurrency 1. The profile
records:

- process start to model construction;
- process start to the first model output;
- first-tick inference wall time;
- raw steady-state tick wall times after a declared warmup;
- RTF over only the measured steady-state frames.

The process clock comes from Linux `/proc`; tick timing uses the monotonic
high-resolution Python performance counter. CUDA work is synchronized at each
Realtime-VAP tick. DualTurn's ONNX call copies probability outputs to CPU and
therefore completes before its tick returns.

The canonical initial profiles use 64 warmup frames and 512 measured frames.
The deterministic input is low-amplitude user-channel audio with a silent agent
channel. This fixes compute shape but is not an accuracy workload.

## Meaning of cold

A profile starts a new Python process, but uses warm operating-system and
Hugging Face caches. `cold_first_tick_latency_ms` captures lazy runtime/device
initialization within that fresh process. It has one sample per profile and is
not a distribution. A cold-start distribution requires multiple independent
profiles under the same cache and hardware condition.

## Resource scope

RAM uses `/proc/self/status` `VmRSS` and `VmHWM`. CPU utilization is process CPU
time divided by wall time and can exceed 100% when multiple CPU threads run.
GPU VRAM uses `nvidia-smi` compute-process sampling matched to the profiler PID;
it excludes memory owned by unrelated processes. Profiles record the sampling
interval, logical-core count, configured threads, concurrency, visible CUDA
devices, and hardware/software identity.

Model artifact size is the pinned local checkpoint payload. Realtime-VAP totals
its VAP checkpoint and CPC file; DualTurn reports its self-contained streaming
ONNX graph. Python environments, CUDA libraries, and caches are not included in
artifact size.

## Comparison rules

Runtime rows may be compared directly only when frame interval, batch size,
concurrency, warmup, measurement count, hardware, and resource scope are shown.
CPU and GPU rows are deployment alternatives, not interchangeable repeats.
Thread counts must remain visible. Raw tick arrays are required so every
reported percentile can be independently recomputed.

The canonical comparison artifacts are stored in
`results/runtime/20260905T103200Z/`.
