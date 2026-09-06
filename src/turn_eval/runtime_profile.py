from __future__ import annotations

import os
import platform
import statistics
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic, process_time

import numpy as np


def process_age_seconds(pid: int | None = None) -> float | None:
    target = os.getpid() if pid is None else pid
    try:
        stat = Path(f"/proc/{target}/stat").read_text()
        fields_from_state = stat.rpartition(") ")[2].split()
        start_ticks = int(fields_from_state[19])
        uptime_s = float(Path("/proc/uptime").read_text().split()[0])
        return max(0.0, uptime_s - start_ticks / os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError, IndexError):
        return None


def summarize_latencies_seconds(values: list[float]) -> dict[str, float | int | None]:
    samples = np.asarray(values, dtype=np.float64) * 1_000

    def percentile(q: float) -> float | None:
        return float(np.quantile(samples, q)) if len(samples) else None

    return {
        "count": len(samples),
        "mean": float(np.mean(samples)) if len(samples) else None,
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "minimum": float(np.min(samples)) if len(samples) else None,
        "maximum": float(np.max(samples)) if len(samples) else None,
    }


def _proc_memory_bytes(pid: int) -> tuple[int | None, int | None]:
    path = Path(f"/proc/{pid}/status")
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None, None
    values: dict[str, int] = {}
    for line in lines:
        name, separator, remainder = line.partition(":")
        if not separator or name not in {"VmRSS", "VmHWM"}:
            continue
        fields = remainder.split()
        if len(fields) >= 2 and fields[1] == "kB":
            values[name] = int(fields[0]) * 1_024
    return values.get("VmRSS"), values.get("VmHWM")


def _gpu_vram_bytes(pid: int) -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode:
        return None
    total_mib = 0
    found = False
    for line in completed.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            row_pid, used_mib = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        if row_pid == pid:
            total_mib += used_mib
            found = True
    return total_mib * 1_024 * 1_024 if found else 0


def hardware_identity() -> dict[str, object]:
    cpu_name = platform.processor() or None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_name = line.partition(":")[2].strip()
                break
    except OSError:
        pass
    gpu_rows: list[dict[str, object]] = []
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if completed.returncode == 0:
            for line in completed.stdout.splitlines():
                fields = [value.strip() for value in line.split(",", maxsplit=3)]
                if len(fields) != 4:
                    continue
                gpu_rows.append(
                    {
                        "physical_index": int(fields[0]),
                        "name": fields[1],
                        "total_memory_bytes": int(fields[2]) * 1_024 * 1_024,
                        "uuid": fields[3],
                    }
                )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": cpu_name,
        "logical_cpu_count": os.cpu_count(),
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": gpu_rows,
    }


@dataclass(slots=True)
class ProcessResourceSampler:
    interval_s: float = 0.1
    sample_gpu: bool = False
    _pid: int = field(init=False, repr=False)
    _stop: threading.Event = field(init=False, repr=False)
    _thread: threading.Thread | None = field(init=False, default=None, repr=False)
    _rss: list[int] = field(init=False, default_factory=list, repr=False)
    _hwm: list[int] = field(init=False, default_factory=list, repr=False)
    _vram: list[int] = field(init=False, default_factory=list, repr=False)
    _cpu_percent: list[float] = field(init=False, default_factory=list, repr=False)
    _started_wall: float | None = field(init=False, default=None, repr=False)
    _started_cpu: float | None = field(init=False, default=None, repr=False)
    _baseline_rss: int | None = field(init=False, default=None, repr=False)
    _baseline_vram: int | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.interval_s <= 0:
            raise ValueError("resource sampling interval must be positive")
        self._pid = os.getpid()
        self._stop = threading.Event()

    def start(self) -> ProcessResourceSampler:
        if self._thread is not None:
            raise RuntimeError("resource sampler is already running")
        self._started_wall = monotonic()
        self._started_cpu = process_time()
        self._baseline_rss, baseline_hwm = _proc_memory_bytes(self._pid)
        if self._baseline_rss is not None:
            self._rss.append(self._baseline_rss)
        if baseline_hwm is not None:
            self._hwm.append(baseline_hwm)
        if self.sample_gpu:
            self._baseline_vram = _gpu_vram_bytes(self._pid)
            if self._baseline_vram is not None:
                self._vram.append(self._baseline_vram)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def _sample_loop(self) -> None:
        previous_wall = monotonic()
        previous_cpu = process_time()
        while not self._stop.wait(self.interval_s):
            current_wall = monotonic()
            current_cpu = process_time()
            elapsed = current_wall - previous_wall
            if elapsed > 0:
                self._cpu_percent.append((current_cpu - previous_cpu) / elapsed * 100)
            previous_wall, previous_cpu = current_wall, current_cpu
            rss, hwm = _proc_memory_bytes(self._pid)
            if rss is not None:
                self._rss.append(rss)
            if hwm is not None:
                self._hwm.append(hwm)
            if self.sample_gpu:
                vram = _gpu_vram_bytes(self._pid)
                if vram is not None:
                    self._vram.append(vram)

    def stop(self) -> dict[str, object]:
        if self._thread is None or self._started_wall is None or self._started_cpu is None:
            raise RuntimeError("resource sampler is not running")
        self._stop.set()
        self._thread.join(timeout=max(2, self.interval_s * 4))
        final_wall = monotonic()
        final_cpu = process_time()
        rss, hwm = _proc_memory_bytes(self._pid)
        if rss is not None:
            self._rss.append(rss)
        if hwm is not None:
            self._hwm.append(hwm)
        if self.sample_gpu:
            vram = _gpu_vram_bytes(self._pid)
            if vram is not None:
                self._vram.append(vram)
        wall_s = final_wall - self._started_wall
        cpu_s = final_cpu - self._started_cpu
        peak_rss = max((*self._rss, *self._hwm), default=None)
        peak_vram = max(self._vram, default=None)
        result = {
            "sampling_interval_ms": self.interval_s * 1_000,
            "sample_count": len(self._rss),
            "process_scope": f"pid:{self._pid}",
            "wall_s": wall_s,
            "cpu_time_s": cpu_s,
            "cpu_utilization_percent": {
                "definition": "process CPU time / wall time; may exceed 100%",
                "mean_over_profile": cpu_s / wall_s * 100 if wall_s else None,
                "sample_mean": (statistics.fmean(self._cpu_percent) if self._cpu_percent else None),
                "sample_peak": max(self._cpu_percent, default=None),
                "logical_core_count": os.cpu_count(),
            },
            "ram": {
                "baseline_rss_bytes": self._baseline_rss,
                "final_rss_bytes": rss,
                "peak_rss_bytes": peak_rss,
                "peak_minus_baseline_bytes": (
                    peak_rss - self._baseline_rss
                    if peak_rss is not None and self._baseline_rss is not None
                    else None
                ),
                "source": "/proc/self/status VmRSS/VmHWM",
            },
            "gpu_vram": {
                "baseline_bytes": self._baseline_vram,
                "peak_bytes": peak_vram,
                "peak_minus_baseline_bytes": (
                    peak_vram - self._baseline_vram
                    if peak_vram is not None and self._baseline_vram is not None
                    else None
                ),
                "source": "nvidia-smi compute-apps PID sampling" if self.sample_gpu else None,
            },
        }
        self._thread = None
        return result

    def __enter__(self) -> ProcessResourceSampler:
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._thread is not None:
            self.stop()
