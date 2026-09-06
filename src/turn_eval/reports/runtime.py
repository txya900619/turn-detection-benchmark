from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def runtime_profile_row(path: Path) -> dict[str, object]:
    profile = json.loads(path.read_text())
    if profile.get("schema_version") != 1:
        raise ValueError(f"unsupported runtime profile schema: {path}")
    if profile.get("profile_type") != "isolated-cold-start-and-warm-streaming-ticks":
        raise ValueError(f"unexpected runtime profile type: {path}")
    measurement = profile["measurement"]
    raw = np.asarray(measurement.get("warm_tick_latencies_ms"), dtype=np.float64)
    if raw.ndim != 1 or not len(raw) or not np.all(np.isfinite(raw)) or np.any(raw < 0):
        raise ValueError(f"invalid warm tick latency trace: {path}")
    if len(raw) != measurement["measurement_frames"]:
        raise ValueError(f"runtime trace count mismatch: {path}")
    summary = measurement["warm_inference_latency_ms"]
    recomputed = {
        "count": len(raw),
        "mean": float(np.mean(raw)),
        "p50": _percentile(raw, 0.50),
        "p90": _percentile(raw, 0.90),
        "p95": _percentile(raw, 0.95),
        "p99": _percentile(raw, 0.99),
        "minimum": float(np.min(raw)),
        "maximum": float(np.max(raw)),
    }
    for key, value in recomputed.items():
        if not np.isclose(float(summary[key]), float(value), rtol=1e-12, atol=1e-9):
            raise ValueError(f"runtime summary mismatch for {key}: {path}")

    model = profile["model"]
    execution = profile["execution"]
    resources = profile["resources"]
    artifact_size = model.get("artifact_size_bytes")
    if artifact_size is None:
        artifact_size = model["artifact_sizes"]["total_bytes"]
    return {
        "model": model["name"],
        "backend": execution.get("provider", execution.get("device")),
        "threads": execution["threads"],
        "frame_interval_ms": execution["frame_interval_ms"],
        "process_to_constructed_s": measurement["process_start_to_model_constructed_s"],
        "process_to_first_output_s": measurement["process_start_to_first_output_s"],
        "cold_first_tick_ms": measurement["cold_first_tick_latency_ms"],
        "warm_tick_count": len(raw),
        "warm_mean_ms": recomputed["mean"],
        "warm_p50_ms": recomputed["p50"],
        "warm_p90_ms": recomputed["p90"],
        "warm_p95_ms": recomputed["p95"],
        "warm_p99_ms": recomputed["p99"],
        "rtf": measurement["rtf"],
        "realtime_multiplier": measurement["realtime_multiplier"],
        "process_cpu_percent": resources["cpu_utilization_percent"]["mean_over_profile"],
        "ram_peak_bytes": resources["ram"]["peak_rss_bytes"],
        "gpu_vram_peak_bytes": resources["gpu_vram"]["peak_bytes"],
        "model_artifact_size_bytes": artifact_size,
        "profile": str(path.resolve()),
    }


def _mib(value: object) -> str:
    return "n/a" if value is None else f"{float(value) / 1_048_576:.1f} MiB"


def render_runtime_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Isolated Runtime Profiles",
        "",
        "Each row is a fresh Python process with concurrency 1. Cold first-tick values have",
        "one sample per row and use warm OS/Hugging Face caches; warm latency statistics are",
        "recomputed from the raw per-tick trace. CPU utilization is process CPU time divided",
        "by wall time and may exceed 100% for multi-threaded runs.",
        "",
        "| Model | Backend | Threads | Constructed | First output | Cold tick | "
        "Warm p50 | Warm p90 | Warm p95 | RTF | CPU | RAM peak | VRAM peak | Artifacts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {backend} | {threads} | {constructed:.3f}s | {first:.3f}s | "
            "{cold:.1f}ms | {p50:.1f}ms | {p90:.1f}ms | {p95:.1f}ms | {rtf:.3f} | "
            "{cpu:.1f}% | {ram} | {vram} | {artifacts} |".format(
                model=row["model"],
                backend=row["backend"],
                threads=row["threads"],
                constructed=float(row["process_to_constructed_s"]),
                first=float(row["process_to_first_output_s"]),
                cold=float(row["cold_first_tick_ms"]),
                p50=float(row["warm_p50_ms"]),
                p90=float(row["warm_p90_ms"]),
                p95=float(row["warm_p95_ms"]),
                rtf=float(row["rtf"]),
                cpu=float(row["process_cpu_percent"]),
                ram=_mib(row["ram_peak_bytes"]),
                vram=_mib(row["gpu_vram_peak_bytes"]),
                artifacts=_mib(row["model_artifact_size_bytes"]),
            )
        )
    return "\n".join(lines) + "\n"


def write_runtime_report(paths: list[Path], output_dir: Path) -> Path:
    if not paths:
        raise ValueError("runtime report requires at least one profile")
    rows = [runtime_profile_row(path) for path in paths]
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (output_dir / "summary.json", output_dir / "summary.csv", output_dir / "report.md")
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"runtime report output already exists: {output_dir}")
    outputs[0].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_count": len(rows),
                "profiles": [str(path.resolve()) for path in paths],
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with outputs[1].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outputs[2].write_text(render_runtime_markdown(rows), encoding="utf-8")
    return outputs[2]
