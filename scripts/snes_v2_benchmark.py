#!/usr/bin/env python3
"""G3 hardware benchmark. Measures feasibility; trains nothing.

**No checkpoint is saved and no sample is generated.** This answers one
question: does the 10.6x larger vocabulary change the practical training
configuration? Level 1's 0.99 GB is historical context on different hardware
and a different measurement procedure, not a comparable figure.

Real exported G3 windows are used -- actual snapshots, padding and loss mask --
because those determine the logits shape and the scored-target count. A random
tensor of the same shape measures the hardware, not this training job.

Reported per batch size: peak ALLOCATED and peak RESERVED separately (reserved
is what the allocator holds and what a second process would collide with --
the distinction that cost this project a 502 when memory was eyeballed rather
than measured), steady-state ms/step with warmup excluded, raw token
throughput, scored-target throughput, and gradient-free evaluation throughput.
AdamW state is materialised before measurement, since its two moments per
parameter appear only after the first step.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rse.snes_lm.train import Config, Decoder  # noqa: E402

WARMUP = 5
MEASURE = 25


def environment(device: torch.device) -> dict:
    info = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device_type": device.type,
    }
    if device.type == "cuda":
        info.update({
            "gpu": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        })
    return info


def measure(model, optimiser, tokens, mask, pad_id, device, *,
            batch: int, autocast: bool) -> dict:
    n = tokens.shape[0]
    steps = WARMUP + MEASURE
    times: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(steps):
        start = (step * batch) % max(1, n - batch)
        x = tokens[start:start + batch].to(device, non_blocking=True).long()
        m = mask[start:start + batch].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=autocast):
            logits = model(x[:, :-1])
            target = x[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                target.reshape(-1), reduction="none")
            loss = (loss * m[:, 1:].reshape(-1)).sum() / m[:, 1:].sum().clamp(min=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if step >= WARMUP:
            times.append(time.perf_counter() - t0)

    median = st.median(times)
    scored = int(mask[:batch, 1:].sum())
    out = {
        "batch": batch,
        "autocast_bf16": autocast,
        "median_ms_per_step": 1000 * median,
        "p90_ms_per_step": 1000 * sorted(times)[int(0.9 * len(times))],
        "raw_tokens_per_sec": batch * (tokens.shape[1] - 1) / median,
        "scored_targets_per_sec": scored / median,
    }
    if device.type == "cuda":
        out["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
        out["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
    return out


@torch.no_grad()
def measure_eval(model, tokens, mask, device, *, batch: int,
                 autocast: bool) -> dict:
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for step in range(WARMUP + MEASURE):
        start = (step * batch) % max(1, tokens.shape[0] - batch)
        x = tokens[start:start + batch].to(device).long()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=autocast):
            model(x[:, :-1])
        if device.type == "cuda":
            torch.cuda.synchronize()
        if step >= WARMUP:
            times.append(time.perf_counter() - t0)
    model.train()
    out = {"batch": batch, "median_ms_per_step": 1000 * st.median(times),
           "raw_tokens_per_sec": batch * (tokens.shape[1] - 1) / st.median(times)}
    if device.type == "cuda":
        out["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
        out["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    # Default relative to the SCRIPT, not to $HOME: a transferred bundle has
    # its own windows/ directory, and resolving Path.home() on a rented pod
    # pointed at /root/workplace/... and failed.
    parser.add_argument("--windows", default=str(
        Path(__file__).resolve().parent.parent / "windows"))
    parser.add_argument("--batches", default="4,8,16")
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument("--out", default=str(Path.home() /
                        "workplace/v2_artifacts/benchmark.json"))
    args = parser.parse_args()

    window_dir = Path(args.windows)
    manifest = json.loads((window_dir / "MANIFEST.json").read_text())
    data = np.load(window_dir / "train.npz")
    tokens = torch.from_numpy(data["tokens"].astype(np.int32))
    mask = torch.from_numpy(data["mask"])
    pad_id = manifest["pad_id"]
    # PAD needs an embedding row, so the model's vocabulary is one past it.
    vocab_size = pad_id + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(context=manifest["context"], d_model=256, n_layers=6,
                 n_heads=8, dropout=0.1, device=device.type)
    model = Decoder(vocab_size, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=0.01)

    report = {
        "schema": "snes_v2.benchmark.v1",
        "environment": environment(device),
        "model": {"vocab_size": vocab_size, "n_parameters": n_params,
                  "d_model": 256, "n_layers": 6, "n_heads": 8,
                  "context": manifest["context"], "weight_tied": True},
        "data": {"source": "real exported G3 windows",
                 "n_windows": int(tokens.shape[0]),
                 "scored_targets_total": int(mask.sum()),
                 **manifest["splits"].get("train", {})},
        "warmup_steps": WARMUP, "measured_steps": MEASURE,
        "note": "Trains nothing; no checkpoint saved, no samples generated.",
    }
    print(f"  device {device.type}  params {n_params:,}  vocab {vocab_size:,}")
    if device.type == "cuda":
        print(f"  {report['environment']['gpu']}, CUDA "
              f"{report['environment']['cuda']}, "
              f"{report['environment']['total_memory_gb']:.1f} GB")
    else:
        print("  NO CUDA: this cannot produce the required GPU measurements")

    autocast = (not args.no_autocast) and device.type == "cuda"
    report["train"] = []
    for batch in [int(b) for b in args.batches.split(",")]:
        try:
            row = measure(model, optimiser, tokens, mask, pad_id, device,
                          batch=batch, autocast=autocast)
        except torch.cuda.OutOfMemoryError:
            row = {"batch": batch, "oom": True}
            torch.cuda.empty_cache()
        report["train"].append(row)
        print(f"  train b{batch:<3} {row.get('median_ms_per_step', float('nan')):7.1f} ms/step"
              f"  {row.get('scored_targets_per_sec', 0):9,.0f} scored/s"
              f"  alloc {row.get('peak_allocated_gb', 0):.2f} GB"
              f"  reserved {row.get('peak_reserved_gb', 0):.2f} GB", flush=True)

    report["eval"] = []
    for batch in [int(b) for b in args.batches.split(",")]:
        try:
            row = measure_eval(model, tokens, mask, device, batch=batch,
                               autocast=autocast)
        except torch.cuda.OutOfMemoryError:
            row = {"batch": batch, "oom": True}
            torch.cuda.empty_cache()
        report["eval"].append(row)
        print(f"  eval  b{batch:<3} {row.get('median_ms_per_step', float('nan')):7.1f} ms/step"
              f"  alloc {row.get('peak_allocated_gb', 0):.2f} GB", flush=True)

    # Epoch accounting, so an equal-step budget is not mistaken for equal data.
    for row in report["train"]:
        if "median_ms_per_step" not in row:
            continue
        steps = int(np.ceil(tokens.shape[0] / row["batch"]))
        row["steps_per_epoch"] = steps
        row["seconds_per_epoch"] = steps * row["median_ms_per_step"] / 1000
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
