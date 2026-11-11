#!/usr/bin/env python3
"""The single registered RSE v2 training run. One seed, fixed budget.

Every value comes from `v2_artifacts/REPRODUCIBILITY_MANIFEST.json`. Nothing
is a command-line choice that could alter the registered experiment: the seed,
schedule, budget, batch size and selection rule are constants here.

Denominator assertions run on this path, not in tests, and exit nonzero naming
the mismatched population. Three denominator failures in this project each
produced a plausible result with a clean verdict line.

Not authorised and not implemented: additional seeds, a longer budget, a
different optimizer, or selecting a checkpoint by ear.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rse.snes_lm.train import Config, Decoder  # noqa: E402
from rse.snes_v2.accounting import Accounting, eligible_from_manifest  # noqa: E402

# --- registered constants. Not configurable. ---
SEED = 0
BATCH = 8
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP = 200
EVAL_EVERY = 600
PATIENCE_EVALS = 8
D_MODEL, N_LAYERS, N_HEADS, DROPOUT = 256, 6, 8, 0.1
CONTEXT = 1024


def load(window_dir: Path, split: str):
    data = np.load(window_dir / f"{split}.npz")
    return (torch.from_numpy(data["tokens"].astype(np.int64)),
            torch.from_numpy(data["mask"]))


def lr_at(step: int, total: int) -> float:
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, total - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model, tokens, mask, device, batch: int) -> tuple[float, int]:
    """Validation bits per SCORED target. Returns the count too, so the
    denominator is asserted rather than assumed."""
    model.eval()
    total_bits, n = 0.0, 0
    for start in range(0, tokens.shape[0], batch):
        x = tokens[start:start + batch].to(device)
        m = mask[start:start + batch].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                               x[:, 1:].reshape(-1), reduction="none")
        keep = m[:, 1:].reshape(-1)
        total_bits += float((loss * keep).sum()) / math.log(2)
        n += int(keep.sum())
    model.train()
    return total_bits / n, n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", default=str(ROOT / "windows"))
    parser.add_argument("--out-dir", default=str(ROOT / "run"))
    args = parser.parse_args()

    window_dir = Path(args.windows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = Accounting(stage="training")
    train_manifest = eligible_from_manifest(window_dir, "train")
    valid_manifest = eligible_from_manifest(window_dir, "valid")

    train_tokens, train_mask = load(window_dir, "train")
    valid_tokens, valid_mask = load(window_dir, "valid")
    acc.check("train_windows", observed=int(train_tokens.shape[0]),
              intended=train_manifest["n_windows"],
              population="train.npz rows vs MANIFEST")
    acc.check("train_scored_targets", observed=int(train_mask.sum()),
              intended=train_manifest["scored_targets"],
              population="train loss mask vs MANIFEST")
    acc.check("valid_scored_targets", observed=int(valid_mask.sum()),
              intended=valid_manifest["scored_targets"],
              population="validation loss mask vs MANIFEST")
    acc.exit_if_failed()

    vocab_size = train_manifest["pad_id"] + 1
    steps_per_epoch = -(-train_manifest["n_windows"] // BATCH)
    total_steps = EPOCHS * steps_per_epoch
    acc.note("steps_per_epoch", steps_per_epoch)
    acc.note("optimizer_updates", total_steps)
    acc.note("scored_target_exposure", EPOCHS * train_manifest["scored_targets"])

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("  REFUSING: the registered run requires CUDA", file=sys.stderr)
        sys.exit(3)
    model = Decoder(vocab_size, Config(
        context=CONTEXT, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
        dropout=DROPOUT, device="cuda")).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    acc.check("n_parameters", observed=n_params, intended=5_646_336,
              population="manifest-declared parameter count")
    acc.exit_if_failed()
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)

    print(f"  {torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}, "
          f"torch {torch.__version__}")
    print(f"  params {n_params:,}  vocab {vocab_size:,}  "
          f"{steps_per_epoch:,} steps/epoch  {total_steps:,} updates")
    print(f"  exposure {EPOCHS * train_manifest['scored_targets']:,} "
          f"scored targets\n", flush=True)

    generator = torch.Generator().manual_seed(SEED)
    history, best = [], {"bits": float("inf"), "step": -1}
    stale = 0
    started = time.time()
    step = 0
    while step < total_steps:
        order = torch.randperm(train_tokens.shape[0], generator=generator)
        for position in range(0, len(order) - BATCH + 1, BATCH):
            if step >= total_steps:
                break
            index = order[position:position + BATCH]
            x = train_tokens[index].to(device, non_blocking=True)
            m = train_mask[index].to(device, non_blocking=True)
            for group in optimiser.param_groups:
                group["lr"] = lr_at(step, total_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                x[:, 1:].reshape(-1), reduction="none")
            keep = m[:, 1:].reshape(-1)
            loss = (loss * keep).sum() / keep.sum().clamp(min=1)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            step += 1

            if step % EVAL_EVERY == 0 or step == total_steps:
                bits, n_valid = evaluate(model, valid_tokens, valid_mask,
                                         device, BATCH)
                if n_valid != valid_manifest["scored_targets"]:
                    print(f"  POPULATION MISMATCH validation: {n_valid:,} vs "
                          f"{valid_manifest['scored_targets']:,}",
                          file=sys.stderr)
                    sys.exit(2)
                row = {"step": step, "epoch": step / steps_per_epoch,
                       "train_bits": float(loss) / math.log(2),
                       "valid_bits": bits, "lr": lr_at(step, total_steps),
                       "grad_norm": float(grad),
                       "elapsed_sec": round(time.time() - started, 1)}
                history.append(row)
                improved = bits < best["bits"]
                if improved:
                    best = {"bits": bits, "step": step}
                    torch.save({"model": model.state_dict(), "step": step,
                                "valid_bits": bits, "seed": SEED,
                                "vocab_size": vocab_size,
                                "config": {"context": CONTEXT,
                                           "d_model": D_MODEL,
                                           "n_layers": N_LAYERS,
                                           "n_heads": N_HEADS,
                                           "dropout": DROPOUT}},
                               out_dir / "best.pt")
                    stale = 0
                else:
                    stale += 1
                print(f"  step {step:6}/{total_steps} ep {row['epoch']:5.2f}  "
                      f"valid {bits:.4f}  train {row['train_bits']:.4f}  "
                      f"lr {row['lr']:.2e}  |g| {row['grad_norm']:.2f}"
                      f"{'  *best' if improved else ''}", flush=True)
                (out_dir / "history.json").write_text(json.dumps(history, indent=1))
                if stale >= PATIENCE_EVALS:
                    print(f"  early stop: {stale} evaluations without "
                          f"improvement", flush=True)
                    step = total_steps
                    break

    acc.note("selected_step", best["step"])
    acc.note("selected_valid_bits", best["bits"])
    acc.note("total_steps_run", len(history) and history[-1]["step"])
    acc.note("wall_clock_sec", round(time.time() - started, 1))
    acc.note("environment", {"gpu": torch.cuda.get_device_name(0),
                             "cuda": torch.version.cuda,
                             "torch": torch.__version__,
                             "precision": "bf16 autocast, fp32 loss"})
    acc.write(out_dir / "training_accounting.json")
    print(f"\n  SELECTED step {best['step']} at {best['bits']:.4f} "
          f"validation bits/scored target")
    print(f"  wall clock {time.time() - started:.0f}s")
    print(f"  artifacts: {out_dir}/best.pt, history.json, "
          f"training_accounting.json")
    acc.exit_if_failed()


if __name__ == "__main__":
    main()
