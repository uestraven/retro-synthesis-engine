#!/usr/bin/env python3
"""GPU runner for the pre-registered snesmdb Level 1 experiment.

Subcommands, in the order the frozen protocol specifies:

    throughput   measure real step time and VRAM before committing anything
    sweep        the 6 pre-registered configurations, 4000 steps each
    full         the selected configuration to 50,000 steps

Protocol: snesmdb/docs/PROTOCOL_FREEZE.md. Nothing here may change the
tokenizer, population, windowing, masking or target construction.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rse.snes_lm import representation as rep  # noqa: E402
from rse.snes_lm.train import Config, Decoder, train  # noqa: E402

#: Where the window tensors live. They are DERIVED RESEARCH-ONLY DATA and are
#: never committed to this repository, so they are placed here separately.
#: Override with SNES_LM_DATA.
HERE = Path(os.environ.get("SNES_LM_DATA", "/workspace/snes_lm_data"))

#: Pre-registered sweep grid. Fixed before any result was seen; must not grow.
SWEEP = [
    {"lr": lr, "d_model": d, "n_layers": n, "n_heads": h}
    for lr in (1e-4, 3e-4, 1e-3)
    for (d, n, h) in ((256, 6, 8), (224, 8, 8))
]
SWEEP_STEPS = 4_000


def load():
    tr = torch.load(HERE / "snes_train_windows.pt", weights_only=False)
    va = torch.load(HERE / "snes_valid_windows.pt", weights_only=False)
    return tr["windows"], va["windows"], rep.Vocab.from_dict(tr["vocab"])


def device_report() -> dict:
    info = {"cuda": torch.cuda.is_available(), "torch": torch.__version__}
    if info["cuda"]:
        info["name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        )
        info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    return info


def cmd_throughput(args) -> None:
    """Measure what the pod actually does, before committing to a long run."""
    train_w, valid_w, vocab = load()
    info = device_report()
    print(json.dumps(info, indent=1))
    if not info["cuda"]:
        raise SystemExit("no CUDA device: this is the GPU runner")

    device = "cuda"
    for d_model, n_layers, n_heads in ((256, 6, 8), (224, 8, 8)):
        cfg = Config(device=device, d_model=d_model, n_layers=n_layers,
                     n_heads=n_heads, run_dir="/tmp/throughput")
        model = Decoder(vocab.size, cfg).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)
        batch = train_w[: cfg.batch_size].to(device)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=vocab.PAD)

        def step() -> None:
            logits = model(batch)
            loss = loss_fn(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                batch[:, 1:].reshape(-1),
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

        for _ in range(5):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        n = 30
        for _ in range(n):
            step()
        torch.cuda.synchronize()
        per_step = (time.time() - start) / n
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(
            f"  {d_model}x{n_layers}  {model.n_params:,} params  "
            f"{per_step*1000:7.1f} ms/step  {cfg.batch_size*cfg.context/per_step:9.0f} tok/s  "
            f"peak VRAM {peak_gb:.2f} GB"
        )
        print(
            f"     4,000 steps = {4000*per_step/60:5.1f} min   "
            f"50,000 steps = {50000*per_step/60:5.1f} min   "
            f"(${50000*per_step/3600*0.72:.2f} at $0.72/h)"
        )
        del model, optimiser
        torch.cuda.empty_cache()


def cmd_sweep(args) -> None:
    """The six pre-registered configurations, identical token budget each."""
    train_w, valid_w, vocab = load()
    out = []
    chosen = enumerate(SWEEP) if args.only is None else [(args.only, SWEEP[args.only])]
    for index, spec in chosen:
        cfg = Config(
            device="cuda", max_steps=SWEEP_STEPS, eval_every=500,
            patience_evals=99, seed=args.seed,
            run_dir=f"{args.out_dir}/sweep/cfg{index}",
            notes=f"pre-registered sweep {index}", **spec,
        )
        print(f"\n=== sweep {index}: {spec} ===", flush=True)
        summary = train(cfg, train_w, valid_w, vocab)
        out.append({"index": index, **spec, "n_params": summary["n_params"],
                    "best": summary["best"]})
        print(f"  best valid {summary['best']['bits_per_token']:.4f} "
              f"at step {summary['best']['step']}", flush=True)
    out.sort(key=lambda r: r["best"]["bits_per_token"])
    results = Path(args.out_dir) / "sweep_results.json"
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps(out, indent=1))
    print("\nSWEEP RESULTS (validation only)")
    for row in out:
        print(f"  lr={row['lr']:<7} {row['d_model']}x{row['n_layers']} "
              f"{row['n_params']:,}p -> {row['best']['bits_per_token']:.4f}")
    print(f"\nSELECTED: {out[0]}")


def cmd_full(args) -> None:
    """The selected configuration to 50,000 steps with early stopping."""
    train_w, valid_w, vocab = load()
    cfg = Config(
        device="cuda", max_steps=args.steps, eval_every=500,
        patience_evals=8, seed=args.seed, lr=args.lr, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
        run_dir=f"{args.out_dir}/full",
        notes="selected configuration, full run",
    )
    print(json.dumps(cfg.to_dict(), indent=1), flush=True)
    summary = train(cfg, train_w, valid_w, vocab)
    print(f"\nBEST valid {summary['best']['bits_per_token']:.4f} "
          f"at step {summary['best']['step']}")
    print(f"checkpoint sha256 {summary['best'].get('checkpoint_sha256')}")
    print(f"\nFrozen baseline (validation): 3.3839 bits/token")
    print(f"Level 1 validation reference : 3.2147 bits/token")
    print("Level 1 judgment is made on the SEALED TEST SET against the "
          "baseline re-measured there, not against this reference.")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("throughput").set_defaults(func=cmd_throughput)
    s = sub.add_parser("sweep"); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out-dir", dest="out_dir", default="/workspace/snes_lm")
    s.add_argument("--only", type=int, default=None,
                   help="run a single sweep index, for concurrent execution")
    s.set_defaults(func=cmd_sweep)
    f = sub.add_parser("full")
    f.add_argument("--lr", type=float, required=True)
    f.add_argument("--d-model", dest="d_model", type=int, required=True)
    f.add_argument("--n-layers", dest="n_layers", type=int, required=True)
    f.add_argument("--n-heads", dest="n_heads", type=int, default=8)
    f.add_argument("--steps", type=int, default=50_000)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--out-dir", dest="out_dir", default="/workspace/snes_lm")
    f.set_defaults(func=cmd_full)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
