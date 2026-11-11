#!/usr/bin/env python3
"""Materialise real G3 training windows for the hardware benchmark.

**Real windows, not randomly shaped tensors.** The benchmark must see actual
snapshots, actual padding and the actual loss mask, because those are what
determine the logits shape and the scored-target count. A random tensor of the
same shape would measure the hardware but not this training job.

Three arrays per split:

- `tokens`  int16 [n_windows, context]  -- padded with PAD
- `mask`    bool  [n_windows, context]  -- scored targets only: padding
  excluded, and SNAPSHOT POSITIONS excluded because a snapshot is given
  conditioning state rather than model output
- `lengths` int32 [n_windows]           -- real token count per window

`position` ids are implicit 0..context-1 and are NOT stored: the whole point of
the group-aligned window convention is that training and inference index
positions identically, so a stored offset would be a second source of truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rse.snes_lm.release import Release  # noqa: E402
from rse.snes_v2 import codec as cx  # noqa: E402
from rse.snes_v2 import delta as dx  # noqa: E402
from rse.snes_v2 import grammar as gx  # noqa: E402

GRID_HZ = 4000.0
CONTEXT = 1024
#: One past the vocabulary, reserved for padding only.
PAD_OFFSET = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="~/snesmdb-releases/snesmdb-50game-rc1")
    parser.add_argument("--splits", default="train,valid")
    parser.add_argument("--out-dir", default=str(Path.home() /
                        "workplace/v2_artifacts/windows"))
    args = parser.parse_args()

    raw = json.loads((Path.home() /
                      "workplace/v2_artifacts/delta_vocab_k512.json").read_text())
    codec = dx.DeltaCodec(direct={int(k): v for k, v in raw["direct"].items()},
                          n_observed=raw["n_distinct_train"])
    vocab = gx.Vocab(delta=codec)
    pad_id = vocab.size            # PAD sits just past the real vocabulary
    release = Release(args.release)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"vocab_size": vocab.size, "pad_id": pad_id, "context": CONTEXT,
               "grid_hz": GRID_HZ, "splits": {}}
    for split in args.splits.split(","):
        rows, masks, lengths = [], [], []
        tracks = release.tracks(split=split)
        for index, track in enumerate(tracks):
            events = release.events(track)
            if not events:
                continue
            groups, _ = cx.encode(events, grid_hz=GRID_HZ)
            for window in gx.build_windows(groups, vocab, context=CONTEXT):
                n = window.n_tokens
                row = np.full(CONTEXT, pad_id, dtype=np.int32)
                row[:n] = window.tokens
                mask = np.zeros(CONTEXT, dtype=bool)
                # Scored targets: positions predicting a real next token,
                # excluding snapshot positions (conditioning, not output).
                first = 1 + window.n_snapshot_tokens
                mask[first:n] = True
                rows.append(row.astype(np.int16))
                masks.append(mask)
                lengths.append(n)
            if (index + 1) % 400 == 0:
                print(f"    {split} {index + 1}/{len(tracks)}", flush=True)

        tokens = np.stack(rows)
        mask = np.stack(masks)
        path = out_dir / f"{split}.npz"
        np.savez_compressed(path, tokens=tokens, mask=mask,
                            lengths=np.array(lengths, dtype=np.int32))
        summary["splits"][split] = {
            "n_windows": int(tokens.shape[0]),
            "scored_targets": int(mask.sum()),
            "padding_positions": int((tokens == pad_id).sum()),
            "bytes": path.stat().st_size,
            "file": path.name,
        }
        s = summary["splits"][split]
        print(f"  {split}: {s['n_windows']:,} windows, "
              f"{s['scored_targets']:,} scored targets, "
              f"{s['bytes'] / 1e6:.1f} MB", flush=True)

    (out_dir / "MANIFEST.json").write_text(json.dumps(summary, indent=1))
    print(f"\n  written to {out_dir}")


if __name__ == "__main__":
    main()
