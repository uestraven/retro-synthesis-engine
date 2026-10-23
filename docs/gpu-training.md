# Training on a Rented GPU

Proposal §13.2 scenario C. The trainer is deliberately independent of the UI
and of any particular corpus, so moving a run to a GPU is a file copy and one
command — not a port.

## Why rent rather than grind

Measured on the 8-core Xeon this project is developed on, with a 4.8M-parameter
model at 16-bar windows and 1024-token sequences: **~11 s/step**. A properly
trained model wants several thousand steps, so about **10+ hours**. The same run
is well under an hour on a mid-range rented GPU, for roughly **$25**.

Two things make the difference beyond raw FLOPs: mixed precision turns on
automatically on CUDA (`use_amp = config.amp and device.type == "cuda"`), and a
GPU can hold a far larger batch, which is where most of the wall-clock saving
actually comes from.

**AMD note.** PyTorch's ROCm builds are Linux-only — there are no Windows ROCm
wheels — so an AMD card on Windows cannot run this as-is. The Windows path
would be `torch-directml`, which does not implement
`scaled_dot_product_attention` and would need a manual attention fallback. A
rented NVIDIA instance avoids the question entirely.

## Upload nothing: let the pod fetch it

The dataset URLs and checksums are committed in `configs/datasets/nesmdb.json`,
so a pod can acquire the corpus itself and no file has to move between
machines. This is the preferred path — it needs no SSH keys, no `runpodctl
send`, and it is reproducible from the repo alone.

```bash
rse fetch   --dataset nesmdb --release exprsco      # ~12 MB, checksum-verified
rse ingest  --dataset nesmdb --release exprsco      # a couple of minutes
rse windows --source dataset --dataset nesmdb --release exprsco \
            --window-bars 16 --out windows_16bar.jsonl
```

### If you would rather transfer the windows file

`rse windows` writes a single file that is the whole corpus, tokenized:

```bash
rse windows --source dataset --dataset nesmdb --release exprsco \
            --window-bars 16 --out windows_16bar.jsonl
```

That produces `windows_16bar.jsonl` plus a `.meta.json` recording the corpus,
the token statistics and the tokenizer config. The GPU box then needs only:

| File | Why |
|---|---|
| The repo | `git clone` it; no dataset required |
| `windows_16bar.jsonl` | The tokenized corpus |

The vocabulary does not need uploading: it is rebuilt deterministically from
the tokenizer config, and `load_checkpoint` refuses a checkpoint whose saved
vocabulary disagrees with the one rebuilt from its own config.

### A governance point worth making deliberately

The windows file is NES-MDB in tokenized form, and NES-MDB is registered
`research_only` — the nesmdb repository is MIT, but the compositions in it are
third-party copyrighted works. Uploading it to a rented machine is normal
research practice, but it is a conscious act of putting someone else's music on
a third-party host, so:

- Prefer a provider you would be comfortable naming in a provenance record.
- Delete the instance's copy when the run finishes, along with the instance.
- Keep the `use_class` on the resulting checkpoint: a model trained on this
  corpus inherits its restriction, and §5.5 wants a rights review before any
  commercial distribution of a model or its output.

`rse windows` prints this reminder itself when the source is a dataset.

## The run

Runpod's PyTorch images already have torch installed, so **do not create a
virtualenv** — it would hide the preinstalled torch and pull a multi-gigabyte
wheel instead. Install into the image's own environment, and install the base
package rather than `.[ml]`: torch is already present, and `miditok` and
`symusic` are declared for future work that no current code imports.

```bash
git clone https://github.com/uestraven/retro-synthesis-engine.git
cd retro-synthesis-engine
pip install -e .                      # mido + numpy; torch is already there
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

rse train \
  --windows windows_16bar.jsonl \
  --steps 8000 --batch-size 32 --seq-len 1024 \
  --layers 8 --heads 8 --width 512 \
  --learning-rate 3e-4 --seed 0 --device cuda \
  --out checkpoints/nesmdb.pt
```

**A pod with no network volume keeps nothing after termination.** Download the
checkpoint before you destroy the pod, or mount a network volume at
`/workspace` first so it outlives the pod.

Then bring back `checkpoints/nesmdb.pt` (and its `.report.json`) and everything
else runs locally without a GPU:

```bash
rse sample   --checkpoint checkpoints/nesmdb.pt --bars 16 --seed 7 --stems
rse evaluate --checkpoint checkpoints/nesmdb.pt --n 16
```

## Sizing

Proposal §6.1 suggests 5–20M parameters, chosen empirically. `--layers 8
--heads 8 --width 512` is about 25M, which a 16 GB card handles comfortably at
batch 32 and is a reasonable upper end for a ~12k-window corpus. Smaller is
safer if the validation loss stops improving early — that is the signal the
model is bigger than the data supports, and the trainer early-stops on it.

| Knob | Effect |
|---|---|
| `--batch-size` | Raise until VRAM is nearly full; the biggest single speed win |
| `--seq-len` | 1024 covers the 90th percentile of 16-bar windows; raising it costs quadratically |
| `--steps` | Watch `valid` in the printed curve; early stopping fires after 5 evals without improvement |
| `--layers` / `--width` | Capacity. Increase only if train and valid loss are still falling together |

Start with one short run (`--steps 200`) to confirm the device is actually CUDA
and the step time is what you expect, before paying for a long one. The report
records `"device"` so there is no ambiguity about what it ran on.

## What the checkpoint carries

Weights, the model config, the training config, the full tokenizer vocabulary,
and the loss history. That is enough for `rse sample` and `rse evaluate` to run
anywhere, and enough for the provenance record proposal §9.2 asks for.
