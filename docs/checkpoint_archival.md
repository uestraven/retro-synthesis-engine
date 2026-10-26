# Archiving the native checkpoints off Runpod

The trained checkpoints are the only artifacts in this project with no
reproducible source. The corpus can be rebuilt from NES-MDB, the
representation is frozen and versioned (`rse-repr/1.0.0-d4k4-tempo1.01`), and
every experiment script is in git — but a 100,000-step run cannot be recreated
without paying for the GPU again.

## Current state (read from the Runpod API, 2026-10-26)

| Fact | Value |
|---|---|
| Pods | `qx0wqvu5qmcjyu` (`soft_peach_fox`), `qq7kn1nuxt8z08` (`soft_peach_fox-migration`) |
| Status of both | `EXITED` — GPU billing already stopped |
| GPU rate if restarted | $0.72/hr each |
| Network volume | `kcc2582kg8` (`thoughtful_cyan_penguin`), 50 GB `STANDARD`, `US-KS-2` |
| Volume mount in both pods | `/workspace` |
| Registered SSH keys on the account | **none** |
| Today's spend | GPU $0.00, pod disk $0.00, storage $0.0097 |
| Storage run rate | ~$0.117/day ≈ **$3.50/month** |

So there is no idle GPU to shut down; the only live charge is the volume.

## The one real risk

**It is not yet known whether the checkpoints sit on the network volume or on a
pod's container disk.** The documented training command writes to
`checkpoints/nesmdb.pt` *relative to the cloned repo*
(`docs/gpu-training.md`), so the answer depends on where the repo was cloned:

- cloned under `/workspace/...` → on the network volume, survives pod termination
- cloned under `/root/...` or similar → on the container disk, **destroyed by termination**

A terminated pod's container disk is unrecoverable. Nothing should be
terminated until a shell has confirmed where the files are.

## Retrieval

The Runpod MCP tools manage infrastructure only — no file transfer, no shell.
With no SSH key registered on the account and no `runpodctl` available, the
pod-side step has to be driven by hand. Jupyter is already enabled on both
pods (`8888/http`, password in the pod's `JUPYTER_PASSWORD`), which needs no
SSH key.

### 1. Start one pod and open a terminal

Start `qx0wqvu5qmcjyu` from the console (or `pod-action start`), then open
Jupyter → New → Terminal. Billing resumes at $0.72/hr while it runs, so the
whole retrieval should take minutes.

### 2. Find out what exists and where

```bash
ls -la /workspace
find / -name "*.pt" -size +1M -not -path "*/site-packages/*" 2>/dev/null \
  | xargs -r ls -la
df -h /workspace /
```

The `find` is the important line: it answers the volume-vs-container-disk
question for every checkpoint at once.

### 3. Collect, digest, and compress

```bash
mkdir -p /workspace/archive
# adjust the copy sources to whatever step 2 actually found
cp -av <each .pt and .report.json found> /workspace/archive/

cd /workspace/archive
sha256sum *.pt *.json > POD_SHA256.txt    # computed BEFORE any transfer
cat POD_SHA256.txt
du -sh .
tar czf /workspace/rse_checkpoints.tar.gz -C /workspace archive
sha256sum /workspace/rse_checkpoints.tar.gz
```

`POD_SHA256.txt` is the point of the exercise: a hash computed after the copy
only proves the copy is self-consistent. The pod-side digest is what makes a
transfer verifiable.

### 4. Get it off the pod

Either download `rse_checkpoints.tar.gz` through the Jupyter file browser
straight to Windows, or use `runpodctl send` on the pod and
`runpodctl receive <code>` on the receiving machine. ~70 MB either way.

### 5. Verify before trusting it

```bash
python scripts/archive_checkpoints.py <archive-dir> \
  --expect <archive-dir>/POD_SHA256.txt \
  --out checkpoint_manifest.json
```

The script checks three independent things, weakest to strongest:

1. **SHA-256 against the pod-side digests.** Files listed in `POD_SHA256.txt`
   that never arrived are reported too — a missing file cannot fail a check,
   which is what makes it dangerous.
2. **`load_checkpoint` accepts it**, which enforces `rse-checkpoint/1.x` and
   rebuilds the model from the payload's own `model_config`.
3. **The weights are finite and a forward pass runs.**

All three matter independently. A self-test with one byte flipped in the
middle of a checkpoint still reported `load ok / forward ok` — the flip landed
in tensor data rather than structure, so torch accepted it without complaint.
Load-testing alone would have passed a corrupted file as healthy.

The script exits non-zero and prints `NOT fully verified: do not terminate the
pod yet` unless every file is accounted for. Treat that literally.

The vocabulary travels *inside* each payload (`load_checkpoint` rebuilds the
tokenizer from `payload["vocabulary"]`), so there is no separate vocab file to
lose. The manifest records the vocab size and a hash of it anyway, so a later
mismatch against a representation id is detectable.

## Teardown, in order

1. `archive_checkpoints.py` exits 0 with `fully_verified: true`.
2. A second copy exists somewhere independent of both Runpod and the first copy.
3. Confirm the checkpoints were on `/workspace`, or that any container-disk
   copies have been retrieved.
4. Stop the pod, then terminate it.
5. Decide on the volume separately. At $3.50/month it is cheap insurance while
   the archive is young; it is also the last Runpod-side copy, so it should
   outlive the pods rather than go with them. Delete it only once the off-site
   copies are verified and no further training is planned.
