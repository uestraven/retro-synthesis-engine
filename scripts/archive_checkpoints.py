"""Verify and manifest native RSE checkpoints retrieved from Runpod.

The trained checkpoints are the only artifacts in this project with no
reproducible source: the corpus can be rebuilt, the representation is frozen
and versioned, every experiment script is in git, but a 100,000-step training
run cannot be recreated without paying for the GPU again. So they get a real
archival check rather than a copy and a hope.

What "verified" means here, in order of strength:

1. **The bytes survived the transfer.** SHA-256 of each file against the
   digests computed on the pod before it was torn down. A hash computed only
   after the copy proves the copy is self-consistent, not that it matches the
   original, so `--expect` takes the pod-side digest file and anything missing
   from it is reported rather than silently passed.
2. **The file is a checkpoint this build can read.** `load_checkpoint`
   enforces `rse-checkpoint/1.x` and reconstructs the model from the payload's
   own `model_config`, so a test-load catches a truncated or mismatched file
   that a hash alone would accept as "intact but useless".
3. **The weights are finite and the model runs.** A forward pass on one dummy
   batch, because a state dict can load cleanly and still hold NaNs.

The vocabulary travels *inside* the payload (`load_checkpoint` rebuilds the
tokenizer from `payload["vocabulary"]`), so a checkpoint is self-describing
and there is no separate vocab file to lose. The manifest records the vocab
size and hash anyway, so a later mismatch against a representation id is
detectable.

Nothing here deletes or moves anything on Runpod. Teardown is a separate,
explicit step that should happen only after this script reports every file
verified, because a pod's container disk does not survive termination.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Importable when run as a file from a checkout where the package is not
# installed.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import hashlib
import json
from typing import Any

#: Read in chunks: a checkpoint is tens of megabytes and there is no reason to
#: hold one in memory to digest it.
CHUNK = 1 << 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def parse_expected(path: Path) -> dict[str, str]:
    """Read a `sha256sum` output file into {basename: digest}.

    Keyed by basename rather than full path, because the pod-side paths will
    not match the local ones and matching on them would fail every file for
    the wrong reason.
    """
    expected: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        digest, name = parts[0], parts[-1].lstrip("*")
        expected[Path(name).name] = digest.lower()
    return expected


def inspect(path: Path, *, test_load: bool) -> dict[str, Any]:
    """Hash a checkpoint, then optionally load it and run one forward pass."""
    record: dict[str, Any] = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "loaded": None,
        "forward_ok": None,
        "error": None,
    }
    if not test_load:
        return record

    try:
        import torch

        from rse.training.train import load_checkpoint

        model, tokenizer, payload = load_checkpoint(path, device="cpu")
        config = payload.get("model_config", {})
        vocab = payload.get("vocabulary", {})
        tokens = vocab.get("tokens") if isinstance(vocab, dict) else None
        record.update(
            {
                "loaded": True,
                "checkpoint_version": payload.get("checkpoint_version"),
                "step": payload.get("step"),
                "best_valid_loss": payload.get("best_valid_loss")
                or payload.get("best_loss"),
                "model_config": config,
                "n_parameters": sum(p.numel() for p in model.parameters()),
                "vocab_size": len(tokens) if tokens is not None else None,
                "vocab_sha256": hashlib.sha256(
                    json.dumps(vocab, sort_keys=True).encode()
                ).hexdigest()
                if vocab
                else None,
                "representation_id": payload.get("representation_id"),
                "git_commit": payload.get("git_commit") or payload.get("commit"),
                "has_optimizer_state": payload.get("optimizer") is not None,
                "max_seq_len": config.get("max_seq_len"),
                "tokenizer_codec": getattr(tokenizer.config, "codec", None),
            }
        )

        # A state dict can load cleanly and still be full of NaNs, so check
        # the parameters and then actually push a batch through.
        nonfinite = [
            name
            for name, tensor in model.named_parameters()
            if not torch.isfinite(tensor).all()
        ]
        record["nonfinite_parameters"] = nonfinite
        # `ModelConfig` calls it `max_seq_len`; a longer dummy batch than the
        # positional embedding allows would fail for the wrong reason.
        seq = min(32, int(config.get("max_seq_len", 32) or 32))
        vocab_n = record["vocab_size"] or 2
        with torch.no_grad():
            dummy = torch.randint(0, max(1, vocab_n - 1), (1, seq), dtype=torch.long)
            logits = model(dummy)
        record["forward_ok"] = bool(torch.isfinite(logits).all()) and not nonfinite
        record["logits_shape"] = list(logits.shape)
    except Exception as error:  # noqa: BLE001 - the error IS the result here
        record["loaded"] = False
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive", type=Path, help="directory holding the retrieved checkpoints"
    )
    parser.add_argument(
        "--expect",
        type=Path,
        default=None,
        help="sha256sum output produced ON THE POD, before teardown",
    )
    parser.add_argument(
        "--no-test-load",
        action="store_true",
        help="hash only; skip loading and the forward pass",
    )
    parser.add_argument("--glob", default="*.pt")
    parser.add_argument("--out", type=Path, default=Path("checkpoint_manifest.json"))
    args = parser.parse_args()

    files = sorted(args.archive.rglob(args.glob))
    if not files:
        print(f"error: no {args.glob} under {args.archive}", file=sys.stderr)
        return 2

    expected = parse_expected(args.expect) if args.expect else {}
    if args.expect and not expected:
        print(f"error: {args.expect} held no digests", file=sys.stderr)
        return 2

    records = []
    for path in files:
        record = inspect(path, test_load=not args.no_test_load)
        want = expected.get(path.name)
        record["expected_sha256"] = want
        record["hash_verified"] = (want == record["sha256"]) if want else None
        records.append(record)
        status = (
            "no digest"
            if record["hash_verified"] is None
            else ("hash ok" if record["hash_verified"] else "HASH MISMATCH")
        )
        load = (
            "skipped"
            if record["loaded"] is None
            else ("load ok" if record["loaded"] else "LOAD FAILED")
        )
        forward = (
            ""
            if record["forward_ok"] is None
            else (" forward ok" if record["forward_ok"] else " FORWARD BAD")
        )
        print(
            f"  {path.name:<34}{record['bytes'] / 1e6:>8.1f} MB  "
            f"{status:<14}{load:<12}{forward}"
        )
        if record["error"]:
            print(f"      {record['error']}")

    # Files the pod listed that never arrived are the dangerous case: the
    # transfer looks clean because what is missing cannot fail a check.
    arrived = {record["file"] for record in records}
    missing = sorted(set(expected) - arrived) if expected else []

    mismatched = [r["file"] for r in records if r["hash_verified"] is False]
    unverified = [r["file"] for r in records if r["hash_verified"] is None]
    failed_load = [r["file"] for r in records if r["loaded"] is False]
    bad_forward = [r["file"] for r in records if r["forward_ok"] is False]

    payload = {
        "archive": str(args.archive.resolve()),
        "expected_from": str(args.expect) if args.expect else None,
        "n_files": len(records),
        "n_expected": len(expected) or None,
        "missing_from_archive": missing,
        "hash_mismatched": mismatched,
        "hash_unverified": unverified,
        "load_failed": failed_load,
        "forward_failed": bad_forward,
        "fully_verified": not (missing or mismatched or failed_load or bad_forward)
        and not unverified,
        "checkpoints": records,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\n{len(records)} files, {sum(r['bytes'] for r in records) / 1e6:.1f} MB")
    if missing:
        print(f"MISSING from the archive ({len(missing)}): {missing}")
    if mismatched:
        print(f"HASH MISMATCH ({len(mismatched)}): {mismatched}")
    if failed_load:
        print(f"LOAD FAILED ({len(failed_load)}): {failed_load}")
    if bad_forward:
        print(f"FORWARD FAILED ({len(bad_forward)}): {bad_forward}")
    if unverified:
        print(f"no pod-side digest to check against ({len(unverified)}): {unverified}")
    print(f"\nwrote {args.out}")
    if payload["fully_verified"]:
        print("VERIFIED: safe to tear down the pod.")
        return 0
    print("NOT fully verified: do not terminate the pod yet.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
