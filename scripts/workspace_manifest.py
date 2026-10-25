"""Inventory and verify a Runpod network volume.

Run on the pod. Walks the volume, records size and sha256 for every artifact
worth keeping, and reports what is missing against the expected set.

Why this exists: `/workspace` survives pod termination but nothing outside it
does, and this project has already lost a foundation checkpoint that sat at
`/root/ckpt_foundation.pt`. A manifest makes the difference between "the
volume has our work" and "we believe the volume has our work".

It also classifies each artifact by whether it can be regenerated, because
that decides what has to be copied off Runpod. A cache rebuilds deterministically
from the dataset; a trained checkpoint does not exist anywhere else.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Importable when run as a file from a checkout where the package is not
# installed, which is the normal state of a fresh GPU host.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import hashlib
import json
from typing import Any

#: Glob patterns worth recording, with whether the artifact is reproducible.
#: `irreplaceable` entries must be copied off Runpod; the rest can be rebuilt.
PATTERNS: tuple[tuple[str, str], ...] = (
    ("*.pt", "irreplaceable"),
    ("*.report.json", "irreplaceable"),
    ("t_*.log", "irreplaceable"),
    ("*.log", "reproducible"),
    ("caches/*.jsonl", "reproducible"),
    ("caches/*.json", "reproducible"),
    ("battery_*.json", "reproducible"),
    ("native_vs_fermata.json", "reproducible"),
    ("*.sha256", "irreplaceable"),
    ("listen/*.wav", "reproducible"),
    ("battery_audio*/*.wav", "reproducible"),
    ("native_d4_audio/*.wav", "reproducible"),
)

#: Artifacts this project expects to find. Absence is reported, not assumed
#: harmless: a missing checkpoint is the failure mode that already happened.
EXPECTED = (
    "ckpt_foundation_d4_100k.pt",
    "ckpt_foundation_d4_150k.pt",
    "caches/d4_full.jsonl",
    "caches/d4_full.jsonl.contract.json",
)


def digest(path: Path, *, chunk: int = 1 << 20) -> str:
    out = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            out.update(block)
    return out.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace"))
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="skip digests, which are slow on multi-gigabyte files",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root: Path = args.root
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    seen: dict[Path, str] = {}
    for pattern, kind in PATTERNS:
        for path in sorted(root.glob(pattern)):
            if path.is_file() and path not in seen:
                seen[path] = kind

    rows: list[dict[str, Any]] = []
    for path, kind in sorted(seen.items()):
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "kind": kind,
                "sha256": None if args.no_hash else digest(path),
            }
        )

    missing = [name for name in EXPECTED if not (root / name).is_file()]
    irreplaceable = [r for r in rows if r["kind"] == "irreplaceable"]
    total = sum(r["bytes"] for r in rows)
    must_copy = sum(r["bytes"] for r in irreplaceable)

    payload = {
        "root": str(root),
        "n_files": len(rows),
        "total_bytes": total,
        "irreplaceable_bytes": must_copy,
        "missing_expected": missing,
        "files": rows,
    }
    out = args.out or (root / "WORKSPACE_MANIFEST.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"{root}: {len(rows)} files, {total / 2**20:.1f} MiB\n")
    print(f"{'kind':<16}{'MiB':>9}{'files':>7}")
    print("-" * 32)
    for kind in ("irreplaceable", "reproducible"):
        group = [r for r in rows if r["kind"] == kind]
        print(
            f"{kind:<16}{sum(r['bytes'] for r in group) / 2**20:>9.1f}{len(group):>7}"
        )
    print("\nirreplaceable -- copy these off Runpod:")
    for row in irreplaceable:
        print(f"  {row['bytes'] / 2**20:>9.1f} MiB  {row['path']}")
    if missing:
        print("\nMISSING expected artifacts:")
        for name in missing:
            print(f"  ! {name}")
    else:
        print("\nall expected artifacts present")
    print(f"\nwrote {out}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
