#!/usr/bin/env python3
"""Build and validate the sampling-selection deployment bundle.

The pod gets no Brazil release, no git remote and no network path back here,
so everything the frozen procedure needs travels in one tarball with its
checksums. `--validate` runs against an extracted bundle and is the gate the
pod run depends on: it is run here, on this desktop, against the same tarball
that will be transferred.

What validation proves, in order:

1. every declared file is present and its SHA-256 matches the manifest;
2. the checkpoint loads and is the locked one -- step 5,400 and 5,646,336
   parameters, asserted rather than trusted;
3. a **scripted performance** -- a token stream written by hand, not sampled
   -- loads, parses, reconstructs notes, and scores under the frozen
   statistics, so the measurement path is exercised without a GPU;
4. the 40 predetermined output identifiers are generated before any excerpt
   exists, so a missing output is detectable rather than invisible;
5. the selection module declares failure, and the driver exits nonzero, when
   no candidate is eligible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ARTIFACTS = Path.home() / "workplace/v2_artifacts"

#: (source, path inside the bundle). The source package is shipped whole
#: rather than by module, because the frozen procedure's §7 guarantee is that
#: generation uses the SAME code path as training -- a trimmed copy would
#: quietly break that.
PAYLOAD: tuple[tuple[Path, str], ...] = (
    (ROOT / "tmp/run/best.pt", "best.pt"),
    (ARTIFACTS / "delta_vocab_k512.json", "delta_vocab_k512.json"),
    (ARTIFACTS / "reference_stats.json", "reference_stats.json"),
    (ARTIFACTS / "windows/MANIFEST.json", "windows/MANIFEST.json"),
    (ARTIFACTS / "windows/train.npz", "windows/train.npz"),
    (ARTIFACTS / "windows/valid.npz", "windows/valid.npz"),
    (ARTIFACTS / "REPRODUCIBILITY_MANIFEST.json",
     "REPRODUCIBILITY_MANIFEST.json"),
    (ROOT / "scripts/snes_v2_select_sampling.py",
     "scripts/snes_v2_select_sampling.py"),
    # Shipped so validation exercises the driver's exit path against the
    # bundled bytes rather than the working tree.
    (ROOT / "tests/test_snes_v2_select_driver.py",
     "tests/test_snes_v2_select_driver.py"),
    (ROOT / "tests/test_snes_v2_selection.py",
     "tests/test_snes_v2_selection.py"),
    (ROOT / "tests/test_snes_v2_statistics.py",
     "tests/test_snes_v2_statistics.py"),
    # The property whose absence hid the termination defect.
    (ROOT / "tests/test_snes_v2_generate_parses.py",
     "tests/test_snes_v2_generate_parses.py"),
)

#: Shipped whole: the G3 codec, delta vocabulary, grammar, generator,
#: statistics, selection, accounting, and the decoder definition.
PACKAGE_DIRS = ("rse/snes_v2", "rse/snes_lm")

EXPECTED_STEP = 5400
EXPECTED_PARAMETERS = 5_646_336


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def revision_ids() -> dict[str, str]:
    out: dict[str, str] = {}
    for name, path in (("retro-synthesis-engine", ROOT),
                       ("snesmdb", Path.home() / "workplace/snesmdb")):
        try:
            out[name] = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(path), "status", "--porcelain"],
                capture_output=True, text=True, check=True).stdout.strip()
            out[f"{name}_dirty"] = bool(dirty)
        except Exception as exc:                   # noqa: BLE001
            out[name] = f"unavailable: {exc}"
    return out


def build(out_path: Path) -> dict:
    files: list[tuple[Path, str]] = list(PAYLOAD)
    for directory in PACKAGE_DIRS:
        for source in sorted((ROOT / directory).rglob("*.py")):
            files.append((source, str(source.relative_to(ROOT))))

    missing = [str(s) for s, _ in files if not s.exists()]
    if missing:
        raise SystemExit("missing payload: " + ", ".join(missing))

    manifest = {
        "schema": "snes_v2.selection_bundle.v1",
        "purpose": "frozen sampling selection, one GPU pod, no network path "
                   "back to the authoring host",
        "revisions": revision_ids(),
        "expected_checkpoint_step": EXPECTED_STEP,
        "expected_parameters": EXPECTED_PARAMETERS,
        "entrypoint": "python3 scripts/snes_v2_select_sampling.py --arm model "
                      "--checkpoint best.pt --windows windows "
                      "--delta-vocab delta_vocab_k512.json "
                      "--reference reference_stats.json --out out "
                      "--device cuda",
        "files": [{"path": arc, "sha256": sha256(src),
                   "bytes": src.stat().st_size} for src, arc in files],
    }
    manifest["n_files"] = len(manifest["files"])
    manifest["total_bytes"] = sum(f["bytes"] for f in manifest["files"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    staged = out_path.with_suffix(".manifest.json")
    staged.write_text(json.dumps(manifest, indent=1))
    def _neutral(info: tarfile.TarInfo) -> tarfile.TarInfo:
        """Own the archive to root:root with no names.

        The authoring uid does not exist on the pod, so restoring ownership
        fails there and `tar` exits nonzero even though every file extracted
        correctly -- a cosmetic failure that reads exactly like a real one.
        """
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    with tarfile.open(out_path, "w:gz") as tar:
        for src, arc in files:
            tar.add(src, arcname=arc, filter=_neutral)
        tar.add(staged, arcname="BUNDLE_MANIFEST.json", filter=_neutral)
    print(f"  {manifest['n_files']} files, "
          f"{manifest['total_bytes'] / 1e6:.1f} MB payload -> {out_path} "
          f"({out_path.stat().st_size / 1e6:.1f} MB compressed)")
    return manifest


# --------------------------------------------------------------------------
# validation, run against an EXTRACTED bundle
# --------------------------------------------------------------------------

def validate(bundle_dir: Path, *, device: str = "cpu") -> None:
    manifest = json.loads((bundle_dir / "BUNDLE_MANIFEST.json").read_text())
    failures: list[str] = []

    # 1. checksums
    for entry in manifest["files"]:
        path = bundle_dir / entry["path"]
        if not path.exists():
            failures.append(f"missing {entry['path']}")
            continue
        if sha256(path) != entry["sha256"]:
            failures.append(f"checksum mismatch {entry['path']}")
    print(f"  checksums: {len(manifest['files'])} files, "
          f"{len(failures)} problems")
    if failures:
        raise SystemExit("  FAIL\n    " + "\n    ".join(failures))

    sys.path.insert(0, str(bundle_dir))
    import torch

    from rse.snes_lm.train import Config, Decoder
    from rse.snes_v2 import delta as dx
    from rse.snes_v2 import grammar as gx
    from rse.snes_v2 import selection as sel
    from rse.snes_v2 import statistics as sx

    # 2. the locked checkpoint
    state = torch.load(bundle_dir / "best.pt", map_location="cpu",
                       weights_only=False)
    cfg = Config(**{**state["config"], "device": device})
    model = Decoder(state["vocab_size"], cfg)
    model.load_state_dict(state["model"])
    n_parameters = sum(p.numel() for p in model.parameters())
    assert state["step"] == EXPECTED_STEP, \
        f"checkpoint step {state['step']} != {EXPECTED_STEP}"
    assert n_parameters == EXPECTED_PARAMETERS, \
        f"parameters {n_parameters:,} != {EXPECTED_PARAMETERS:,}"
    print(f"  checkpoint: step {state['step']}, {n_parameters:,} parameters, "
          f"vocab {state['vocab_size']:,}")

    # 3. scripted performance: load, decode, score
    raw = json.loads((bundle_dir / "delta_vocab_k512.json").read_text())
    vocab = gx.Vocab(delta=dx.DeltaCodec(
        direct={int(k): v for k, v in raw["direct"].items()},
        n_observed=raw["n_distinct_train"]))
    assert state["vocab_size"] == vocab.size + 1, \
        f"checkpoint vocab {state['vocab_size']} != G3 {vocab.size} + PAD"
    from rse.snes_v2.scripted import reference_performance
    tokens = reference_performance(vocab)
    parsed, _ = gx.parse(tokens, vocab)
    notes = gx.notes_from_groups(parsed)
    statistics = sx.excerpt_statistics(notes, grid_hz=4000.0)
    assert statistics is not None, "scripted performance scored as undefined"
    reference = json.loads(
        (bundle_dir / "reference_stats.json").read_text())["reference"]
    objective = sx.objective(sx.aggregate([statistics]), reference)
    assert len(notes) == 2, f"scripted performance rebuilt {len(notes)} notes"
    assert statistics["n_notes"] == 2
    print(f"  scripted performance: {len(tokens)} tokens -> {len(notes)} notes, "
          f"{statistics['attacks_per_sec']:.3f} attacks/s, "
          f"objective {objective:.4f}")

    # one real forward pass, so the model is exercised, not just loaded
    model.eval().to(device)
    with torch.no_grad():
        row = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = model(row)
    assert logits.shape == (1, len(tokens), state["vocab_size"]), logits.shape
    print(f"  forward pass on {device}: logits {tuple(logits.shape)}")

    # 4. the 40 predetermined identifiers
    sys.path.insert(0, str(bundle_dir / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "driver", bundle_dir / "scripts/snes_v2_select_sampling.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    identifiers = [driver.identifier("model", t, p, s)
                   for t in sel.TEMPERATURES for p in sel.TOP_P_VALUES
                   for s in sel.DEVELOPMENT_SEEDS]
    assert len(identifiers) == sel.EXPECTED_ATTEMPTED == 40
    assert len(set(identifiers)) == 40, "identifier collision"
    print(f"  identifiers: {len(identifiers)} predetermined and distinct, "
          f"e.g. {identifiers[0]}")

    # 5. an entirely ineligible grid is a declared failure
    dead = [sel.Candidate(temperature=t, top_p=p, outcomes=[
        sel.Outcome(seed=s, outcome="invalid", statistics=None)
        for s in sel.DEVELOPMENT_SEEDS])
        for t in sel.TEMPERATURES for p in sel.TOP_P_VALUES]
    result = sel.select(dead, reference)
    assert result.successful is False and result.selected is None
    assert result.n_attempted_outcomes == 40
    assert result.n_eligible_candidates == 0
    print("  Amendment A1: an entirely ineligible grid is declared "
          "unsuccessful, 40 outcomes still reported")

    # 6. the DRIVER's own exit path, run against the EXTRACTED copy.
    #
    # These tests drive `main()` with a stubbed generator and assert the exit
    # status and the manifest, which is the one thing a library assertion
    # cannot reach. They are shipped in the bundle so this runs against the
    # same bytes that will be transferred, not against the working tree.
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q",
         str(bundle_dir / "tests/test_snes_v2_select_driver.py"),
         str(bundle_dir / "tests/test_snes_v2_selection.py"),
         str(bundle_dir / "tests/test_snes_v2_statistics.py"),
         str(bundle_dir / "tests/test_snes_v2_generate_parses.py")],
        capture_output=True, text=True, cwd=bundle_dir)
    tail = (tests.stdout + tests.stderr).strip().splitlines()[-3:]
    if tests.returncode != 0:
        raise SystemExit("  driver tests FAILED against the extracted "
                         "bundle:\n" + tests.stdout + tests.stderr)
    print(f"  driver tests against the extracted bundle: "
          f"{' | '.join(line.strip() for line in tail if line.strip())}")
    print("\n  BUNDLE VALIDATION: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "tmp/g3select.tar.gz"))
    parser.add_argument("--validate", default="",
                        help="validate an extracted bundle at this path")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.validate:
        validate(Path(args.validate), device=args.device)
    else:
        build(Path(args.out))


if __name__ == "__main__":
    main()
