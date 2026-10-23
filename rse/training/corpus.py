"""Assembling a training corpus.

Two sources, one shape. An ingested dataset release is the real target; the
procedural generator is a stand-in that makes the whole training path runnable
and testable before NES-MDB is in hand.

Training on procedural output is a *pipeline* test, not a music result. If a
model can learn the rule-based generator's distribution, tokenization,
training, sampling and evaluation all work — and nothing more than that is
proven. Say so whenever reporting numbers from it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rse.data.pipeline import processed_dir
from rse.events.convert import composition_from_intake
from rse.events.schema import Composition
from rse.generation.plan import ENERGIES, MOODS, make_plan
from rse.generation.procedural import generate
from rse.hardware.profile import HardwareProfile
from rse.jsonio import read_json, read_jsonl
from rse.training.dataset import TRAIN, assign_splits

#: A corpus entry: the music, which split it belongs to, and its identity.
CorpusEntry = tuple[Composition, str, str]

PROCEDURAL = "procedural"
DATASET = "dataset"


@dataclass
class CorpusInfo:
    """Where a corpus came from, for the training report."""

    source: str
    n_compositions: int = 0
    n_skipped: int = 0
    per_split: dict[str, int] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)
    #: True when the corpus is synthetic, so reports can say so plainly.
    synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "n_compositions": self.n_compositions,
            "n_skipped": self.n_skipped,
            "per_split": dict(sorted(self.per_split.items())),
            "synthetic": self.synthetic,
            "detail": self.detail,
        }


def corpus_from_procedural(
    profile: HardwareProfile,
    *,
    n: int = 200,
    seed: int = 0,
    n_bars: int = 16,
    valid_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> tuple[list[CorpusEntry], CorpusInfo]:
    """Generate a synthetic corpus with the rule-based composer."""
    rng = random.Random(seed)
    moods = [m for m in MOODS]
    energies = [e for e in ENERGIES]

    pieces: list[tuple[Composition, str]] = []
    for index in range(n):
        plan = make_plan(
            seed=seed * 100_000 + index,
            mood=rng.choice(moods),
            energy=rng.choice(energies),
            n_bars=n_bars,
        )
        generated = generate(profile, plan)
        pieces.append((composition_from_intake(generated.record), generated.composition_id))

    splits = assign_splits(
        [identifier for _, identifier in pieces],
        seed=seed,
        valid_fraction=valid_fraction,
        test_fraction=test_fraction,
    )
    entries = [
        (composition, splits.get(identifier, TRAIN), identifier)
        for composition, identifier in pieces
    ]
    info = CorpusInfo(source=PROCEDURAL, n_compositions=len(entries), synthetic=True)
    info.detail = {"seed": seed, "n_bars": n_bars, "generator": "procedural/0.1.0"}
    for _, split, _ in entries:
        info.per_split[split] = info.per_split.get(split, 0) + 1
    return entries, info


def corpus_from_release(
    dataset_id: str,
    release_id: str,
    *,
    datasets_dir: Path | None = None,
    limit: int | None = None,
    exclude_duplicates: bool = True,
    require_compliant: bool = False,
    seed: int = 0,
) -> tuple[list[CorpusEntry], CorpusInfo]:
    """Load an ingested release as a corpus.

    Splits come from the release's own index where present — NES-MDB's are
    composer-disjoint, and replacing them with a random split would invite
    exactly the memorization the originality screening looks for. Only when the
    index has no splits are they assigned here, and by composition rather than
    by window.
    """
    out_dir = processed_dir(dataset_id, release_id, datasets_dir=datasets_dir)
    index_path = out_dir / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"no index at {index_path}; run `rse ingest --dataset {dataset_id} "
            f"--release {release_id}` first"
        )

    rows = list(read_jsonl(index_path))
    info = CorpusInfo(source=DATASET)
    info.detail = {
        "dataset_id": dataset_id,
        "release_id": release_id,
        "exclude_duplicates": exclude_duplicates,
        "require_compliant": require_compliant,
    }

    selected: list[dict[str, Any]] = []
    for row in rows:
        if exclude_duplicates and row.get("duplicate_of"):
            info.n_skipped += 1
            continue
        if require_compliant and not row.get("hardware_compliant", True):
            info.n_skipped += 1
            continue
        if not row.get("record_path"):
            info.n_skipped += 1
            continue
        selected.append(row)
    if limit:
        selected = selected[:limit]

    has_splits = any(row.get("split") for row in selected)
    fallback = (
        assign_splits([row["composition_id"] for row in selected], seed=seed)
        if not has_splits
        else {}
    )
    info.detail["splits_from_index"] = has_splits

    entries: list[CorpusEntry] = []
    for row in selected:
        record = read_json(out_dir / row["record_path"])
        composition = composition_from_intake(record)
        if not composition.sounding_events():
            info.n_skipped += 1
            continue
        split = row.get("split") or fallback.get(row["composition_id"], TRAIN)
        entries.append((composition, split, row["composition_id"]))
        info.per_split[split] = info.per_split.get(split, 0) + 1

    info.n_compositions = len(entries)
    return entries, info
