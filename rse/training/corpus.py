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
from pathlib import Path, PurePosixPath
from typing import Any

from rse.data.pipeline import processed_dir
from rse.events.convert import composition_from_intake
from rse.events.schema import Composition
from rse.generation.plan import ENERGIES, MOODS, make_plan
from rse.generation.procedural import generate
from rse.hardware.profile import HardwareProfile
from rse.jsonio import read_json, read_jsonl
from rse.training.dataset import TRAIN, VALID, assign_splits

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


def _matches_title(name: str, wanted: tuple[str, ...]) -> bool:
    """Whether a release filename belongs to one of the named games.

    NES-MDB names a track `NNN_GameTitle_TT_TrackName.ext` under a split
    directory, so the game is the second underscore-delimited field of the
    BASENAME -- matching the whole path would let the split directory and the
    track name produce false positives.

    Matched as a whole field rather than a substring, which matters more than
    it looks here: "FinalFantasy" is a prefix of "FinalFantasyII" and
    "FinalFantasyIII", and "DragonWarrior" of three more. A substring test
    asking for Final Fantasy alone would silently pull in all three.
    """
    stem = PurePosixPath(name).name
    parts = stem.split("_")
    if len(parts) < 2:
        return False
    return parts[1].lower() in wanted


def _stratified_holdout(
    rows: list[dict[str, Any]], fraction: float, *, seed: int
) -> set[str]:
    """Composition ids to move into the validation split, per game.

    Stratified so every game contributes proportionally -- a flat random
    sample of 15% across seven games of 21 to 59 tracks can easily take none
    from the smallest and skew the signal toward the largest.
    """
    if fraction <= 0:
        return set()
    by_title: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        stem = PurePosixPath(row["name"]).name
        title = stem.split("_")[1] if "_" in stem else "unknown"
        by_title.setdefault(title, []).append(row)

    held: set[str] = set()
    for title in sorted(by_title):
        group = sorted(by_title[title], key=lambda r: r["composition_id"])
        # At least one per game, so no game is unrepresented in the signal.
        want = max(1, round(len(group) * fraction))
        random.Random(f"{seed}:{title}").shuffle(group)
        held.update(row["composition_id"] for row in group[:want])
    return held


def corpus_from_release(
    dataset_id: str,
    release_id: str,
    *,
    datasets_dir: Path | None = None,
    limit: int | None = None,
    exclude_duplicates: bool = True,
    require_compliant: bool = False,
    splits: tuple[str, ...] | None = None,
    titles: tuple[str, ...] | None = None,
    holdout_fraction: float = 0.0,
    seed: int = 0,
) -> tuple[list[CorpusEntry], CorpusInfo]:
    """Load an ingested release as a corpus.

    Splits come from the release's own index where present — NES-MDB's are
    composer-disjoint, and replacing them with a random split would invite
    exactly the memorization the originality screening looks for. Only when the
    index has no splits are they assigned here, and by composition rather than
    by window.

    `titles` restricts the corpus to named games, matched case-insensitively
    against the composition id. That is how a single-genre corpus is built:
    the seven NES JRPGs are 235 of the release's 5,278 tracks.

    `holdout_fraction` carves an evaluation split from what survives the
    title filter, and exists because NES-MDB splits BY GAME. Every one of
    those seven JRPGs lands wholly in `train`, so a title-filtered corpus has
    no validation data at all and training would run blind. The split is
    stratified per title and seeded, so each game contributes proportionally
    and the same request gives the same split.

    Be clear about what that buys: a track-level holdout leaks the game. The
    model sees other tracks from the same game, so this measures "has it
    learned the idiom" and NOT "does it generalize to an unseen composer",
    which is what the release's own composer-disjoint splits measure. Use it
    as a training signal, not as evidence of generalization.
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
        "splits": list(splits) if splits else None,
        "titles": list(titles) if titles else None,
        "holdout_fraction": holdout_fraction,
    }
    wanted = tuple(t.lower() for t in titles) if titles else ()

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
        # Filter before the limit, or asking for 60 pieces of the test split
        # returns the handful that survive a corpus-wide truncation.
        if splits and row.get("split") not in splits:
            info.n_skipped += 1
            continue
        if wanted and not _matches_title(row["name"], wanted):
            info.n_skipped += 1
            continue
        selected.append(row)
    if limit and limit < len(selected):
        # Shuffle before truncating. The index is ordered by path, so `test/`
        # sorts before `train/` and a plain prefix would hand back a corpus
        # that was almost entirely the wrong split.
        random.Random(seed).shuffle(selected)
        selected = selected[:limit]
        selected.sort(key=lambda row: row["composition_id"])

    held = _stratified_holdout(selected, holdout_fraction, seed=seed)
    info.detail["n_held_out"] = len(held)

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
        if row["composition_id"] in held:
            split = VALID
        entries.append((composition, split, row["composition_id"]))
        info.per_split[split] = info.per_split.get(split, 0) + 1

    info.n_compositions = len(entries)
    return entries, info
