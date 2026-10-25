"""Arm assignment by batch-time substitution of one token.

The three Gate 3 arms share a single frozen cache. Every cached window already
carries `Motif_unspecified`, because `encode` always emits the conditioning
token, so arm A needs no substitution at all. Arms B and C replace that one
token id — nothing else about the sequence changes.

Why substitution rather than three caches: it makes "identical except for one
token" structurally true instead of a property to be checked across three
separately built files. A rebuild could differ for reasons nobody intended —
a tokenizer change between runs, a different sampling order — and the
comparison would silently stop being controlled.

The shuffle is computed once from a fixed seed and stored, not recomputed per
batch. Reshuffling per batch would mean arm C saw a different permutation
every epoch, which is a weaker control: it would average over permutations
rather than testing one fixed wrong labelling.

UNKNOWN is shuffled among the UNKNOWN rows only. Permuting it into the labelled
population would let arm C assign a confident label to a window with no
melodic evidence, which is a different intervention from "the labels are
wrong".
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rse.conditioning import UNKNOWN
from rse.tokenizers.vocab import MOTIF

ARMS = ("A", "B", "C")


@dataclass(frozen=True)
class ArmPlan:
    """Which motif token id each window gets, per arm."""

    #: (composition_id, start_bar) -> label, from the manifest.
    true_labels: dict[tuple[str, int], str]
    #: (composition_id, start_bar) -> label, permuted once under a fixed seed.
    shuffled_labels: dict[tuple[str, int], str]
    seed: int
    label_field: str

    def label_for(self, arm: str, key: tuple[str, int]) -> str:
        if arm == "A":
            return "unspecified"
        if arm == "B":
            return self.true_labels.get(key, "unspecified")
        if arm == "C":
            return self.shuffled_labels.get(key, "unspecified")
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

    def prevalence(self) -> dict[str, dict[str, int]]:
        """Label counts per arm, for confirming C preserves B's distribution."""
        return {
            "B": dict(Counter(self.true_labels.values())),
            "C": dict(Counter(self.shuffled_labels.values())),
        }


def load_plan(manifest: Path, *, seed: int = 0, label_field: str = "label") -> ArmPlan:
    """Read the window-label manifest and precompute the shuffled assignment.

    Shuffling is stratified by label eligibility: UNKNOWN rows are permuted
    only among themselves, and the labelled rows only among themselves. That
    preserves both the overall prevalence and the eligibility structure, so
    arm C differs from arm B in which window gets which label and in nothing
    else.
    """
    true_labels: dict[tuple[str, int], str] = {}
    with Path(manifest).open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row["piece_id"]), int(row["start_bar"]))
            true_labels[key] = str(row[label_field])

    groups: dict[bool, list[tuple[str, int]]] = defaultdict(list)
    for key, label in true_labels.items():
        groups[label == UNKNOWN].append(key)

    rng = random.Random(seed)
    shuffled: dict[tuple[str, int], str] = {}
    for is_unknown, keys in groups.items():
        ordered = sorted(keys)
        labels = [true_labels[k] for k in ordered]
        if not is_unknown:
            rng.shuffle(labels)
        shuffled.update(dict(zip(ordered, labels, strict=True)))
    return ArmPlan(
        true_labels=true_labels,
        shuffled_labels=shuffled,
        seed=seed,
        label_field=label_field,
    )


def motif_token_ids(vocabulary: Any) -> dict[str, int]:
    """Motif label -> token id, for substitution without string work per batch."""
    return {
        token[len(MOTIF) :]: vocabulary.id_of(token)
        for token in vocabulary.tokens
        if token.startswith(MOTIF)
    }


def substitute(
    ids: list[int],
    *,
    position: int,
    token_id: int,
) -> list[int]:
    """Replace the conditioning token, leaving every other token untouched.

    `position` is the index of the Motif token in the prefix. Passed in rather
    than searched for, so a sequence whose prefix is malformed raises here
    instead of silently rewriting a musical token.
    """
    if position >= len(ids):
        raise IndexError(f"sequence of {len(ids)} has no position {position}")
    out = list(ids)
    out[position] = token_id
    return out


def apply_arm(
    window: Any,
    *,
    arm: str,
    plan: ArmPlan,
    token_ids: dict[str, int],
    position: int,
) -> list[int]:
    """The token ids this window should train on, for one arm."""
    key = (str(window.composition_id), int(window.start_bar))
    label = plan.label_for(arm, key)
    target = token_ids.get(label)
    if target is None:
        raise KeyError(f"no Motif token for label {label!r}")
    return substitute(window.ids, position=position, token_id=target)
