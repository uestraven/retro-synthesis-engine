"""Everything that must be true before a cache is worth GPU time.

Each check here exists because its absence cost something. Training once ran
on 200 procedural windows while reporting success, and the Gate 3 arms trained
against a cache whose position 6 held `Section_A` rather than a Motif token --
so the source and the prefix are verified against the real file, not against a
freshly encoded example.

Exit code is 0 only if every check passes. The report prints every failure
rather than the first, because a bad cache usually fails several at once and
fixing them one error at a time wastes a build each round.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from rse.codec.benchmark import percentile
from rse.tokenizers.frozen import (
    CONDITIONING_GRAMMAR_VERSION,
    CONTEXT_TOKENS,
    REPRESENTATION_ID,
    frozen_config,
)
from rse.tokenizers.grammar import TokenGrammar
from rse.tokenizers.tokenizer import RseTokenizer
from rse.tokenizers.vocab import MOTIF, SECTION
from rse.training.cache_contract import (
    CacheContractError,
    load_contract,
    vocabulary_hash,
)
from rse.training.cache_contract import verify as contract_verify
from rse.training.dataset import load_windows

#: Windows inspected token-by-token. The structural checks below cover all.
DEEP_SAMPLE = 2000


@dataclass
class Report:
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, passed: bool, note: str = "") -> None:
        self.checks.append((name, passed, note))

    @property
    def ok(self) -> bool:
        return all(passed for _n, passed, _d in self.checks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--window-bars", type=int, default=8)
    parser.add_argument("--context", type=int, default=CONTEXT_TOKENS)
    parser.add_argument("--mask-prefix", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    tokenizer = RseTokenizer(frozen_config())
    report = Report()

    print(f"cache     : {args.cache}", flush=True)
    windows = load_windows(args.cache)
    print(f"windows   : {len(windows)} loaded", flush=True)

    # -- representation and contract ---------------------------------------
    try:
        contract = contract_verify(
            args.cache,
            windows,
            tokenizer,
            window_bars=args.window_bars,
            representation_id=REPRESENTATION_ID,
            conditioning_grammar_version=CONDITIONING_GRAMMAR_VERSION,
            context_tokens=args.context,
        )
        report.add("semantic cache contract", True, "all fields match the frozen tokenizer")
    except CacheContractError as exc:
        report.add("semantic cache contract", False, str(exc))
        contract = load_contract(args.cache)

    if contract is not None:
        report.add(
            "representation id",
            contract.representation_id == REPRESENTATION_ID,
            f"{contract.representation_id!r}",
        )
        report.add(
            "vocabulary hash",
            contract.vocab_hash == vocabulary_hash(tokenizer),
            f"{contract.vocab_hash[:16]} ({contract.vocab_size} tokens)",
        )
        report.detail["contract"] = contract.to_dict()

    # -- source is real music, not the procedural generator ----------------
    source = str((contract.corpus or {}).get("source", "")) if contract else ""
    synthetic = bool((contract.corpus or {}).get("synthetic", False)) if contract else True
    generated_ids = sum(1 for w in windows if str(w.composition_id).startswith("generated:"))
    report.add(
        "source is the real dataset",
        source == "dataset" and not synthetic and generated_ids == 0,
        f"source={source!r} synthetic={synthetic} generated-id windows={generated_ids}",
    )

    # -- official splits ----------------------------------------------------
    per_split = Counter(w.split for w in windows)
    pieces_by_split: dict[str, set[str]] = {}
    for window in windows:
        pieces_by_split.setdefault(window.split, set()).add(str(window.composition_id))
    overlap = {
        f"{a}/{b}": sorted(pieces_by_split.get(a, set()) & pieces_by_split.get(b, set()))[:5]
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))
    }
    leaked = {k: v for k, v in overlap.items() if v}
    report.add(
        "no composition appears in two splits",
        not leaked,
        f"windows {dict(per_split)}; leaks {leaked}" if leaked else f"windows {dict(per_split)}",
    )
    report.detail["per_split_windows"] = dict(per_split)
    report.detail["per_split_pieces"] = {k: len(v) for k, v in pieces_by_split.items()}

    # -- sequence lengths ---------------------------------------------------
    lengths = [len(w.ids) for w in windows]
    over = [w for w in windows if len(w.ids) > args.context]
    report.add(
        f"every window fits {args.context} tokens",
        not over,
        f"max {max(lengths)}, over {len(over)}",
    )
    report.detail["tokens"] = {
        "median": round(median(lengths)),
        "p90": round(percentile([float(n) for n in lengths], 0.90)),
        "p99": round(percentile([float(n) for n in lengths], 0.99)),
        "max": max(lengths),
    }

    # -- fragments ----------------------------------------------------------
    fragments = [w for w in windows if w.is_fragment]
    parents = {(w.composition_id, w.source_start_bar) for w in fragments}
    bad_provenance = [
        w for w in fragments if w.source_start_bar is None or w.source_n_bars != args.window_bars
    ]
    report.add(
        "fragments carry complete provenance",
        not bad_provenance,
        f"{len(fragments)} fragments from {len(parents)} split windows",
    )

    # Every fragment group must cover its parent window's bars exactly once.
    groups: dict[tuple[str, int], list[Any]] = {}
    for w in fragments:
        groups.setdefault((str(w.composition_id), int(w.source_start_bar or 0)), []).append(w)
    broken: list[str] = []
    for (piece, start), members in groups.items():
        covered: list[int] = []
        for w in members:
            rel = w.start_bar - start
            covered.extend(range(rel, rel + w.n_bars))
        if sorted(covered) != list(range(args.window_bars)):
            broken.append(f"{piece}@{start}: {sorted(covered)}")
    report.add(
        "fragment groups cover their window with no gap or overlap",
        not broken,
        f"{len(groups)} groups checked" + (f"; broken {broken[:3]}" if broken else ""),
    )

    # -- keys are unique, so a label join cannot double-match --------------
    keys = [(str(w.composition_id), int(w.start_bar)) for w in windows]
    duplicates = [k for k, n in Counter(keys).items() if n > 1]
    report.add(
        "cache keys are unique",
        not duplicates,
        f"{len(duplicates)} duplicated" + (f", e.g. {duplicates[:3]}" if duplicates else ""),
    )

    # -- prefix grammar, on the real tokens -------------------------------
    motif_values = set(tokenizer.config.motif_values)
    wrong_prefix: list[str] = []
    conditioned: Counter[str] = Counter()
    masked_music: list[str] = []
    ungrammatical: list[str] = []
    for window in windows[:DEEP_SAMPLE]:
        tokens = [tokenizer.vocabulary.token_of(i) for i in window.ids]
        if len(tokens) <= args.mask_prefix:
            wrong_prefix.append(f"{window.composition_id}@{window.start_bar}: too short")
            continue
        motif = tokens[6]
        if not motif.startswith(MOTIF):
            wrong_prefix.append(
                f"{window.composition_id}@{window.start_bar}: position 6 is {motif!r}"
            )
        else:
            conditioned[motif[len(MOTIF) :]] += 1
        # The masked region must be exactly the prefix. A Section or Bar token
        # inside it would be excluded from the loss, which is the mistake that
        # would quietly stop the model learning structure.
        for position in range(min(args.mask_prefix, len(tokens))):
            token = tokens[position]
            if token == "Bar" or token.startswith(SECTION):
                masked_music.append(
                    f"{window.composition_id}@{window.start_bar}: {token!r} at {position}"
                )
                break
        grammar = TokenGrammar(tokenizer.vocabulary)
        for token in tokens:
            if not grammar.advance(token):
                ungrammatical.append(f"{window.composition_id}@{window.start_bar}: {token!r}")
                break

    report.add(
        "position 6 holds a Motif token in every sampled window",
        not wrong_prefix,
        f"{len(wrong_prefix)} wrong" + (f", first {wrong_prefix[0]}" if wrong_prefix else ""),
    )
    report.add(
        "foundation is unconditioned: every Motif token is `unspecified`",
        set(conditioned) <= {"unspecified"},
        f"{dict(conditioned)}",
    )
    report.add(
        f"--mask-prefix {args.mask_prefix} masks no musical token",
        not masked_music,
        f"{len(masked_music)} windows mask music"
        + (f", first {masked_music[0]}" if masked_music else ""),
    )
    report.add(
        "every sampled window is grammatical",
        not ungrammatical,
        f"{len(ungrammatical)} rejected" + (f", first {ungrammatical[0]}" if ungrammatical else ""),
    )
    report.detail["motif_values_in_cache"] = dict(conditioned)
    report.detail["unused_motif_values"] = sorted(motif_values - set(conditioned))

    # -- labels are not inherited ------------------------------------------
    report.add(
        "no fragment claims an inherited label",
        all(not w.to_dict().get("labels_inherited", False) for w in fragments),
        f"{len(fragments)} fragments checked",
    )

    # -- output -------------------------------------------------------------
    width = max(len(name) for name, _p, _d in report.checks)
    print()
    for name, passed, note in report.checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<{width}}  {note}")
    print()
    for key, value in report.detail.items():
        if key != "contract":
            print(f"  {key}: {value}")

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "cache": str(args.cache),
                    "ok": report.ok,
                    "checks": [{"name": n, "passed": p, "note": d} for n, p, d in report.checks],
                    "detail": report.detail,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.out}")

    print(f"\npreflight: {'PASS' if report.ok else 'FAIL'}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
