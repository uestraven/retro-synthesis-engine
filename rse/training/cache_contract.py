"""What a window cache promises, and how that promise is checked.

A cache used to be treated as neutral storage: `--windows path.jsonl` reused
whatever was there. That cost a whole experiment. The Gate 3 arms trained on a
cache built before the `Motif_` token existed, so its prefix was six tokens and
index 6 held `Section_A`; substituting "the motif token" at index 6 therefore
overwrote a structural token in all 11,027 windows. Every consistency check
passed, because each one compared the cache against itself.

The lesson is specific: a position index is meaningless without the tokenizer
that produced the sequence, and a bare list of token ids carries no record of
which tokenizer that was. So a cache now ships a contract naming the tokenizer,
the vocabulary, the codec and the prefix schema, and `verify` checks the
contract against the tokens actually loaded rather than against freshly encoded
examples. Checking a fresh encode would have passed happily while the real
cache was wrong.

Verification is deliberately loud and fail-closed. A cache with no contract is
rejected rather than trusted, because every cache built before this module
existed is exactly the kind that caused the failure.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rse import __version__
from rse.data.provenance import sha256_bytes
from rse.tokenizers.grammar import TokenGrammar
from rse.tokenizers.tokenizer import RseTokenizer

#: How many windows to inspect token-by-token. The prefix schema and grammar
#: are properties of the encoder, so a defect shows up in the first handful;
#: the join and count checks below cover every row regardless.
SAMPLE = 256


class CacheContractError(RuntimeError):
    """A cache does not match the tokenizer that is about to train on it."""


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def vocabulary_hash(tokenizer: RseTokenizer) -> str:
    """Digest of the exact token list, not of the config that produced it.

    Two configs can differ in a field that does not reach the vocabulary, and
    a config hash would reject those caches needlessly. The token list is what
    a trained embedding table is indexed by, so it is what must match.
    """
    payload = json.dumps(list(tokenizer.vocabulary.tokens), separators=(",", ":"))
    return sha256_bytes(payload.encode())


def prefix_schema(tokenizer: RseTokenizer) -> list[str]:
    """Token-family prefix expected at each conditioning-prefix position.

    Derived by encoding nothing at all, so it is whatever this tokenizer
    actually emits rather than a constant that can fall out of step with it.
    """
    from rse.events.schema import Composition

    empty = Composition(hardware_profile=tokenizer.config.profile_id, ticks_per_beat=480)
    encoded = tokenizer.encode(empty)
    schema: list[str] = []
    for token in encoded.tokens:
        if token.startswith("Bar") or token.startswith("Section_") or token.startswith("Loop_"):
            break
        # `<bos>` and friends are whole tokens; the rest are families.
        schema.append(token if token.startswith("<") else token.split("_", 1)[0] + "_")
    return schema


@dataclass
class CacheContract:
    """Everything needed to decide whether a cache may be trained on."""

    rse_version: str = __version__
    build_commit: str = field(default_factory=_git_commit)
    vocab_version: str = ""
    vocab_hash: str = ""
    vocab_size: int = 0
    codec: str = "A"
    sub_row_divisions: int = 1
    steps_per_bar: int = 16
    aligned_grid: bool = True
    #: Named representation this cache was built under, and the grammar the
    #: conditioning prefix follows. Separate fields because a prefix change can
    #: invalidate a cache without touching the codec, and vice versa.
    representation_id: str = ""
    conditioning_grammar_version: str = ""
    #: Tempo configuration. Part of the vocabulary, so the hash would catch a
    #: mismatch -- recorded explicitly so the error message says which field.
    tempo_ratio: float = 1.03
    tempo_min: int = 40
    tempo_max: int = 360
    #: Context the cache was built for, and what happens to a window that does
    #: not fit. A cache built under one policy is not valid under another.
    context_tokens: int = 0
    overflow_policy: str = "none"
    n_fragments: int = 0
    #: Digest of the source composition ids, so a cache cannot be reused
    #: across corpora or after a re-ingest.
    source_manifest_hash: str = ""
    #: Recorded even though it leaves the vocabulary untouched: the whole
    #: point of the Sections ablation is that it changes the data and
    #: nothing else, so a vocabulary hash cannot detect reusing the wrong
    #: cache for it.
    emit_sections: bool = True
    #: Expected token family at each prefix position, in order.
    prefix_schema: list[str] = field(default_factory=list)
    window_bars: int = 0
    n_windows: int = 0
    per_split: dict[str, int] = field(default_factory=dict)
    #: Where the music came from, so a cache cannot be silently reused across
    #: corpora. `corpus_hash` covers the ids, not the audio.
    corpus: dict[str, Any] = field(default_factory=dict)
    corpus_hash: str = ""

    @classmethod
    def build(
        cls,
        tokenizer: RseTokenizer,
        windows: list[Any],
        *,
        window_bars: int,
        corpus: dict[str, Any] | None = None,
        representation_id: str = "",
        conditioning_grammar_version: str = "",
        context_tokens: int = 0,
        overflow_policy: str = "none",
    ) -> CacheContract:
        per_split: dict[str, int] = {}
        for window in windows:
            per_split[window.split] = per_split.get(window.split, 0) + 1
        ids = sorted({f"{w.composition_id}:{w.start_bar}" for w in windows})
        return cls(
            vocab_version=tokenizer.config.vocab_version,
            vocab_hash=vocabulary_hash(tokenizer),
            vocab_size=len(tokenizer),
            codec=tokenizer.config.codec,
            sub_row_divisions=tokenizer.config.sub_row_divisions,
            steps_per_bar=tokenizer.config.steps_per_bar,
            aligned_grid=tokenizer.config.aligned_grid,
            emit_sections=tokenizer.config.emit_sections,
            representation_id=representation_id,
            conditioning_grammar_version=conditioning_grammar_version,
            tempo_ratio=tokenizer.config.tempo_ratio,
            tempo_min=tokenizer.config.tempo_min,
            tempo_max=tokenizer.config.tempo_max,
            context_tokens=context_tokens,
            overflow_policy=overflow_policy,
            n_fragments=sum(1 for w in windows if getattr(w, "is_fragment", False)),
            source_manifest_hash=sha256_bytes(
                "\n".join(sorted({str(w.composition_id) for w in windows})).encode()
            ),
            prefix_schema=prefix_schema(tokenizer),
            window_bars=window_bars,
            n_windows=len(windows),
            per_split=per_split,
            corpus=dict(corpus or {}),
            corpus_hash=sha256_bytes("\n".join(ids).encode()),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheContract:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def contract_path(cache: Path) -> Path:
    return Path(str(cache) + ".contract.json")


def save_contract(cache: Path, contract: CacheContract) -> Path:
    path = contract_path(cache)
    path.write_text(json.dumps(contract.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def load_contract(cache: Path) -> CacheContract | None:
    path = contract_path(cache)
    if not path.is_file():
        return None
    return CacheContract.from_dict(json.loads(path.read_text()))


def verify(
    cache: Path,
    windows: list[Any],
    tokenizer: RseTokenizer,
    *,
    window_bars: int | None = None,
    representation_id: str = "",
    conditioning_grammar_version: str = "",
    context_tokens: int = 0,
    sample: int = SAMPLE,
) -> CacheContract:
    """Check a loaded cache against the tokenizer about to train on it.

    Raises `CacheContractError` listing every problem found, rather than the
    first: a stale cache usually fails several checks at once, and fixing them
    one error message at a time wastes a GPU reservation each round.
    """
    problems: list[str] = []
    contract = load_contract(cache)

    if contract is None:
        raise CacheContractError(
            f"{cache} has no contract at {contract_path(cache).name}. Every cache "
            f"built before the contract existed is unverifiable -- including the "
            f"one that invalidated Gate 3 -- so it is rejected rather than "
            f"trusted. Rebuild it with the current tokenizer."
        )

    expected_hash = vocabulary_hash(tokenizer)
    if contract.vocab_hash != expected_hash:
        problems.append(
            f"vocabulary hash {contract.vocab_hash[:12]} != {expected_hash[:12]} "
            f"(cache size {contract.vocab_size}, tokenizer {len(tokenizer)})"
        )
    if contract.codec != tokenizer.config.codec:
        problems.append(f"codec {contract.codec!r} != {tokenizer.config.codec!r}")
    for name, mine in (
        ("tempo_ratio", tokenizer.config.tempo_ratio),
        ("tempo_min", tokenizer.config.tempo_min),
        ("tempo_max", tokenizer.config.tempo_max),
    ):
        theirs = getattr(contract, name)
        if theirs != mine:
            problems.append(f"{name} {theirs} != {mine}")
    if representation_id and contract.representation_id != representation_id:
        problems.append(f"representation {contract.representation_id!r} != {representation_id!r}")
    if (
        conditioning_grammar_version
        and contract.conditioning_grammar_version != conditioning_grammar_version
    ):
        problems.append(
            f"conditioning grammar {contract.conditioning_grammar_version!r} != "
            f"{conditioning_grammar_version!r}"
        )
    if context_tokens and contract.context_tokens != context_tokens:
        problems.append(f"context_tokens {contract.context_tokens} != {context_tokens}")
    over = [w for w in windows if len(w.ids) > contract.context_tokens > 0]
    if over:
        problems.append(
            f"{len(over)} windows exceed the cache's own {contract.context_tokens}-token "
            f"context, longest {max(len(w.ids) for w in over)}; the overflow policy "
            f"({contract.overflow_policy}) did not hold"
        )
    if contract.emit_sections != tokenizer.config.emit_sections:
        problems.append(
            f"emit_sections {contract.emit_sections} != {tokenizer.config.emit_sections}"
        )
    if contract.aligned_grid != tokenizer.config.aligned_grid:
        problems.append(f"aligned_grid {contract.aligned_grid} != {tokenizer.config.aligned_grid}")
    if contract.steps_per_bar != tokenizer.config.steps_per_bar:
        problems.append(
            f"steps_per_bar {contract.steps_per_bar} != {tokenizer.config.steps_per_bar}"
        )
    if contract.sub_row_divisions != tokenizer.config.sub_row_divisions:
        problems.append(
            f"sub_row_divisions {contract.sub_row_divisions} != "
            f"{tokenizer.config.sub_row_divisions}"
        )
    if window_bars is not None and contract.window_bars != window_bars:
        problems.append(f"window_bars {contract.window_bars} != {window_bars}")
    if contract.n_windows != len(windows):
        problems.append(f"contract counts {contract.n_windows} windows, loaded {len(windows)}")

    # The check that matters: the tokens actually present, not a fresh encode.
    expected = prefix_schema(tokenizer)
    if contract.prefix_schema != expected:
        problems.append(f"prefix schema {contract.prefix_schema} != {expected}")

    bad_prefix: list[str] = []
    bad_grammar: list[str] = []
    for window in windows[:sample]:
        tokens = [tokenizer.vocabulary.token_of(i) for i in window.ids]
        for position, family in enumerate(expected):
            if position >= len(tokens):
                bad_prefix.append(f"{window.composition_id}:{window.start_bar} too short")
                break
            token = tokens[position]
            matches = token == family if family.startswith("<") else token.startswith(family)
            if not matches:
                bad_prefix.append(
                    f"{window.composition_id}:{window.start_bar} position {position} "
                    f"holds {token!r}, expected {family}*"
                )
                break
        grammar = TokenGrammar(tokenizer.vocabulary)
        for token in tokens:
            if not grammar.advance(token):
                bad_grammar.append(f"{window.composition_id}:{window.start_bar} rejects {token!r}")
                break

    if bad_prefix:
        problems.append(
            f"{len(bad_prefix)} of {min(len(windows), sample)} sampled windows have a "
            f"wrong prefix; first: {bad_prefix[0]}"
        )
    if bad_grammar:
        problems.append(
            f"{len(bad_grammar)} of {min(len(windows), sample)} sampled windows are "
            f"ungrammatical; first: {bad_grammar[0]}"
        )

    if problems:
        raise CacheContractError(
            f"{cache} does not match this tokenizer:\n  - " + "\n  - ".join(problems)
        )
    return contract
