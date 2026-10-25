"""Widening a trained checkpoint for the conditioning prefix.

Two tables grow and nothing else changes:

- the token embedding, 267 rows to 272, for the five `Motif_*` values;
- the position embedding, 2209 rows to 2305, so an 8-bar window plus its
  one-token prefix fits at `seq_len` 2304. Every observed window is at most
  2209 tokens, so 2304 means no example is ever cropped — and cropping
  matters here beyond the lost notes: at 1024 tokens, 1114 of 3900 cropped
  windows change motif label, so the model would be trained against labels
  describing music outside its context.

Output weights need no separate handling because the head is tied to the token
embedding.

The pretrained rows are copied verbatim and the new rows are drawn once from a
single seed, so every experimental arm starts from byte-identical weights and
the only difference between them is the information in the prefix token.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from rse.models.transformer import ModelConfig

#: New rows are initialized like the model's own embeddings (see
#: `RseTransformer._init_weights`), not left at zero: a zero row gives an
#: identical logit for every new token and a vanishing gradient direction.
INIT_STD = 0.02


def migrate_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    old_vocab: int,
    new_vocab: int,
    old_positions: int,
    new_positions: int,
    seed: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Widen the embedding tables, preserving every pretrained row exactly.

    Returns the new state dict and a record of what changed, so a migration can
    be audited rather than trusted.
    """
    if new_vocab < old_vocab:
        raise ValueError(f"cannot shrink vocabulary {old_vocab} -> {new_vocab}")
    if new_positions < old_positions:
        raise ValueError(f"cannot shrink positions {old_positions} -> {new_positions}")

    out = dict(state_dict)
    generator = torch.Generator().manual_seed(seed)
    report: dict[str, Any] = {
        "vocab": [old_vocab, new_vocab],
        "positions": [old_positions, new_positions],
        "init_std": INIT_STD,
        "seed": seed,
        "expanded": [],
    }

    for key, old_rows, new_rows in (
        ("token_embedding.weight", old_vocab, new_vocab),
        ("position_embedding.weight", old_positions, new_positions),
    ):
        if key not in out:
            continue
        weight = out[key]
        if weight.shape[0] != old_rows:
            raise ValueError(
                f"{key} has {weight.shape[0]} rows, expected {old_rows}; "
                "refusing to migrate a checkpoint that does not match"
            )
        if new_rows == old_rows:
            continue
        extra = torch.empty((new_rows - old_rows, weight.shape[1]), dtype=weight.dtype).normal_(
            mean=0.0, std=INIT_STD, generator=generator
        )
        out[key] = torch.cat([weight, extra], dim=0)
        report["expanded"].append({"key": key, "from": old_rows, "to": new_rows})

    # The head is tied to the token embedding, so it must point at the widened
    # tensor rather than keep a stale copy of the narrower one.
    if "head.weight" in out and "token_embedding.weight" in out:
        out["head.weight"] = out["token_embedding.weight"]
        report["head_retied"] = True
    return out, report


def migrate_checkpoint(
    payload: dict[str, Any],
    *,
    new_vocab: int,
    new_positions: int,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Widen a loaded checkpoint payload in place of its old dimensions."""
    config = payload["model_config"]
    old_vocab = int(config["vocab_size"] if isinstance(config, dict) else config.vocab_size)
    old_positions = int(config["max_seq_len"] if isinstance(config, dict) else config.max_seq_len)
    state, report = migrate_state_dict(
        payload["state_dict"],
        old_vocab=old_vocab,
        new_vocab=new_vocab,
        old_positions=old_positions,
        new_positions=new_positions,
        seed=seed,
    )
    out = dict(payload)
    out["state_dict"] = state
    if isinstance(config, dict):
        out["model_config"] = {
            **config,
            "vocab_size": new_vocab,
            "max_seq_len": new_positions,
        }
    else:
        out["model_config"] = replace(config, vocab_size=new_vocab, max_seq_len=new_positions)
    out["migration"] = report
    return out, report


def geometry_for_finetune(
    pretrained: ModelConfig,
    requested: ModelConfig,
    *,
    needed_positions: int,
) -> ModelConfig:
    """The architecture a fine-tune must use, which is the checkpoint's own.

    A position embedding is a trained tensor, so a fine-tuning run cannot size
    one from its own data: a smaller window set would otherwise build a narrower
    table and fail to load, and a larger one would index past what was trained.
    Taking the geometry from the checkpoint also means every experimental arm
    has the same architecture by construction rather than by matching flags.

    Raises `ValueError` when the request cannot be served by this checkpoint —
    a different architecture, or more positions than it has.
    """
    mismatch = {
        name: (getattr(pretrained, name), getattr(requested, name))
        for name in ("n_layers", "n_heads", "d_model", "vocab_size")
        if getattr(pretrained, name) != getattr(requested, name)
    }
    if mismatch:
        detail = ", ".join(
            f"{name} {theirs} != {ours}" for name, (theirs, ours) in sorted(mismatch.items())
        )
        raise ValueError(f"checkpoint architecture does not match the requested one: {detail}")
    if needed_positions > pretrained.max_seq_len:
        raise ValueError(
            f"this run needs {needed_positions} positions but the checkpoint has "
            f"{pretrained.max_seq_len}"
        )
    return pretrained
