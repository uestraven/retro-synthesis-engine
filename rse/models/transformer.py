"""A small decoder-only Transformer.

Proposal 6.1 recommends roughly 5-20M parameters, chosen empirically rather
than fixed before training. `ModelConfig` therefore carries the size, is saved
with every checkpoint, and is reported in parameter counts rather than being
asserted up front.

Deliberately plain: learned positional embeddings, pre-norm blocks, GELU, tied
output weights. Nothing here should be interesting — the interesting parts of
this project are the tokenizer and the hardware constraints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    """Architecture. Saved with the checkpoint; a model without it is unusable."""

    vocab_size: int
    n_layers: int = 6
    n_heads: int = 8
    d_model: int = 256
    d_ff: int | None = None
    max_seq_len: int = 1024
    dropout: float = 0.1
    tie_weights: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model {self.d_model} must divide by n_heads {self.n_heads}")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2")

    @property
    def feed_forward(self) -> int:
        return self.d_ff or 4 * self.d_model

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "tie_weights": self.tie_weights,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention that cannot look ahead."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)

        if pad_mask is None:
            out = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # scaled_dot_product_attention will not combine `is_causal` with an
            # explicit mask, so when there is padding the causal triangle has
            # to be folded into the mask by hand.
            out = self._masked_attention(query, key, value, pad_mask)
        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.resid_dropout(self.out(out))

    def _masked_attention(
        self, query: Tensor, key: Tensor, value: Tensor, pad_mask: Tensor
    ) -> Tensor:
        seq_len = query.size(-2)
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device).tril()
        allowed = causal[None, None, :, :] & pad_mask[:, None, None, :]
        # A padded row can end up with nothing visible; let it attend to itself
        # so the softmax has something to normalize and no NaN appears.
        allowed = (
            allowed | torch.eye(seq_len, dtype=torch.bool, device=query.device)[None, None, :, :]
        )
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=self.dropout if self.training else 0.0,
        )


class Block(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.feed_forward),
            nn.GELU(),
            nn.Linear(config.feed_forward, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x), pad_mask)
        return x + self.mlp(self.norm2(x))


class RseTransformer(nn.Module):
    """Decoder-only Transformer over RSE tokens."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def n_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, ids: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        """`ids` is `(batch, seq)`; returns logits `(batch, seq, vocab)`.

        `pad_mask` is True at real positions and False at padding.
        """
        _batch, seq_len = ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"sequence of {seq_len} exceeds max_seq_len {self.config.max_seq_len}")
        positions = torch.arange(seq_len, device=ids.device)
        x = self.token_embedding(ids) * math.sqrt(self.config.d_model)
        x = self.dropout(x + self.position_embedding(positions)[None, :, :])
        for block in self.blocks:
            x = block(x, pad_mask)
        return self.head(self.norm(x))
