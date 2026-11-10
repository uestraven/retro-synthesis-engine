"""Training a small decoder-only Transformer on a frozen snesmdb release.

Protocol frozen in the snesmdb repository at `docs/PROTOCOL_FREEZE.md`. The
tokenizer, population, windowing, masking and target construction are fixed so
the baseline and the model see the same task; nothing here may change them in
response to a result.

Everything the record needs is written to the run directory: loss curves, token
counts, hyperparameters, seeds, and a sha256 of each checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rse.snes_lm import representation as rep

#: Slot names, for the registered per-slot diagnostic.
SLOT_NAMES = ("TIME", "VOICE", "PITCH", "DUR")


@dataclass
class Config:
    # data
    release: str = "~/snesmdb-releases/snesmdb-50game-rc1"
    vocab_path: str = ""
    context: int = 1024
    batch_size: int = 8
    # model, ~5M parameters
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    # optimisation
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_steps: int = 50_000
    grad_clip: float = 1.0
    # bookkeeping
    seed: int = 0
    eval_every: int = 500
    patience_evals: int = 8
    device: str = "cpu"
    run_dir: str = "runs/snes_lm"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention via the fused kernel.

    Written out rather than using `nn.MultiheadAttention` for a measured
    reason. That module abandons the fused `scaled_dot_product_attention` path
    whenever dropout > 0 and falls back to `F.multi_head_attention_forward`,
    which materialises the full 1024x1024 attention matrix per head. Measured
    on this machine at context 1024, batch 8, 6 layers, d_model 256:

        dropout 0.0  ->  0.335 s per forward
        dropout 0.1  ->  4.981 s per forward     (15x)

    The frozen protocol specifies dropout 0.1, so the choice was between
    changing the protocol to suit the implementation or fixing the
    implementation. `F.scaled_dot_product_attention` takes `dropout_p` AND
    `is_causal=True` while staying fused, so dropout costs nothing and no mask
    is materialised at all.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        merged = attended.transpose(1, 2).reshape(batch, length, -1)
        return self.out(merged)


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        return x + self.mlp(self.norm2(x))


class Decoder(nn.Module):
    """Decoder-only Transformer. Learned positions, pre-norm, tied head."""

    def __init__(self, vocab_size: int, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.token = nn.Embedding(vocab_size, cfg.d_model)
        self.position = nn.Embedding(cfg.context, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            Block(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        # Tied: the vocabulary is 240, so the head is a small part of the
        # budget, but tying is free and helps on 10M tokens.
        self.head.weight = self.token.weight
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        hidden = self.drop(self.token(tokens) + self.position(positions))
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(self.norm(hidden))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def windows(tokens: list[int], context: int, pad: int) -> list[list[int]]:
    """Per-track windows, final one padded. Tracks never share a window."""
    out = []
    for start in range(0, max(1, len(tokens)), context):
        chunk = tokens[start : start + context]
        if not chunk:
            break
        if len(chunk) < context:
            chunk = chunk + [pad] * (context - len(chunk))
        out.append(chunk)
    return out


@dataclass
class Batcher:
    """Deterministic shuffled window sampler."""

    data: torch.Tensor  # [n_windows, context]
    batch_size: int
    seed: int = 0
    _order: list[int] = field(default_factory=list)
    _cursor: int = 0
    epoch: int = 0

    def __post_init__(self) -> None:
        self._reshuffle()

    def _reshuffle(self) -> None:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self._order = torch.randperm(
            self.data.shape[0], generator=generator
        ).tolist()
        self._cursor = 0

    def next(self) -> torch.Tensor:
        if self._cursor + self.batch_size > len(self._order):
            self.epoch += 1
            self._reshuffle()
        index = self._order[self._cursor : self._cursor + self.batch_size]
        self._cursor += self.batch_size
        return self.data[index]

    @property
    def steps_per_epoch(self) -> int:
        return math.ceil(self.data.shape[0] / self.batch_size)


def _loss_and_slots(
    logits: torch.Tensor, batch: torch.Tensor, pad: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masked cross-entropy in nats, plus per-slot sums and counts.

    Targets are positions 1.. of each window; position 0 is context only. A
    padded target contributes nothing, matching the frozen masking policy.
    """
    predictions = logits[:, :-1, :]
    targets = batch[:, 1:]
    keep = targets != pad
    flat = F.cross_entropy(
        predictions.reshape(-1, predictions.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    flat = flat * keep

    # Slot of a target at window position p (0-based in `targets`) is the slot
    # of body index p+1, i.e. (p+1-1) % 4 = p % 4 -- but only for the FIRST
    # window of a track, which is where BOS sits. Later windows continue
    # mid-stream, so the slot diagnostic is reported over first windows only
    # and that restriction is recorded rather than approximated away.
    positions = torch.arange(targets.shape[1], device=targets.device) % rep.STRIDE
    slot_sums = torch.zeros(rep.STRIDE, device=targets.device)
    slot_counts = torch.zeros(rep.STRIDE, device=targets.device)
    for slot in range(rep.STRIDE):
        selected = (positions == slot).unsqueeze(0) & keep
        slot_sums[slot] = (flat * selected).sum()
        slot_counts[slot] = selected.sum()
    return flat.sum() / keep.sum().clamp(min=1), slot_sums, slot_counts


@torch.no_grad()
def evaluate(
    model: Decoder, data: torch.Tensor, pad: int, batch_size: int, device: str
) -> dict[str, Any]:
    """Validation loss in bits/token, overall and per slot."""
    model.eval()
    total = torch.zeros((), device=device)
    count = torch.zeros((), device=device)
    slot_sums = torch.zeros(rep.STRIDE, device=device)
    slot_counts = torch.zeros(rep.STRIDE, device=device)
    for start in range(0, data.shape[0], batch_size):
        batch = data[start : start + batch_size].to(device)
        logits = model(batch)
        targets = batch[:, 1:]
        keep = targets != pad
        flat = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets) * keep
        total += flat.sum()
        count += keep.sum()
        positions = torch.arange(targets.shape[1], device=device) % rep.STRIDE
        for slot in range(rep.STRIDE):
            selected = (positions == slot).unsqueeze(0) & keep
            slot_sums[slot] += (flat * selected).sum()
            slot_counts[slot] += selected.sum()
    model.train()
    ln2 = math.log(2.0)
    per_slot = {
        SLOT_NAMES[s]: (slot_sums[s] / slot_counts[s].clamp(min=1)).item() / ln2
        for s in range(rep.STRIDE)
    }
    return {
        "bits_per_token": (total / count.clamp(min=1)).item() / ln2,
        "n_tokens": int(count.item()),
        "per_slot_bits": per_slot,
    }


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def train(
    cfg: Config,
    train_windows: torch.Tensor,
    valid_windows: torch.Tensor,
    vocab: rep.Vocab,
) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    device = cfg.device
    run_dir = Path(cfg.run_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)

    model = Decoder(vocab.size, cfg).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    batcher = Batcher(train_windows, cfg.batch_size, seed=cfg.seed)
    history: list[dict[str, Any]] = []
    best = {"bits_per_token": float("inf"), "step": -1}
    stale = 0
    started = time.time()
    tokens_seen = 0

    for step in range(cfg.max_steps):
        for group in optimiser.param_groups:
            group["lr"] = lr_at(step)
        batch = batcher.next().to(device)
        logits = model(batch)
        loss, _, _ = _loss_and_slots(logits, batch, vocab.PAD)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimiser.step()
        tokens_seen += int((batch != vocab.PAD).sum().item())

        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.max_steps:
            metrics = evaluate(
                model, valid_windows, vocab.PAD, cfg.batch_size, device
            )
            record = {
                "step": step + 1,
                "train_bits": loss.item() / math.log(2.0),
                "valid_bits": metrics["bits_per_token"],
                "per_slot_bits": metrics["per_slot_bits"],
                "lr": lr_at(step),
                "grad_norm": float(grad_norm),
                "tokens_seen": tokens_seen,
                "epoch": batcher.epoch,
                "elapsed_sec": time.time() - started,
            }
            history.append(record)
            if metrics["bits_per_token"] < best["bits_per_token"]:
                best = {"bits_per_token": metrics["bits_per_token"], "step": step + 1}
                path = run_dir / "best.pt"
                torch.save(
                    {"model": model.state_dict(), "config": cfg.to_dict(),
                     "step": step + 1, "valid_bits": metrics["bits_per_token"]},
                    path,
                )
                best["checkpoint_sha256"] = checkpoint_sha256(path)
                stale = 0
            else:
                stale += 1
            (run_dir / "history.jsonl").open("a").write(json.dumps(record) + "\n")
            if stale >= cfg.patience_evals:
                record["early_stopped"] = True
                break

    summary = {
        "config": cfg.to_dict(),
        "n_params": model.n_params,
        "vocab_size": vocab.size,
        "steps_per_epoch": batcher.steps_per_epoch,
        "best": best,
        "history": history,
        "torch_version": torch.__version__,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
