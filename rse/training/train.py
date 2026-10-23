"""Training, validation and checkpointing.

Everything proposal 6.3 asks for: train/validation separation, checkpointing,
gradient accumulation, learning-rate scheduling, validation-loss monitoring,
checkpoint selection, reproducible seeds, and a logged configuration.

Deliberately independent of the UI and of any particular corpus, so the same
script runs on a laptop CPU or a rented GPU (proposal 8.4). A checkpoint
carries its model config and its tokenizer vocabulary, because a model without
the exact vocabulary it was trained on is not usable.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from rse import __version__
from rse.models.transformer import ModelConfig, RseTransformer
from rse.tokenizers.tokenizer import RseTokenizer
from rse.training.dataset import TRAIN, VALID, Window, split_windows

CHECKPOINT_VERSION = "rse-checkpoint/1.0.0"


@dataclass
class TrainConfig:
    """Training hyperparameters. Logged into every checkpoint."""

    seed: int = 0
    max_steps: int = 2000
    batch_size: int = 8
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    #: Longest sequence fed to the model; longer windows are truncated.
    seq_len: int = 512
    eval_every: int = 100
    eval_batches: int = 20
    #: Stop when validation loss has not improved for this many evaluations.
    patience: int = 5
    #: Write the best-so-far checkpoint every N evaluations, so a crash or a
    #: dropped terminal costs minutes rather than the whole run. 0 disables it.
    checkpoint_every_evals: int = 1
    #: Mixed precision, when the device supports it.
    amp: bool = True
    device: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "max_steps": self.max_steps,
            "batch_size": self.batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "learning_rate": self.learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "seq_len": self.seq_len,
            "eval_every": self.eval_every,
            "eval_batches": self.eval_batches,
            "patience": self.patience,
            "checkpoint_every_evals": self.checkpoint_every_evals,
            "amp": self.amp,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class TrainHistory:
    """What happened during a run, for reporting and plotting."""

    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    valid_loss: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    best_valid_loss: float = float("inf")
    best_step: int = 0
    stopped_early: bool = False
    wall_clock_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "train_loss": [round(v, 5) for v in self.train_loss],
            "valid_loss": [round(v, 5) for v in self.valid_loss],
            "learning_rates": [round(v, 8) for v in self.learning_rates],
            "best_valid_loss": round(self.best_valid_loss, 5),
            "best_step": self.best_step,
            "stopped_early": self.stopped_early,
            "wall_clock_sec": round(self.wall_clock_sec, 2),
        }


def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed every generator the run touches."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_batch(
    windows: list[Window],
    indices: list[int],
    *,
    pad_id: int,
    seq_len: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build one padded batch of `(inputs, targets, pad_mask)`.

    Targets are inputs shifted by one. Padded positions are masked out of the
    loss, so a short window does not teach the model to predict padding.
    """
    chosen = [windows[i].ids[: seq_len + 1] for i in indices]
    longest = max(len(ids) for ids in chosen)
    longest = max(longest, 2)

    batch_inputs = torch.full((len(chosen), longest - 1), pad_id, dtype=torch.long)
    batch_targets = torch.full((len(chosen), longest - 1), pad_id, dtype=torch.long)
    mask = torch.zeros((len(chosen), longest - 1), dtype=torch.bool)
    for row, ids in enumerate(chosen):
        if len(ids) < 2:
            continue
        inputs = torch.tensor(ids[:-1], dtype=torch.long)
        targets = torch.tensor(ids[1:], dtype=torch.long)
        batch_inputs[row, : inputs.numel()] = inputs
        batch_targets[row, : targets.numel()] = targets
        mask[row, : inputs.numel()] = True
    return batch_inputs.to(device), batch_targets.to(device), mask.to(device)


def learning_rate_at(step: int, config: TrainConfig) -> float:
    """Linear warmup then cosine decay to `min_learning_rate`."""
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    remaining = max(1, config.max_steps - config.warmup_steps)
    progress = min(1.0, (step - config.warmup_steps) / remaining)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine


@torch.no_grad()
def evaluate(
    model: RseTransformer,
    windows: list[Window],
    config: TrainConfig,
    *,
    pad_id: int,
    device: torch.device,
    max_batches: int | None = None,
    seed: int = 0,
) -> float:
    """Mean cross-entropy over a fixed sample of windows.

    The sample is seeded, so validation loss is comparable across runs rather
    than jittering with whichever windows happened to be drawn.
    """
    if not windows:
        return float("nan")
    model.eval()
    rng = random.Random(seed)
    order = list(range(len(windows)))
    rng.shuffle(order)

    total_loss = 0.0
    total_tokens = 0
    batches = 0
    for start in range(0, len(order), config.batch_size):
        if max_batches is not None and batches >= max_batches:
            break
        indices = order[start : start + config.batch_size]
        inputs, targets, mask = make_batch(
            windows, indices, pad_id=pad_id, seq_len=config.seq_len, device=device
        )
        logits = model(inputs, mask)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=pad_id,
            reduction="sum",
        )
        n_tokens = int(mask.sum().item())
        total_loss += float(loss.item())
        total_tokens += n_tokens
        batches += 1
    model.train()
    return total_loss / total_tokens if total_tokens else float("nan")


def train(
    windows: list[Window],
    tokenizer: RseTokenizer,
    *,
    model_config: ModelConfig | None = None,
    config: TrainConfig | None = None,
    checkpoint_path: Path | None = None,
    progress: bool = False,
) -> tuple[RseTransformer, TrainHistory, dict[str, Any]]:
    """Train a model, returning it with its history and a run report."""
    config = config or TrainConfig()
    set_seed(config.seed)
    device = resolve_device(config.device)

    by_split = split_windows(windows)
    train_windows = by_split.get(TRAIN, [])
    valid_windows = by_split.get(VALID, [])
    if not train_windows:
        raise ValueError("no training windows; check the corpus and its splits")

    model_config = model_config or ModelConfig(vocab_size=len(tokenizer))
    if model_config.vocab_size != len(tokenizer):
        raise ValueError(
            f"model vocab_size {model_config.vocab_size} does not match the "
            f"tokenizer's {len(tokenizer)}"
        )
    model = RseTransformer(model_config).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    history = TrainHistory()
    pad_id = tokenizer.vocabulary.pad_id
    rng = random.Random(config.seed)
    started = time.monotonic()
    since_improvement = 0
    best_state: dict[str, Tensor] | None = None
    evals_done = 0

    for step in range(config.max_steps):
        lr = learning_rate_at(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(config.grad_accum_steps):
            indices = [rng.randrange(len(train_windows)) for _ in range(config.batch_size)]
            inputs, targets, mask = make_batch(
                train_windows, indices, pad_id=pad_id, seq_len=config.seq_len, device=device
            )
            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(inputs, mask)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=pad_id,
                )
            scaler.scale(loss / config.grad_accum_steps).backward()
            step_loss += float(loss.item()) / config.grad_accum_steps

        if config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        is_last = step == config.max_steps - 1
        if (step + 1) % config.eval_every == 0 or is_last:
            valid_loss = evaluate(
                model,
                valid_windows,
                config,
                pad_id=pad_id,
                device=device,
                max_batches=config.eval_batches,
                seed=config.seed,
            )
            history.steps.append(step + 1)
            history.train_loss.append(step_loss)
            history.valid_loss.append(valid_loss)
            history.learning_rates.append(lr)
            if progress:
                print(
                    f"  step {step + 1:>6}  train {step_loss:.4f}  "
                    f"valid {valid_loss:.4f}  lr {lr:.2e}"
                )

            evals_done += 1
            improved = not math.isnan(valid_loss) and valid_loss < history.best_valid_loss
            if improved:
                history.best_valid_loss = valid_loss
                history.best_step = step + 1
                # Keep the best weights, not the last: validation loss is the
                # selection criterion proposal 6.3 asks for.
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                since_improvement = 0
            else:
                since_improvement += 1

            # Persist the best weights as we go. A long run on a rented GPU is
            # exactly where a crash is most expensive, and writing only at the
            # end means a dropped connection costs the entire run.
            if (
                checkpoint_path is not None
                and best_state is not None
                and config.checkpoint_every_evals
                and evals_done % config.checkpoint_every_evals == 0
            ):
                _write_interim(checkpoint_path, model, best_state, tokenizer, config, history)

            if not improved and config.patience and since_improvement >= config.patience:
                history.stopped_early = True
                break

    history.wall_clock_sec = time.monotonic() - started
    if best_state is not None:
        model.load_state_dict(best_state)

    report = {
        "rse_version": __version__,
        "checkpoint_version": CHECKPOINT_VERSION,
        "device": str(device),
        "amp": use_amp,
        "n_parameters": model.n_parameters,
        "n_train_windows": len(train_windows),
        "n_valid_windows": len(valid_windows),
        "vocab_size": len(tokenizer),
        "model": model_config.to_dict(),
        "train": config.to_dict(),
        "history": history.to_dict(),
    }
    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, model, tokenizer, config, history, report)
        report["checkpoint"] = str(checkpoint_path)
    return model, history, report


def _write_interim(
    path: Path,
    model: RseTransformer,
    best_state: dict[str, Tensor],
    tokenizer: RseTokenizer,
    config: TrainConfig,
    history: TrainHistory,
) -> None:
    """Save the best-so-far weights without disturbing the live model.

    The running model holds whatever the last step produced, which is not the
    checkpoint we want; `best_state` is the best-by-validation snapshot. Saving
    is wrapped because a full disk on a rented pod should interrupt the run with
    a warning, not destroy it.
    """
    live = model.state_dict()
    try:
        model.load_state_dict(best_state)
        save_checkpoint(path, model, tokenizer, config, history)
    except OSError as exc:  # pragma: no cover - depends on the filesystem
        print(f"  ! could not write interim checkpoint: {exc}", flush=True)
    finally:
        model.load_state_dict(live)


def save_checkpoint(
    path: Path,
    model: RseTransformer,
    tokenizer: RseTokenizer,
    config: TrainConfig,
    history: TrainHistory,
    report: dict[str, Any] | None = None,
) -> Path:
    """Write weights, model config and vocabulary as one file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "rse_version": __version__,
            "state_dict": model.state_dict(),
            "model_config": model.config.to_dict(),
            "train_config": config.to_dict(),
            "vocabulary": tokenizer.vocabulary.to_dict(),
            "history": history.to_dict(),
            "report": report or {},
        },
        path,
    )
    return path


def load_checkpoint(
    path: Path, *, device: str | None = None
) -> tuple[RseTransformer, RseTokenizer, dict[str, Any]]:
    """Load weights together with the vocabulary they were trained on."""
    from rse.tokenizers.vocab import Vocabulary

    resolved = resolve_device(device)
    payload = torch.load(Path(path), map_location=resolved, weights_only=False)
    version = str(payload.get("checkpoint_version", ""))
    if not version.startswith("rse-checkpoint/1"):
        raise ValueError(
            f"unsupported checkpoint version {version!r}; this build reads rse-checkpoint/1.x"
        )
    tokenizer = RseTokenizer(vocabulary=Vocabulary.from_dict(payload["vocabulary"]))
    model_config = ModelConfig.from_dict(payload["model_config"])
    model = RseTransformer(model_config).to(resolved)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, tokenizer, payload
