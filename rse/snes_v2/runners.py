"""Model loaders and the per-excerpt measurement step.

Shared by sampling selection and the primary evaluation batch, so the two
cannot diverge in how they load the checkpoint, build the baseline, or decide
whether an outcome is scorable. They lived in the selection script and were
reached by `importlib`, which meant a test could not substitute them and the
evaluation batch had a second path to the same decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import grammar as gx
from . import statistics as sx

GRID_HZ = 4000.0


def transformer_distribution(checkpoint: Path, device: str):
    """Returns (callable, metadata). The callable maps a context to probs."""
    import torch
    import torch.nn.functional as F

    from rse.snes_lm.train import Config, Decoder

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = Config(**{**state["config"], "device": device})
    model = Decoder(state["vocab_size"], cfg)
    model.load_state_dict(state["model"])
    model.eval()
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters())

    def distribution(context):
        with torch.no_grad():
            row = torch.tensor([list(context)], dtype=torch.long, device=device)
            logits = model(row)[0, -1, :]
            return F.softmax(logits.float(), dim=-1).cpu().tolist()

    return distribution, {
        "step": state["step"], "n_parameters": n_parameters,
        "vocab_size": state["vocab_size"],
        # Sampling draws from the model's float output, so the kernels that
        # produce it are part of the result's provenance: a different GPU can
        # change low-order bits and therefore which token is drawn. The run
        # is reproducible on the SAME hardware, not bitwise across hardware.
        "hardware": {
            "device": device,
            "gpu_name": (torch.cuda.get_device_name(0)
                         if device.startswith("cuda") else None),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": (torch.backends.cudnn.version()
                              if device.startswith("cuda") else None),
            "deterministic_note": "reproducible on this hardware and this "
                                  "torch build; not guaranteed bitwise on "
                                  "another GPU architecture",
        },
    }


def baseline_distribution(window_dir: Path, vocab_size: int):
    from rse.snes_v2 import baseline as bl

    train = np.load(window_dir / "train.npz")
    streams = [train["tokens"][i, :train["lengths"][i]].astype(int).tolist()
               for i in range(len(train["lengths"]))]
    model = bl.G3KneserNey(vocab_size=vocab_size, order=1).fit(streams)

    def distribution(context):
        return model.distribution(list(context)[-model.order:])

    return distribution, {"family": "kneser_ney", "order": 1,
                          "n_train_streams": len(streams)}


def measure(excerpt, vocab) -> tuple[dict | None, str]:
    """Frozen statistics for one retained outcome, or None with the reason.

    None is Amendment A1's mechanism: the metrics are undefined, so the
    outcome is not scorable and its candidate is ineligible. The outcome is
    still retained and reported.
    """
    if excerpt.outcome == "invalid":
        return None, f"grammar failure: {excerpt.invalid_reason}"
    try:
        parsed, _ = gx.parse(excerpt.tokens, vocab)
    except Exception as exc:                       # noqa: BLE001
        return None, f"undecodable: {type(exc).__name__}: {exc}"
    notes = gx.notes_from_groups(parsed)
    statistics = sx.excerpt_statistics(notes, grid_hz=GRID_HZ)
    if statistics is None:
        return None, (f"decoded to {len(notes)} notes; the frozen metrics are "
                      f"undefined without a positive span")
    return statistics, ""
