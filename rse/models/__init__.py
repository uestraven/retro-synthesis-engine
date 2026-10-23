"""Model definitions and checkpoint loading.

A small decoder-only Transformer. Size is chosen empirically and saved with the
checkpoint rather than fixed before training (proposal 6.1).

Importing this package requires PyTorch, which is an optional extra — the data
pipeline, renderer and rule-based generator all work without it.
"""

from rse.models.transformer import ModelConfig, RseTransformer

__all__ = ["ModelConfig", "RseTransformer"]
