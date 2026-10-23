"""Training, validation and evaluation.

Runs independently of the Gradio interface so the same code works on a local
CPU, a local GPU or a rented one (proposal 8.4).

`dataset`, `corpus` and `evaluate` need no PyTorch; `train` does, and so is not
re-exported here.
"""

from rse.training.corpus import CorpusInfo, corpus_from_procedural, corpus_from_release
from rse.training.dataset import Window, WindowStats, build_windows, split_windows
from rse.training.evaluate import (
    Aggregate,
    MusicMetrics,
    aggregate,
    compare,
    corpus_fingerprints,
    evaluate_composition,
)

__all__ = [
    "Aggregate",
    "CorpusInfo",
    "MusicMetrics",
    "Window",
    "WindowStats",
    "aggregate",
    "build_windows",
    "compare",
    "corpus_fingerprints",
    "corpus_from_procedural",
    "corpus_from_release",
    "evaluate_composition",
    "split_windows",
]
