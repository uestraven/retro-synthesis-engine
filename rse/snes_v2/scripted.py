"""A hand-written reference performance in G3.

Not sampled from any model. It exists so the measurement path -- parse,
reconstruct notes, score under the frozen statistics -- can be exercised
without a checkpoint, and so a defect in the generator cannot hide a defect
in the measurement. One implementation, shared by the bundle validator and
the tests, so the two cannot disagree about what a known-good stream is.

Every token is checked against the real grammar mask as it is written, so
the script cannot encode a grammar the model is not held to.
"""

from __future__ import annotations

from . import grammar as gx


def reference_performance(vocab: gx.Vocab) -> list[int]:
    """Two notes, one legato change, two distinct onsets, a clean close.

    Two distinct onsets matter: an excerpt whose notes all share one onset
    has no positive span, and the frozen metrics are undefined for it.
    """
    state = gx.GenerationState(vocab)
    #: BOS is the state's precondition rather than a sampled token, exactly
    #: as in `generate()`, so it is seeded instead of masked.
    tokens: list[int] = [vocab.BOS]

    def emit(token: int) -> None:
        legal = state.legal()
        if token not in legal:
            raise AssertionError(
                f"scripted token {token} is illegal here; "
                f"{len(legal)} legal alternatives")
        state.advance(token)
        tokens.append(token)

    emit(vocab.pitch_token(60.0, "estimated"))      # handle 0 starts at t=0
    for token in vocab.encode_delta(4000):          # 1.0 s
        emit(token)
    emit(vocab.pitch_token(67.0, "estimated"))      # handle 1 starts
    emit(vocab.change_base + 0)                     # legato on handle 0
    emit(vocab.pitch_token(62.0, "estimated"))      # its new pitch
    for token in vocab.encode_delta(2000):          # 0.5 s
        emit(token)
    emit(vocab.end_base + 1)                        # handle 1 releases
    for token in vocab.encode_delta(2000):
        emit(token)
    # EOS closes an OPEN group, so the final group needs an action: a group
    # with no actions has nothing to close. Handle 0 releases here.
    emit(vocab.end_base + 0)
    emit(vocab.EOS)
    return tokens
