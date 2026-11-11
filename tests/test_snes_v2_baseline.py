"""The matched G3 baseline: no zero mass, proper backoff, honest scoring."""
import math
import pytest
from rse.snes_v2 import baseline as bl


def test_distribution_sums_to_one_at_every_order():
    m = bl.G3NGram(vocab_size=10, order=3).fit([[1,2,3,1,2,3,4,5,1,2]])
    for context in ([], [1], [1,2], [1,2,3], [9,9,9]):
        total = sum(m.distribution(context))
        assert total == pytest.approx(1.0, abs=1e-9), (context, total)


def test_unseen_token_never_gets_zero_probability():
    """The Level 2 n-gram scored worse than its own backoff from zero mass."""
    m = bl.G3NGram(vocab_size=50, order=4).fit([[1,2,3,4,5] * 20])
    for token in range(50):
        p = m.prob([1,2,3,4], token)
        assert p > 0.0, token


def test_unseen_context_falls_through_to_a_shorter_one():
    m = bl.G3NGram(vocab_size=20, order=3).fit([[7,8,9] * 30])
    seen = m.prob([7,8], 9)
    unseen_context = m.prob([19,19,8], 9)   # trailing 8 still informative
    assert seen > 1.0 / 20
    assert unseen_context > 1.0 / 20, unseen_context


def test_prob_matches_distribution():
    m = bl.G3NGram(vocab_size=12, order=2).fit([[3,4,5,3,4,6,3,4,5]])
    dist = m.distribution([3,4])
    for token in range(12):
        assert m.prob([3,4], token) == pytest.approx(dist[token], abs=1e-12)


def test_higher_order_is_not_worse_than_its_own_backoff():
    """Interpolation must not lose to the model it backs off to."""
    train = [[1,2,3,4,5,6,7,8] * 12]
    held = [[1,2,3,4,5,6,7,8] * 3]
    masks = [[True] * len(held[0])]
    results = {}
    for order in (1, 2, 3, 4):
        m = bl.G3NGram(vocab_size=16, order=order).fit(train)
        results[order] = bl.score(m, held, masks)["bits_per_scored_target"]
    for order in (2, 3, 4):
        assert results[order] <= results[order - 1] + 1e-9, results


def test_score_honours_the_mask_denominator():
    m = bl.G3NGram(vocab_size=8, order=2).fit([[1,2,3,1,2,3]])
    stream = [1,2,3,1,2,3]
    all_true = bl.score(m, [stream], [[True]*6])
    half = bl.score(m, [stream], [[True,True,True,False,False,False]])
    assert all_true["n_scored"] == 5          # index 0 is never a target
    assert half["n_scored"] == 2


def test_prediction_depends_on_history_not_absolute_position():
    """G3 is variable-length and stateful. Level 2's comparator was handed the
    slot from absolute position, which made its structural validity look like
    learning. Tested behaviourally: the same context must give the same
    distribution wherever it occurs.

    An earlier version of this test grepped the source for "stride" and
    matched the docstring explaining why strides are NOT used -- a test that
    read documentation instead of behaviour.
    """
    m = bl.G3NGram(vocab_size=16, order=3).fit([[2,5,7,2,5,7,9,2,5,7]])
    early = m.distribution([2, 5, 7])
    # Same trailing context, reached after a different number of tokens.
    late = m.distribution([9, 9, 9, 9, 9, 2, 5, 7])
    assert early == pytest.approx(late, abs=1e-12)


def test_order_candidates_are_declared_before_selection():
    assert bl.ORDER_CANDIDATES == (1, 2, 3, 4, 5)
