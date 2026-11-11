"""Exactness and escape-validity for the delta codec."""
import pytest
from rse.snes_v2 import delta as dx


def test_c1_round_trips_every_observed_delta_exactly():
    observed = [1, 2, 3, 7, 94, 460, 1536, 6146, 24585]
    codec = dx.fit(observed)
    for d in observed:
        toks = codec.encode(d)
        assert codec.decode(toks) == (d, len(toks)), d


def test_c2_escape_round_trips_exactly_including_the_corpus_maximum():
    codec = dx.fit([1, 2, 3], k=2)          # 3 escapes
    for d in (3, 255, 256, 1000, 24585, 65535, 16777215):
        toks = codec.encode(d)
        got, nxt = codec.decode(toks)
        assert got == d, (d, got, toks)
        assert nxt == len(toks)


def test_unseen_delta_encodes_exactly_not_to_a_neighbour():
    codec = dx.fit([4, 8, 12], k=3)
    assert 9999 not in codec.direct
    assert codec.decode(codec.encode(9999))[0] == 9999


def test_truncated_escape_raises_rather_than_decoding_to_zero():
    codec = dx.fit([1], k=1)
    toks = codec.encode(24585)
    assert len(toks) == 3                     # ESCAPE_2 + two digits
    with pytest.raises(dx.IncompleteDelta):
        codec.decode(toks[:-1])               # one digit missing
    with pytest.raises(dx.IncompleteDelta):
        codec.decode(toks[:1])                # escape alone


def test_a_digit_where_a_delta_is_expected_is_rejected():
    codec = dx.fit([1], k=1)
    digit = codec.n_direct + dx.MAX_ESCAPE_DIGITS
    with pytest.raises(ValueError):
        codec.decode([digit])


def test_zero_and_negative_deltas_are_unrepresentable():
    codec = dx.fit([1, 2])
    for bad in (0, -1):
        with pytest.raises(ValueError):
            codec.encode(bad)


def test_token_cost_matches_the_actual_encoding():
    codec = dx.fit([1, 2, 3], k=2)
    for d in (1, 2, 3, 255, 256, 24585):
        assert codec.n_tokens(d) == len(codec.encode(d)), d


def test_vocab_size_accounts_for_escapes_and_digits():
    codec = dx.fit(range(1, 101), k=50)
    assert codec.vocab_size == 50 + dx.MAX_ESCAPE_DIGITS + dx.RADIX


def test_fit_is_deterministic_under_ties():
    a = dx.fit([5, 5, 9, 9, 1], k=2)
    b = dx.fit([9, 9, 5, 5, 1], k=2)
    assert a.direct == b.direct


# --- guarantee 3: valid time progression ---------------------------------

def test_escape_cannot_decode_to_zero_reintroducing_stagnation():
    """`TIME_SAME` was removed from the codec so stalling is unrepresentable.
    A well-formed escape must not smuggle it back in."""
    codec = dx.fit([1, 2, 3], k=2)
    base = codec.n_direct + dx.MAX_ESCAPE_DIGITS
    with pytest.raises(dx.NonPositiveDelta):
        codec.decode([codec.n_direct, base])            # ESCAPE_1 + digit 0
    with pytest.raises(dx.NonPositiveDelta):
        codec.decode([codec.n_direct + 1, base, base])  # ESCAPE_2 + 0, 0


def test_delta_token_budget_is_finite_by_construction():
    """An escape cannot consume unbounded tokens without advancing time."""
    codec = dx.fit([1], k=1)
    assert codec.MAX_TOKENS_PER_DELTA == 1 + dx.MAX_ESCAPE_DIGITS
    for d in (2, 255, 256, 44399, 16777215):
        assert len(codec.encode(d)) <= codec.MAX_TOKENS_PER_DELTA, d


def test_delta_exceeding_the_digit_budget_is_refused_not_truncated():
    codec = dx.fit([1], k=1)
    with pytest.raises(ValueError):
        codec.encode(dx.RADIX ** dx.MAX_ESCAPE_DIGITS)


def test_non_canonical_forms_decode_correctly_and_are_detectable():
    """Leading zeros are not rejected -- they decode to the right value -- but
    they are visible, so a generator emitting them is not silently normalised."""
    codec = dx.fit([1, 2, 3], k=2)
    base = codec.n_direct + dx.MAX_ESCAPE_DIGITS
    non_canonical = [codec.n_direct + 1, base, base + 5]   # ESCAPE_2 [0, 5]
    assert codec.decode(non_canonical)[0] == 5
    assert not codec.is_canonical(non_canonical)
    assert codec.is_canonical(codec.encode(5))


def test_every_direct_token_decodes_strictly_positive():
    codec = dx.fit(range(1, 600), k=512)
    for value, token in codec.direct.items():
        assert value > 0
        assert codec.decode([token]) == (value, 1)
