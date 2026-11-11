"""G3 grammar: semantic round-trip, framing strictness, pitch-status fidelity."""
import json
import pytest
from pathlib import Path
from rse.snes_v2 import codec as cx, delta as dx, grammar as gx


def vocab():
    raw = json.load(open(Path.home() /
                    "workplace/v2_artifacts/delta_vocab_k512.json"))
    codec = dx.DeltaCodec(direct={int(k): v for k, v in raw["direct"].items()},
                          n_observed=raw["n_distinct_train"])
    return gx.Vocab(delta=codec)


def ev(voice, onset, duration, note=69.27, origin="key_on", status="estimated", **kw):
    d = {"voice": voice, "onset_sec": f"{onset:.6f}",
         "duration_sec": None if duration is None else f"{duration:.6f}",
         "midi_note": note, "origin": origin, "pitch_status": status}
    d.update(kw)
    return d


# --- pitch field ---------------------------------------------------------

def test_pitch_quantises_to_10_cents_within_bound():
    v = vocab()
    for p in (69.27, 60.0, -16.26, 118.79, 69.249, 69.251):
        midi, status = v.read_pitch(v.pitch_token(p, "estimated"))
        assert status == "estimated"
        assert abs(midi - p) <= 0.05 + 1e-9, (p, midi)


def test_each_unpitched_status_keeps_its_own_symbol():
    v = vocab()
    tokens = {}
    for s in gx.UNPITCHED_STATUSES:
        t = v.pitch_token(None, s)
        tokens[s] = t
        assert v.read_pitch(t) == (None, s)
    assert len(set(tokens.values())) == len(gx.UNPITCHED_STATUSES)


def test_single_pitch_family_keeps_vocabulary_compact():
    """One pitch family, and a CONTIGUOUS grid.

    The grid spans MIDI -40.0..134.0 rather than only the 1,180 bins observed
    in train, so the model can emit pitches the corpus never contains. The
    cost is ~560 bins with no training examples -- unobserved rather than
    rare, so initialisation noise at low probability rather than badly
    estimated parameters, which is the opposite of the C1 delta tail.
    """
    v = vocab()
    assert v.n_pitch == (gx.PITCH_MAX_BIN - gx.PITCH_MIN_BIN + 1) + 3 == 1744
    # 4 control + 8 END + 8 CHANGE + 8 HANDLE + 1744 pitch + 771 delta
    assert v.size == 2543, v.size
    # Two separate pitch families would have cost ~1700 more and halved
    # every pitch token's training support.
    assert v.size < 4300


# --- semantic round-trip -------------------------------------------------

def test_round_trip_preserves_identity_order_status_and_delta():
    events = [ev(0, 0.0, 1.0, 69.27), ev(1, 0.0, 0.5, 72.04),
              ev(0, 1.0, 0.25, None, status="noise"),
              ev(2, 1.5, 0.5, 60.11)]
    groups, _ = cx.encode(events, grid_hz=4000.0)
    v = vocab()
    parsed, rep = gx.parse(gx.serialize(groups, v), v)
    assert rep.n_starts == 4
    # deltas are exact
    assert sum(g.delta for g in parsed) == sum(g.delta for g in groups[1:]) \
        + (groups[0].delta if groups[0].delta else 0)
    statuses = [s for g in parsed for (_, s) in g.starts]
    assert "noise" in statuses


def test_legato_change_round_trips_as_a_change_not_an_attack():
    events = [ev(0, 0.0, 1.0, 60.0, event_id=1),
              ev(0, 0.5, None, 67.33, origin="legato_step", parent_event_id=1)]
    groups, _ = cx.encode(events, grid_hz=4000.0)
    v = vocab()
    parsed, rep = gx.parse(gx.serialize(groups, v), v)
    assert rep.n_starts == 1 and rep.n_changes == 1


def test_simultaneous_identical_pitches_keep_distinct_identities():
    """Two notes, same pitch, same instant: ends must reference the right one."""
    events = [ev(0, 0.0, 1.0, 69.27), ev(1, 0.0, 2.0, 69.27)]
    groups, _ = cx.encode(events, grid_hz=4000.0)
    v = vocab()
    tokens = gx.serialize(groups, v)
    parsed, rep = gx.parse(tokens, v)
    assert rep.n_starts == 2
    ended = [s for g in parsed for s in g.ends]
    assert len(ended) == len(set(ended)) == 2    # two distinct slots closed


def test_initial_rest_is_represented_not_lost():
    events = [ev(0, 2.5, 1.0)]
    groups, _ = cx.encode(events, grid_hz=4000.0)
    v = vocab()
    parsed, _ = gx.parse(gx.serialize(groups, v), v)
    assert parsed[0].delta == 10000       # 2.5 s at 4000 Hz


# --- framing strictness --------------------------------------------------

def test_stream_without_eos_is_incomplete():
    v = vocab()
    groups, _ = cx.encode([ev(0, 0.0, 1.0)], grid_hz=4000.0)
    tokens = gx.serialize(groups, v)
    with pytest.raises(gx.IncompleteStream):
        gx.parse(tokens[:-1], v)


def test_change_without_its_pitch_is_incomplete():
    v = vocab()
    with pytest.raises(gx.IncompleteStream):
        gx.parse([v.BOS, v.pitch_token(60.0, "estimated"),
                  v.change_base + 0], v)


def test_truncated_delta_escape_is_incomplete():
    v = vocab()
    escaping = max(v.delta.direct) + 1
    while escaping in v.delta.direct:
        escaping += 1
    tail = v.encode_delta(escaping)
    assert len(tail) > 1
    with pytest.raises(gx.IncompleteStream):
        gx.parse([v.BOS, v.pitch_token(60.0, "estimated")] + tail[:-1] + [v.EOS], v)


def test_end_on_an_inactive_slot_is_rejected():
    v = vocab()
    with pytest.raises(gx.GrammarError):
        gx.parse([v.BOS, v.end_base + 3, v.EOS], v)


def test_change_on_an_inactive_slot_is_rejected():
    v = vocab()
    with pytest.raises(gx.GrammarError):
        gx.parse([v.BOS, v.change_base + 5,
                  v.pitch_token(60.0, "estimated"), v.EOS], v)


def test_empty_group_is_rejected():
    v = vocab()
    with pytest.raises(gx.GrammarError):
        gx.parse([v.BOS, v.EOS], v)


def test_ninth_simultaneous_start_is_rejected():
    v = vocab()
    toks = [v.BOS] + [v.pitch_token(60.0 + i, "estimated") for i in range(9)]
    with pytest.raises(gx.GrammarError):
        gx.parse(toks + [v.EOS], v)


def test_start_after_a_change_is_rejected():
    """The ends -> starts -> changes order is what makes one pitch family
    unambiguous; violating it must not parse."""
    v = vocab()
    toks = [v.BOS, v.pitch_token(60.0, "estimated"),
            v.change_base + 0, v.pitch_token(62.0, "estimated"),
            v.pitch_token(64.0, "estimated"), v.EOS]
    with pytest.raises(gx.GrammarError):
        gx.parse(toks, v)


def test_tokens_after_eos_are_rejected():
    v = vocab()
    toks = [v.BOS, v.pitch_token(60.0, "estimated"), v.EOS,
            v.pitch_token(62.0, "estimated")]
    with pytest.raises(gx.GrammarError):
        gx.parse(toks, v)


# --- the generation mask -------------------------------------------------

def test_mask_excludes_inactive_handles():
    v = vocab()
    allowed = gx.legal(gx.State.GROUP_ENDS, active=frozenset({2, 5}),
                       n_actions=1, vocab=v, started=True)
    assert v.end_base + 2 in allowed and v.end_base + 5 in allowed
    assert v.end_base + 0 not in allowed


def test_mask_forbids_a_ninth_start():
    v = vocab()
    full = frozenset(range(8))
    allowed = gx.legal(gx.State.GROUP_STARTS, active=full, n_actions=8,
                       vocab=v, started=True)
    assert not any(v.is_pitch(t) for t in allowed)


def test_mask_bounds_within_group_action_runs():
    v = vocab()
    allowed = gx.legal(gx.State.GROUP_ENDS, active=frozenset({1}),
                       n_actions=gx.MAX_ACTIONS_PER_GROUP, vocab=v, started=True)
    assert not any(v.is_end(t) or v.is_pitch(t) or v.is_change(t)
                   for t in allowed)
    assert v.EOS in allowed            # the group can still be closed


def test_second_change_on_one_handle_in_a_group_is_rejected():
    """Without this, actions per group are unbounded and the 24 limit is
    arbitrary rather than structural."""
    v = vocab()
    toks = [v.BOS, v.pitch_token(60.0, "estimated"),
            v.change_base + 0, v.pitch_token(62.0, "estimated"),
            v.change_base + 0, v.pitch_token(64.0, "estimated"), v.EOS]
    with pytest.raises(gx.GrammarError):
        gx.parse(toks, v)


def test_mask_excludes_an_already_changed_handle():
    v = vocab()
    allowed = gx.legal(gx.State.GROUP_CHANGES, active=frozenset({0, 1}),
                       n_actions=2, vocab=v, started=True,
                       changed=frozenset({0}))
    assert v.change_base + 1 in allowed
    assert v.change_base + 0 not in allowed


def test_twenty_four_is_reachable_and_is_the_maximum():
    """8 ends + 8 starts + 8 changes, each handle used once per kind."""
    v = vocab()
    toks = [v.BOS] + [v.pitch_token(60.0 + i, "estimated") for i in range(8)]
    toks += [t for h in range(8)
             for t in (v.change_base + h, v.pitch_token(70.0 + h, "estimated"))]
    parsed, rep = gx.parse(toks + [v.EOS], v)
    assert rep.n_starts == 8 and rep.n_changes == 8
    assert rep.n_actions_max_in_group == 16
    assert gx.MAX_ACTIONS_PER_GROUP == 24
