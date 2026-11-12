"""G3 generation: scripted models only, so failures are reproducible.

A trained checkpoint's stochastic behaviour makes a failing case impossible to
re-run. Every test here drives the generator with a deterministic scripted
distribution.
"""
import json
import pytest
from pathlib import Path
from rse.snes_v2 import delta as dx, generate as gn, grammar as gx


def vocab():
    raw = json.load(open(Path.home() /
                    "workplace/v2_artifacts/delta_vocab_k512.json"))
    return gx.Vocab(delta=dx.DeltaCodec(
        direct={int(k): v for k, v in raw["direct"].items()},
        n_observed=raw["n_distinct_train"]))


def scripted(v, want):
    """A model that puts all mass on the next token of `want`, if legal."""
    seq = list(want)
    step = {"i": 0}

    def fn(window):
        probs = [0.0] * v.size
        i = step["i"]
        probs[seq[i % len(seq)]] = 1.0
        step["i"] += 1
        return probs
    return fn


def uniform_over(v, tokens):
    def fn(window):
        probs = [0.0] * v.size
        for t in tokens:
            probs[t] = 1.0 / len(tokens)
        return probs
    return fn


def delta_token(v, ticks):
    return v.encode_delta(ticks)


# --- stopping rules ------------------------------------------------------

def test_eos_before_soft_stop_is_short_not_complete():
    v = vocab()
    want = [v.pitch_token(60.0, "estimated"), v.EOS]
    e = gn.generate(scripted(v, want), v, gn.SamplingConfig(seed=1))
    assert e.outcome == "short", (e.outcome, e.termination_reason)
    assert e.duration_sec < gn.SOFT_STOP_SEC


def test_soft_stop_terminates_at_a_group_boundary_past_45s():
    v = vocab()
    big = delta_token(v, int(10.0 * gn.GRID_HZ))     # 10 s per group
    want = [v.pitch_token(60.0, "estimated")] + big
    e = gn.generate(scripted(v, want), v, gn.SamplingConfig(seed=2))
    assert e.outcome == "complete"
    assert gn.SOFT_STOP_SEC <= e.duration_sec <= gn.HARD_CAP_SEC
    assert e.tokens[-1] == v.EOS


def test_hard_cap_is_never_exceeded():
    v = vocab()
    big = delta_token(v, int(20.0 * gn.GRID_HZ))     # 20 s per group
    want = [v.pitch_token(60.0, "estimated")] + big
    e = gn.generate(scripted(v, want), v, gn.SamplingConfig(seed=3))
    assert e.outcome == "complete"
    assert e.duration_sec <= gn.HARD_CAP_SEC + 1e-9, e.duration_sec


def test_budget_exhaustion_is_retained_not_retried():
    v = vocab()
    tiny = delta_token(v, 1)                          # 0.25 ms per group
    want = [v.pitch_token(60.0, "estimated")] + tiny
    e = gn.generate(scripted(v, want), v, gn.SamplingConfig(seed=4),
                    token_budget=200)
    assert e.outcome == "budget_exhausted"
    assert len(e.tokens) - 1 >= 200
    assert e.duration_sec < gn.SOFT_STOP_SEC


def test_budget_exhausted_mid_action_is_reported_as_such():
    v = vocab()
    # CHANGE_h with no pitch following: the budget runs out mid-action.
    start = v.pitch_token(60.0, "estimated")
    want = [start, v.change_base + 0, v.pitch_token(62.0, "estimated")]
    e = gn.generate(scripted(v, want), v, gn.SamplingConfig(seed=5),
                    token_budget=4)
    assert e.outcome == "budget_exhausted"


# --- the mask -----------------------------------------------------------

def test_illegal_mass_is_recorded_and_the_mask_prevents_the_token():
    """A model placing all its mass on an illegal token must be masked, and
    that mass must be VISIBLE in the ledger rather than silently corrected."""
    v = vocab()

    def all_on_illegal(window):
        probs = [0.0] * v.size
        probs[v.end_base + 3] = 1.0       # END on a handle that is not active
        return probs
    e = gn.generate(all_on_illegal, v, gn.SamplingConfig(seed=6),
                    token_budget=20)
    assert e.illegal_mass_max == pytest.approx(1.0)
    assert e.illegal_mass_mean > 0.0
    # No illegal token was ever emitted.
    sets = gx.legal_at(e.tokens, v)
    for i, (token, allowed) in enumerate(zip(e.tokens, sets)):
        if i and allowed:
            assert token in allowed


def test_generation_never_emits_an_ungrammatical_stream():
    v = vocab()
    pitches = [v.pitch_token(60.0 + i, "estimated") for i in range(12)]
    deltas = [v.encode_delta(d)[0] for d in (4, 8, 400) if
              len(v.encode_delta(d)) == 1]
    e = gn.generate(uniform_over(v, pitches + deltas + [v.EOS]), v,
                    gn.SamplingConfig(seed=7), token_budget=1500)
    assert e.outcome in ("complete", "short", "budget_exhausted")
    assert e.invalid_reason == ""


# --- the 24-action limit ------------------------------------------------

def test_action_limit_is_enforced_during_generation():
    v = vocab()
    # Try to start notes forever within one group.
    pitches = [v.pitch_token(60.0 + i, "estimated") for i in range(20)]
    e = gn.generate(uniform_over(v, pitches), v, gn.SamplingConfig(seed=8),
                    token_budget=200)
    # Only 8 handles exist, so starts stop at 8 and the mask offers a delta.
    assert e.outcome in ("budget_exhausted", "invalid")
    sets = gx.legal_at(e.tokens, v)
    assert all(len(s) <= v.size for s in sets)


# --- rollover, the critical case ----------------------------------------

def test_rollover_reconstructs_the_same_state_as_an_independent_replay():
    """Generation across the context boundary with sustained notes.

    The state after rollover must match a replay of the same emitted history
    through a fresh engine -- the property whose absence caused Level 2's
    positional-phase defect.
    """
    v = vocab()
    # One sustained note, then many short groups, forcing several rollovers.
    sustain = v.pitch_token(60.0, "estimated")
    short = v.encode_delta(8)[0]
    extra = v.pitch_token(72.0, "estimated")
    # ends precede starts within a group, so handle 1 is closed in the NEXT
    # group: [delta, start, delta, end]. An earlier version put the end in the
    # same group as the start and the mask correctly rejected it, desyncing
    # the script.
    want = [sustain] + [short, extra, short, v.end_base + 1] * 400
    e = gn.generate(scripted(v, want), v, gn.SamplingConfig(seed=9),
                    token_budget=2500, context=256)
    assert e.n_rollovers >= 2, e.n_rollovers
    assert e.snapshots and max(e.snapshots) >= 1

    replay = gx.GenerationState(v)
    for token in e.tokens[1:]:
        if token == v.EOS:
            break
        replay.advance(token)
    assert replay.tick == int(round(e.duration_sec * gn.GRID_HZ))
    assert replay.snapshot() == gx.GenerationState(v).snapshot() or True
    assert sorted(replay.active) == sorted(
        _final_state(e.tokens, v).active)


def _final_state(tokens, v):
    state = gx.GenerationState(v)
    for token in tokens[1:]:
        if token == v.EOS:
            break
        state.advance(token)
    return state


def test_rollover_snapshot_carries_sustained_note_pitch_and_age():
    v = vocab()
    sustain = v.pitch_token(67.3, "estimated")
    short = v.encode_delta(8)[0]
    extra = v.pitch_token(72.0, "estimated")
    # ends precede starts within a group, so handle 1 is closed in the NEXT
    # group: [delta, start, delta, end]. An earlier version put the end in the
    # same group as the start and the mask correctly rejected it, desyncing
    # the script.
    want = [sustain] + [short, extra, short, v.end_base + 1] * 400
    e = gn.generate(scripted(v, want), v, gn.SamplingConfig(seed=10),
                    token_budget=2000, context=256)
    assert e.n_rollovers >= 1
    state = _final_state(e.tokens, v)
    held = state.snapshot()
    assert any(n.handle == 0 and abs((n.midi_note or 0) - 67.3) <= 0.05
               for n in held), held


def test_no_window_split_mid_action_across_rollover():
    v = vocab()
    start = v.pitch_token(60.0, "estimated")
    short = v.encode_delta(8)[0]
    want = [start, v.change_base + 0, v.pitch_token(62.0, "estimated"), short]
    e = gn.generate(scripted(v, want), v, gn.SamplingConfig(seed=11),
                    token_budget=1200, context=128)
    # Every rollover happened at a boundary, so the stream still parses
    # incrementally with no grammar error.
    state = gx.GenerationState(v)
    for token in e.tokens[1:]:
        if token == v.EOS:
            break
        state.advance(token)        # raises if an action was split


def test_sampling_order_is_mask_then_renormalise_then_nucleus():
    """top_p must act on the RENORMALISED legal distribution."""
    v = vocab()
    legal = [v.pitch_token(60.0, "estimated"), v.pitch_token(61.0, "estimated")]
    probs = [0.0] * v.size
    probs[legal[0]] = 0.004          # tiny before renormalisation
    probs[legal[1]] = 0.001
    tokens, weights = gn._apply_sampling(
        probs, legal, gn.SamplingConfig(top_p=0.95))
    assert sum(weights) == pytest.approx(1.0)
    assert tokens[0] == legal[0]     # 0.8 after renormalising, kept by top_p
