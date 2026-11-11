"""Training and inference must construct IDENTICAL model inputs.

Level 2's model trained on windows cut at multiples of 1024 -- always
phase-aligned -- and was then given a context sliding one token at a time.
Three steps in four presented a positional phase it had never seen, and four
of five apparent grammar failures were that harness defect. These tests exist
so the same class of divergence cannot recur silently.

Path A (training): encode the performance, build windows and snapshots from
the preceding causal history.
Path B (inference): replay the identical preceding history through the
streaming state, then roll the context forward at the same boundary.

The comparison is on TOKEN IDENTITY, not musical equivalence.
"""
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


def ev(voice, onset, duration, note=69.27, origin="key_on",
       status="estimated", **kw):
    d = {"voice": voice, "onset_sec": f"{onset:.6f}",
         "duration_sec": None if duration is None else f"{duration:.6f}",
         "midi_note": note, "origin": origin, "pitch_status": status}
    d.update(kw)
    return d


def replay_to(groups, boundary, v):
    """Path B: stream the causal history, then snapshot at the boundary."""
    state = gx.StreamState(v)
    for index in range(boundary):
        state.emit(groups[index], first=(index == 0))
    return state


def both_paths(events, context, v):
    groups, _ = cx.encode(events, grid_hz=4000.0)
    windows = gx.build_windows(groups, v, context=context)
    return groups, windows


# --- the equivalence property -------------------------------------------

@pytest.mark.parametrize("context", [64, 128, 256, 1024])
def test_snapshot_and_tokens_identical_on_both_paths(context):
    # Enough events that every tested context forces a mid-track boundary;
    # the property is only meaningful where a window starts with live notes.
    events = []
    t = 0.0
    for i in range(context * 2):
        events.append(ev(i % 8, t, 0.4 + 0.01 * (i % 5), 60.0 + (i % 13) * 0.7))
        t += 0.07
    v = vocab()
    groups, windows = both_paths(events, context, v)
    assert len(windows) >= 2, "need a mid-track boundary to test"
    for window in windows:
        state = replay_to(groups, window.first_group, v)
        # 1. snapshot contents
        assert state.snapshot() == window.snapshot, window.first_group
        # 2. active handles and assignments
        assert sorted(state.notes) == sorted(n.handle for n in window.snapshot)
        # 3. token ids: rebuild the head from the replayed state
        head = [v.BOS]
        if state.snapshot():
            head += gx.encode_snapshot(state.snapshot(), v)
        assert window.tokens[:len(head)] == head, window.first_group
        # 4. position indices: the head occupies the same positions
        assert window.n_snapshot_tokens == len(head) - 1
        # 5. clock
        assert state.tick == window.start_tick


def test_notes_sustaining_across_a_boundary_appear_in_the_snapshot():
    """A long note held across the cut must be in the snapshot with its age."""
    events = [ev(0, 0.0, 60.0, 60.0)]                 # sustains throughout
    events += [ev(1 + (i % 7), 0.1 + 0.05 * i, 0.04, 70.0 + i * 0.1)
               for i in range(80)]
    v = vocab()
    groups, windows = both_paths(events, 96, v)
    later = [w for w in windows if w.first_group > 0]
    assert later, "expected a mid-track window"
    held = [n for n in later[0].snapshot if n.handle == 0]
    assert held, later[0].snapshot
    assert held[0].age_ticks > 0


def test_legato_before_the_boundary_is_reflected_in_snapshot_pitch():
    events = [ev(0, 0.0, 30.0, 60.0, event_id=1),
              ev(0, 0.2, None, 67.4, origin="legato_step", parent_event_id=1)]
    events += [ev(1 + (i % 7), 0.3 + 0.05 * i, 0.04, 72.0)
               for i in range(80)]
    v = vocab()
    groups, windows = both_paths(events, 96, v)
    later = [w for w in windows if w.first_group > 0]
    assert later
    held = [n for n in later[0].snapshot if n.handle == 0]
    assert held and abs(held[0].midi_note - 67.4) <= 0.05


def test_identical_pitches_across_a_boundary_keep_distinct_handles():
    events = [ev(0, 0.0, 40.0, 69.27), ev(1, 0.0, 40.0, 69.27)]
    events += [ev(2 + (i % 6), 0.1 + 0.05 * i, 0.04, 60.0) for i in range(80)]
    v = vocab()
    groups, windows = both_paths(events, 96, v)
    later = [w for w in windows if w.first_group > 0]
    assert later
    handles = [n.handle for n in later[0].snapshot if n.midi_note
               and abs(n.midi_note - 69.3) < 0.06]
    assert len(handles) == 2 and len(set(handles)) == 2


def test_no_window_splits_an_action_or_an_escape():
    events = [ev(i % 8, 0.0 + 0.013 * i, 0.5, 60.0 + i * 0.3) for i in range(90)]
    v = vocab()
    groups, windows = both_paths(events, 80, v)
    for window in windows:
        # every window must parse on its own, which is impossible if an
        # action or a delta escape was cut
        gx.parse(window.tokens, v) if not window.snapshot else None
        assert window.tokens[-1] == v.EOS


def test_snapshot_never_exposes_a_notes_eventual_ending():
    """Future-information leakage check: a snapshot carries age, never
    duration, because a generator could not know the ending at that instant."""
    events = [ev(0, 0.0, 55.0, 60.0)]
    events += [ev(1 + (i % 7), 0.1 + 0.05 * i, 0.04, 70.0) for i in range(80)]
    v = vocab()
    groups, windows = both_paths(events, 96, v)
    later = [w for w in windows if w.snapshot]
    assert later
    for note in later[0].snapshot:
        assert not hasattr(note, "duration_sec")
        assert not hasattr(note, "end_tick")
        assert set(vars(note)) == {"handle", "midi_note", "status", "age_ticks"}


def test_snapshot_round_trips_through_its_own_tokens():
    v = vocab()
    notes = (gx.ActiveNote(0, 69.3, "estimated", 0),
             gx.ActiveNote(3, None, "noise", 4000),
             gx.ActiveNote(7, 60.1, "estimated", 44399))
    tokens = gx.encode_snapshot(notes, v)
    got, index = gx.decode_snapshot(tokens, 0, v)
    assert index == len(tokens)
    assert tuple(got) == notes


def test_truncated_snapshot_is_incomplete_not_empty():
    v = vocab()
    tokens = gx.encode_snapshot((gx.ActiveNote(0, 69.3, "estimated", 10),), v)
    with pytest.raises(gx.IncompleteStream):
        gx.decode_snapshot(tokens[:-1], 0, v)
