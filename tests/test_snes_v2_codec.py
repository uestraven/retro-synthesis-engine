"""Codec edge cases the user named, plus the invariants that must hold."""
from rse.snes_v2 import codec as cx


def ev(voice, onset, duration, note=60.0, origin="key_on", **kw):
    d = {"voice": voice, "onset_sec": f"{onset:.6f}",
         "duration_sec": None if duration is None else f"{duration:.6f}",
         "midi_note": note, "origin": origin, "pitch_status": "measured"}
    d.update(kw)
    return d


# --- grouping and time ----------------------------------------------------

def test_simultaneous_events_share_one_group():
    groups, rep = cx.encode([ev(v, 0.0, 1.0) for v in range(8)], grid_hz=100)
    starts = [g for g in groups if any(a.kind == "start" for a in g.actions)]
    assert len(starts) == 1
    assert sum(1 for a in starts[0].actions if a.kind == "start") == 8
    assert rep.n_starts == 8


def test_every_group_after_the_first_advances_time():
    events = [ev(0, 0.0, 0.1), ev(1, 0.5, 0.1), ev(2, 1.0, 0.1)]
    groups, _ = cx.encode(events, grid_hz=100)
    assert all(g.delta > 0 for g in groups[1:]), [g.delta for g in groups]


def test_no_time_same_token_exists():
    """The runaway mechanism must be unrepresentable, not merely unlikely."""
    groups, _ = cx.encode([ev(v, 0.0, 1.0) for v in range(8)], grid_hz=100)
    assert all(g.delta != 0 for g in groups[1:])


# --- note lifetimes -------------------------------------------------------

def test_legato_step_is_a_pitch_change_not_an_attack():
    events = [
        ev(0, 0.0, 1.0, 60.0, event_id=1),
        ev(0, 0.5, None, 67.0, origin="legato_step", parent_event_id=1),
    ]
    groups, rep = cx.encode(events, grid_hz=100)
    assert rep.n_starts == 1
    assert rep.n_pitch_changes == 1
    notes, dec = cx.decode(groups, grid_hz=100)
    assert len(notes) == 1
    assert notes[0]["pitch_changes"][0]["midi_note"] == 67.0
    assert dec.n_orphan_pitch_changes == 0


def test_orphan_legato_is_not_promoted_to_an_attack():
    events = [ev(0, 0.5, None, 67.0, origin="legato_step", parent_event_id=999)]
    _, rep = cx.encode(events, grid_hz=100)
    assert rep.n_starts == 0 and rep.n_pitch_changes == 0


def test_null_duration_is_open_not_infinite():
    _, rep = cx.encode([ev(0, 0.0, None)], grid_hz=100)
    assert rep.n_ends_open == 1 and rep.n_ends_derived == 0


def test_derived_ends_are_labelled_as_derived():
    groups, _ = cx.encode([ev(0, 0.0, 1.0)], grid_hz=100)
    notes, _ = cx.decode(groups, grid_hz=100)
    assert notes[0]["end_source"] == "derived_duration"


# --- release ordering and reuse -------------------------------------------

def test_end_before_start_frees_a_slot_within_one_instant():
    """Nine notes, the ninth starting exactly as the first ends."""
    events = [ev(v, 0.0, 1.0) for v in range(8)] + [ev(0, 1.0, 1.0)]
    groups, _ = cx.encode(events, grid_hz=100)
    notes, dec = cx.decode(groups, grid_hz=100)
    assert dec.n_voice_steals == 0, dec
    assert dec.peak_occupancy <= 8


def test_retrigger_on_one_voice_yields_two_notes():
    events = [ev(3, 0.0, 0.5, 60.0), ev(3, 0.5, 0.5, 62.0)]
    groups, _ = cx.encode(events, grid_hz=100)
    notes, dec = cx.decode(groups, grid_hz=100)
    assert len(notes) == 2 and dec.n_voice_steals == 0


def test_ninth_simultaneous_note_steals_and_is_counted():
    events = [ev(v % 8, 0.0, 5.0, 60.0 + v) for v in range(9)]
    groups, _ = cx.encode(events, grid_hz=100)
    notes, dec = cx.decode(groups, grid_hz=100)
    assert dec.n_voice_steals == 1
    assert dec.peak_occupancy <= 8


def test_zero_duration_note_still_ends():
    groups, rep = cx.encode([ev(0, 0.0, 0.0)], grid_hz=100)
    assert rep.n_ends_derived == 1
    notes, dec = cx.decode(groups, grid_hz=100)
    assert len(notes) == 1
    assert notes[0]["duration_sec"] == 0.0
    assert dec.n_zero_length_notes == 1


# --- provenance and roles -------------------------------------------------

def test_original_voice_is_provenance_and_slot_is_reassigned():
    events = [ev(7, 0.0, 1.0)]
    groups, _ = cx.encode(events, grid_hz=100)
    notes, _ = cx.decode(groups, grid_hz=100)
    assert notes[0]["provenance_voice"] == 7
    assert notes[0]["voice"] == 0          # allocator's own slot
    assert notes[0]["role"] is None        # absence is explicit


def test_unpitched_start_is_preserved_not_dropped():
    groups, rep = cx.encode([ev(0, 0.0, 1.0, None)], grid_hz=100)
    assert rep.n_unpitched_starts == 1
    notes, _ = cx.decode(groups, grid_hz=100)
    assert len(notes) == 1 and notes[0]["midi_note"] is None


# --- occupancy invariant --------------------------------------------------

def test_occupancy_never_exceeds_eight_under_load():
    events = [ev(v % 8, 0.01 * i, 9.0, 60.0 + (i % 12))
              for i, v in enumerate(range(300))]
    groups, _ = cx.encode(events, grid_hz=100)
    notes, dec = cx.decode(groups, grid_hz=100)
    assert dec.peak_occupancy <= 8
    assert dec.n_voice_steals > 0          # the load had to be corrected
