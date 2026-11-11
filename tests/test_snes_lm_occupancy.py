"""The four edge cases the corrected occupancy measurement must handle.

Written after `max_simultaneous_voices` reported 13 and 14 concurrent voices
on eight-voice hardware -- a figure that measured the function, not the data.
"""
from rse.snes_lm import hardware_checks as hc


def ev(voice, onset, duration):
    return {"voice": voice, "onset_sec": f"{onset:.6f}",
            "duration_sec": f"{duration:.6f}", "midi_note": 60}


def test_identical_boundaries_do_not_overlap():
    """[0,1) and [1,2) are adjacent, not simultaneous."""
    events = [ev(0, 0.0, 1.0), ev(1, 1.0, 1.0)]
    assert hc.occupancy(events).concurrent_notes == 1
    assert hc.occupancy(events).occupied_voices == 1


def test_simultaneous_starts_on_distinct_voices_all_count():
    events = [ev(v, 0.0, 1.0) for v in range(8)]
    report = hc.occupancy(events)
    assert report.concurrent_notes == 8
    assert report.occupied_voices == 8
    assert report.concurrent_after_stealing == 8


def test_repeated_key_on_same_voice_occupies_one_voice():
    """Overlapping notes on ONE voice are two instances but one voice."""
    events = [ev(3, 0.0, 10.0), ev(3, 1.0, 10.0), ev(3, 2.0, 10.0)]
    report = hc.occupancy(events)
    assert report.concurrent_notes == 3        # quantity 1 may exceed reality
    assert report.occupied_voices == 1         # quantity 3 is the hardware
    assert report.concurrent_after_stealing == 1


def test_repeated_key_on_at_identical_onset_truncates_to_silence():
    events = [ev(3, 1.0, 5.0), ev(3, 1.0, 5.0)]
    assert hc.occupancy(events).occupied_voices == 1


def test_zero_duration_event_occupies_nothing():
    events = [ev(0, 0.0, 0.0), ev(1, 0.0, 0.0)]
    report = hc.occupancy(events)
    assert report.concurrent_notes == 0
    assert report.occupied_voices == 0


def test_zero_duration_key_on_still_ends_the_previous_note():
    """A key-on with no duration silences its voice at that instant."""
    events = [ev(2, 0.0, 10.0), ev(2, 1.0, 0.0)]
    assert hc.occupancy(events).occupied_voices == 1
    spans = hc._voice_intervals(events)
    assert max(float(end) for _, end in spans) == 1.0


def test_occupied_voices_cannot_exceed_eight():
    """The invariant that the original 13/14 figures violated."""
    events = [ev(v % 8, 0.1 * i, 50.0) for i, v in enumerate(range(200))]
    report = hc.occupancy(events)
    assert report.occupied_voices <= hc.N_HARDWARE_VOICES
    assert report.concurrent_after_stealing <= hc.N_HARDWARE_VOICES
    assert report.concurrent_notes > hc.N_HARDWARE_VOICES   # the diagnostic


def test_float_clamp_does_not_leak_a_residual_overlap():
    """Why 13/14 happened: the renderer clamps in float, so `onset+duration`
    can land a hair past the next onset and read as still sounding."""
    events = [ev(v, 0.333333, 0.333333) for v in range(8)]
    events += [ev(v, 0.666666, 0.5) for v in range(8)]
    assert hc.occupancy(events).concurrent_after_stealing <= 8


def test_accumulated_float_onsets_still_clamp_exactly():
    """Decoded onsets are summed floats, not 6-dp literals.

    Re-deriving the end as `onset + duration` landed an ULP past the next
    onset, so a clamped voice read as still sounding and quantity 2 reported 9
    and 10 on eight-voice hardware.
    """
    onsets = []
    t = 0.0
    for step in (0.0192360000000002, 0.5096180000000001, 0.0834550000000002):
        for _ in range(4):
            t += step
            onsets.append(t)
    events = [{"voice": 4, "onset_sec": o, "duration_sec": 5.0, "midi_note": 60}
              for o in onsets]
    events += [{"voice": v, "onset_sec": onsets[0], "duration_sec": 20.0,
                "midi_note": 60} for v in range(8) if v != 4]
    report = hc.occupancy(events)
    assert report.occupied_voices == 8
    assert report.concurrent_after_stealing == 8, report
