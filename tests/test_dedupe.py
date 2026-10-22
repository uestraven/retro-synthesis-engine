from __future__ import annotations

from rse.data import dedupe


def make_record(pitches, *, channel="P1", noise=(), start=0, step=240):
    events = [
        {
            "ch": channel,
            "type": "note",
            "tick": start + i * step,
            "dur_ticks": step,
            "pitch": pitch,
        }
        for i, pitch in enumerate(pitches)
    ]
    events += [
        {"ch": "NO", "type": "noise", "tick": i * step, "dur_ticks": 60, "value": value}
        for i, value in enumerate(noise)
    ]
    return {"events": events}


MELODY = [60, 62, 64, 65, 67, 65, 64, 62, 60, 67, 64, 62]


def test_content_hash_ignores_metadata_but_not_music():
    a = make_record(MELODY)
    b = make_record(MELODY)
    assert dedupe.content_hash(a) == dedupe.content_hash(b)
    assert dedupe.content_hash(a) != dedupe.content_hash(make_record(MELODY[::-1]))


def test_content_hash_distinguishes_timing():
    assert dedupe.content_hash(make_record(MELODY)) != dedupe.content_hash(
        make_record(MELODY, step=480)
    )


def test_fingerprint_is_transposition_invariant():
    original = dedupe.fingerprint(make_record(MELODY))
    transposed = dedupe.fingerprint(make_record([p + 7 for p in MELODY]))
    assert original and original == transposed


def test_fingerprint_ignores_the_noise_channel():
    without = dedupe.fingerprint(make_record(MELODY))
    with_noise = dedupe.fingerprint(make_record(MELODY, noise=[1, 5, 9, 13, 2, 6, 10]))
    assert without == with_noise


def test_short_melodies_produce_no_grams():
    assert dedupe.fingerprint(make_record([60, 62])) == set()


def test_sketch_caps_the_fingerprint_size():
    import random

    rng = random.Random(0)
    # Non-repeating, so there are far more than `sketch_size` distinct n-grams.
    long_melody = [rng.randint(40, 90) for _ in range(600)]
    record = make_record(long_melody)
    assert len(dedupe.fingerprint(record, sketch_size=16)) == 16
    assert len(dedupe.fingerprint(record, sketch_size=None)) > 16


def test_jaccard_edges():
    assert dedupe.jaccard(set(), {1}) == 0.0
    assert dedupe.jaccard({1, 2}, {3, 4}) == 0.0
    assert dedupe.jaccard({1, 2}, {1, 2}) == 1.0


def test_exact_duplicates_are_grouped_with_a_stable_representative():
    entries = [
        ("b", "hash1", {1, 2, 3}),
        ("a", "hash1", {1, 2, 3}),
        ("c", "hash2", {9, 8, 7}),
    ]
    report = dedupe.find_duplicates(entries)
    assert report.exact_groups == [["a", "b"]]
    assert report.duplicate_of == {"b": "a"}
    assert report.n_exact == 1
    assert report.n_near == 0


def test_transposed_copies_are_flagged_as_near_duplicates():
    entries = [
        (name, f"hash-{name}", dedupe.fingerprint(record))
        for name, record in (
            ("a", make_record(MELODY)),
            ("b", make_record([p + 5 for p in MELODY])),
        )
    ]
    report = dedupe.find_duplicates(entries)
    assert report.duplicate_of == {"b": "a"}
    assert report.near_pairs and report.near_pairs[0][2] == 1.0
    assert report.n_near == 1


def test_unrelated_music_is_not_flagged():
    other = [48, 55, 48, 53, 48, 51, 48, 50, 48, 57, 48, 59]
    entries = [
        ("a", "h1", dedupe.fingerprint(make_record(MELODY))),
        ("b", "h2", dedupe.fingerprint(make_record(other))),
    ]
    report = dedupe.find_duplicates(entries)
    assert report.duplicate_of == {}
    assert report.near_pairs == []


def test_report_serializes_for_the_run_report():
    entries = [("a", "h", {1, 2, 3}), ("b", "h", {1, 2, 3})]
    payload = dedupe.find_duplicates(entries).to_dict()
    assert payload["n_exact_duplicates"] == 1
    assert payload["duplicate_of"] == {"b": "a"}
