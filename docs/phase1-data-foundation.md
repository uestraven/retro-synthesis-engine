# Phase 1 — Data Foundation

Reference for the ingestion pipeline: what it writes, what the intake record
contains, and which design decisions are deliberate and should not be
"simplified" later without a reason.

## Pipeline

```
archive ──fetch──▶ datasets/raw/<ds>/<rel>/        (read-only source tree)
                   datasets/processed/<ds>/<rel>/
                     acquisition.json              where it came from
                     manifest.jsonl                provenance per source file
         ──ingest─▶   records/*.json               one intake record per composition
                      index.jsonl                  one row per composition
                      statistics.json              dataset-wide measurements
                      report.json                  stats + config + failures + duplicates
```

`fetch` never parses music and `ingest` never writes to `raw/`. The raw tree
stays a faithful copy of the archive so a re-run is always reproducible from it.

## Archive handling

Source archives are untrusted input:

- Each extracts into its own directory, which must be empty or explicitly
  overwritten.
- Members whose paths resolve outside that directory are refused.
- Symlinks, device nodes and other non-regular members are refused.
- Tar extraction uses `filter="data"`, dropping ownership and permission data.
- A checksum, when known, is verified before extraction.

## Provenance

One `SourceRecord` per source file, in `manifest.jsonl`:

| Field | Meaning |
|---|---|
| `source_id` | `<dataset>:<16 hex>` over relpath + content hash; stable across re-extraction, unique per file |
| `dataset_id`, `dataset_version` | Which dataset and release |
| `relpath` | Path relative to the release's raw directory |
| `sha256`, `size_bytes` | Content identity |
| `use_class` | `research_only` \| `commercially_cleared` \| `unknown` |
| `license_note` | Free text from the dataset definition |
| `source_url`, `acquisition_method`, `acquired_utc` | How and when it arrived |

The id hashes path *and* content. Hashing content alone would give two
byte-identical files one identity, and de-duplication cannot flag a pair that
shares an id.

`use_class` is validated on construction — an unrecognized value raises rather
than defaulting to permissive.

## Intake record

`schema_version: rse-intake/0.1.0`. This is the lossless landing zone, not the
canonical format; Phase 2's RSE Event Format is derived from it and versioned
separately.

```jsonc
{
  "schema_version": "rse-intake/0.1.0",
  "composition_id": "nesmdb:1a2b...",
  "name": "train/some_song.mid",
  "hardware_profile": "nes",
  "hardware_schema_version": "rse-hardware/0.1.0",
  "source": { /* the provenance record above */ },
  "quantize_grid": null,
  "midi": { "format": 1, "ticks_per_beat": 480, "n_tracks": 5 },
  "tempo_map":       [ { "tick": 0, "beat": 0.0, "bar": 0, "bar_pos": 0.0, "sec": 0.0,
                         "us_per_beat": 500000, "bpm": 120.0 } ],
  "time_signatures": [ { "tick": 0, ..., "numerator": 4, "denominator": 4 } ],
  "key_signatures":  [ { "tick": 0, "key": "Am" } ],
  "loop_markers":    [ { "tick": 0, ..., "type": "marker", "text": "loop_start" } ],
  "text_events":     [ /* every marker/text/cue/lyric, unfiltered */ ],
  "channels":        [ /* one per profile channel, see below */ ],
  "events":          [ /* time-ordered, see below */ ],
  "measurements":    { /* see below */ },
  "warnings":        [ "note left open at end of track 2 ..." ]
}
```

### Events

Every event carries `tick` (source resolution), `beat` (quarter notes), `bar`,
`bar_pos`, and `sec`. With `--quantize-grid G`, a `q_beat` field is **added**;
`tick` and `beat` are never modified. Notes also gain `q_dur_beats`, floored at
one grid unit so quantization cannot delete a note by rounding it to zero
length.

| `type` | Fields |
|---|---|
| `note` | `ch`, `pitch`, `velocity`, `dur_ticks`, `dur_beats` |
| `noise` | `ch`, `value`, `velocity`, `dur_ticks`, `dur_beats` |
| `control` | `ch`, `control`, `value` |
| `program` | `ch`, `program` |
| `pitchwheel` | `ch`, `pitch` |
| `aftertouch` | `ch`, `value` |
| `polytouch` | `ch`, `note`, `value` |

`noise` events use `value`, not `pitch`, and the key is absent rather than null.
Anything that transposes, harmonizes or histograms by pitch therefore cannot
touch the noise channel by accident.

### Channel summaries

One entry per profile channel, present even when unused:

`channel_id`, `role`, `type`, `pitched`, `midi_track`, `midi_channel`,
`track_name`, `assignment_method`, `n_notes`, `value_min`, `value_max`,
`max_simultaneous`, `overlap_count`, `out_of_range_values`, `programs`,
`controllers_used`.

### Measurements

`n_events`, `n_notes`, `n_controls`, `dropped_notes`, `dropped_controls`,
`end_tick`, `end_beat`, `end_sec`, `n_bars`, `constant_tempo`,
`constant_meter`, `channels_used`, `hardware_compliant`.

`n_bars` treats the end tick as exclusive: a note finishing on the downbeat of
bar 8 makes an eight-bar piece, not a nine-bar one.

## Channel assignment

General MIDI program numbers say nothing about console channels, so the mapping
is inferred and the method recorded:

1. `track_name` — the track name matches a `track_name_patterns` regex in the
   profile (`P1`, `pulse 1`, `sq1`, …), case-insensitively.
2. `instrument_name` — same match against the instrument-name meta event.
3. `ordinal` — remaining note-bearing voices, in track order, onto remaining
   channels in profile order.
4. `unassigned` — nothing left to assign to.

A name match wins because it is the only evidence strong enough to reorder
channels relative to track order. Two tracks claiming one channel is a warning,
and the second claimant is placed elsewhere rather than merged or dropped.
Silent tracks never consume a channel. Voices with nowhere to go are counted in
`dropped_notes` and named in `warnings` — never silently discarded.

## Malformed input

| Situation | Behaviour |
|---|---|
| Unreadable file | Recorded in `report.failures`; the run continues |
| `note_on` velocity 0 | Treated as `note_off` |
| Repeated `note_on`, same pitch | First note closed at the second's onset |
| Note left open at track end | Truncated to the end of the track, warned |
| `note_off` with no `note_on` | Warned; no note invented |
| Zero-length note | Dropped, warned |
| Out-of-range pitch | **Kept**, listed in `out_of_range_values`, marks the record non-compliant |
| Overlapping notes on a mono channel | **Kept**, counted in `overlap_count` |

The last two are the point of Phase 1. The profile's pitch ranges are
provisional — the working ranges of the NES-MDB score representation, not a
derivation from APU timer registers. Measuring how often real data falls outside
them is how those numbers get fixed before Phase 2 starts enforcing them.

## De-duplication

Two independent signals, neither of which deletes anything:

- **Exact** — SHA-256 over channel, tick, duration and value of every note
  event. Ignores file metadata, so a re-save with different tempo metadata still
  collides.
- **Near** — melodic n-gram fingerprint: 6-grams of pitch *intervals*, per
  pitched channel, so it is transposition-invariant. The noise channel is
  excluded. Fingerprints are reduced to a bottom-k sketch (k=256) to bound
  memory; candidate pairs come from sharing at least one n-gram, which keeps the
  comparison count near-linear. Grams shared by more than 64 compositions are
  ignored as common idioms. Pairs at or above Jaccard 0.85 are flagged and
  union-found into clusters with the lowest id as representative.

The sketch makes similarity an estimate, which is why the output is a review
flag (`duplicate_of` on the index row) and not a verdict.

## Statistics

`DatasetStats` is a streaming accumulator — it never holds more than one record
at a time, so memory does not scale with dataset size. It collects per-channel
note counts, value ranges, value and duration histograms, peak polyphony,
overlap and out-of-range tallies, controller and program usage, tempo and meter
histograms, assignment-method counts, bar/second/note distributions, warning
categories, and `use_class` counts.

## Still open after Phase 1

- The NES profile's pitch ranges need confirming against real NES-MDB numbers.
- The DPCM channel is excluded; `excluded_channels` in the profile records why.
- Loop-marker detection relies on text matching (`loop`, `repeat start/end`).
  Whether NES-MDB files carry such markers at all is an open question the
  statistics will answer (`n_with_loop_markers`).
- The near-duplicate threshold of 0.85 is a starting guess, not a measured one.
