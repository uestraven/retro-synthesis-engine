# Retro Synthesis Engine (RSE)

Symbolic music generation for vintage game-console sound hardware. RSE composes
structured musical events and renders them through hardware-aware synthesis,
rather than generating audio waveforms directly.

**Compose symbolically. Validate against hardware. Render through chip-aware synthesis.**

Initial target: the NES APU's four channels (two pulse, one triangle, one
noise). Genesis and SNES profiles come later.

## Status

**RSE composes and renders original NES music today.** One command gives you a
structured, looping, hardware-valid piece as WAV and MIDI. The machine learning
is what's still missing — Phases 1 and 3 are done, and that is enough for a
working (if unsurprising) music generator.

```bash
rse generate --mood dark --energy high --bars 16
```

| Phase | Scope | State |
|---|---|---|
| 1 | Repository, environment, dataset ingestion, statistics, provenance | **done** |
| 2 | Canonical RSE Event Format, hardware validator | not started |
| 3 | Rule-based generator, NES renderer, exports | **done** |
| 4 | Tokenizer, Transformer training and inference | not started |
| 5 | Gradio interface | not started |
| 6 | Regression tests, originality screening, alpha release | not started |

The renderer is a native APU synthesizer — no external binary, no subprocess,
no download. Pitch is quantized to what the hardware timer can actually
produce, phase advances in CPU cycles, and the APU's nonlinear channel mixing
is modelled. Verified numerically rather than by ear: rendered pitches match
the achievable hardware frequency within 1% across MIDI 33–108. See
[docs/renderer-survey.md](docs/renderer-survey.md) for what it does *not* model
(hardware envelopes are the big one) and why the alternatives were rejected.

The generator decides structure before notes: form, key and a chord
progression first, then melody, accompaniment, bass and drums written into
that frame. Repeated phrases reuse their material, so an AABA form hears as a
return. Phase 4 has to beat this baseline to justify itself (proposal §11.3).
See [docs/generator.md](docs/generator.md).

## Install

Python 3.11 or 3.12. Phase 4 pins the upper bound: torch, miditok and symusic
have no 3.13+ wheels yet. Ingestion and rendering need only `mido` and `numpy`.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"     # Windows: .venv\Scripts\python
```

Optional extras, added when the phase that needs them arrives:
`.[ml]` (training) and `.[ui]` (Gradio).

No GPU is required for ingestion, testing or interface work.

## Try it without the dataset

NES-MDB is not redistributable, so a synthetic stand-in with the same shape —
four named channels, expressive controllers, loop markers, split directories —
is generated on demand:

```bash
python scripts/make_sample_release.py
rse fetch   --dataset rse_sample --release midi --archive datasets/sample/rse_sample_midi.tar.gz
rse ingest  --dataset rse_sample --release midi
rse inspect --dataset rse_sample --release midi
rse render  --file datasets/raw/rse_sample/midi/train/train_001.mid --out demo.wav --stems
```

The files it produces are not music anyone would want to listen to. They exist
to exercise parsing, normalization, measurement and de-duplication — but that
last command does produce real NES-style audio, and `--stems` writes each
channel separately so you can hear them in isolation.

To hear the renderer on music you'd recognize, point `rse render --file` at any
MIDI file. Name its tracks `P1`, `P2`, `TR` and `NO` to control which part
lands on which channel; otherwise they're assigned in order.

## Use NES-MDB

The dataset's download URLs are deliberately absent from
`configs/datasets/nesmdb.json`: fetch the archive from
[the repository](https://github.com/chrisdonahue/nesmdb) yourself, then register
it. Once you have a URL and its hash, fill both into the config so the
acquisition becomes reproducible.

```bash
rse fetch   --dataset nesmdb --release midi --archive ~/Downloads/nesmdb_midi.tar.gz
rse ingest  --dataset nesmdb --release midi
rse inspect --dataset nesmdb --release midi
```

`nesmdb` is registered as `research_only`. The nesmdb *software* is open source;
the game compositions it contains are not. See "Data governance" below.

## Commands

```
rse fetch    acquire a release, extract it safely, write a provenance manifest
rse ingest   parse, normalize, measure and de-duplicate a fetched release
rse inspect  summarize a processed release, or one MIDI file with --file
rse generate compose an original piece: plan, notes, audio, MIDI, loop check
rse render   synthesize NES audio from a MIDI file or an intake record
rse profiles list hardware profiles and their channel limits
```

Add `--json` to any command for a machine-readable document instead of a
summary. Nothing requires the web UI; `scripts/*.py` are thin wrappers for
people who prefer scripts to entry points.

Inspecting a single file is the fastest way to see what the pipeline makes of it:

```
$ rse inspect --file datasets/raw/nesmdb/midi/train/some_song.mid
  ch  role        notes       range  max_poly  overlaps  assigned_by
  P1  pulse1        128       70-90         1         0  track_name
  ...
```

## Layout

```
rse/
  data/        acquisition, MIDI parsing, normalization, stats, de-duplication
  hardware/    versioned hardware profiles (Phase 2 adds the constraint engine)
  events/      canonical RSE Event Format            — Phase 2
  tokenizers/  symbolic tokenization                 — Phase 4
  models/      Transformer definitions               — Phase 4
  training/    training and evaluation               — Phase 4
  generation/  planning, the rule-based composer     — working
  renderers/   NES APU synthesizer                   — working
  exporters/   WAV and MIDI; tracker formats later   — working
  ui/          local Gradio application              — Phase 5
configs/       hardware profiles and dataset definitions (committed)
datasets/      downloads, raw sources, processed output (git-ignored)
checkpoints/   model weights (git-ignored)
docs/          technical specifications
```

Datasets and checkpoints are never committed.

## What ingestion produces

Under `datasets/processed/<dataset>/<release>/`:

| File | Contents |
|---|---|
| `acquisition.json` | Where the archive came from, its hash, licence terms |
| `manifest.jsonl` | One provenance record per source file |
| `records/*.json` | One intake record per composition |
| `index.jsonl` | One row per composition: split, note count, compliance, duplicate flag |
| `statistics.json` | Dataset-wide measurements |
| `report.json` | Everything above plus the configuration that produced it |

See [docs/phase1-data-foundation.md](docs/phase1-data-foundation.md) for the
intake record schema and the design rules the pipeline holds itself to.

## Three rules the pipeline will not bend

**Controllers are preserved.** NES-MDB encodes volume and timbre as controller
data. A pipeline that strips MIDI controllers to "clean up" the data throws away
the expression it is supposed to learn.

**Quantization is additive.** Original ticks are always present. A quantized
position is written to a *separate* field, never over the source timing.

**Noise is not pitch.** The NES noise channel's note number selects a period,
not a pitch. It is never transposed, never harmonized, and never bucketed into a
pitch histogram as though it were a note.

Phase 1 also *measures* rather than enforces: an out-of-range note or an
overlapping pair is recorded in the report and left in the record. The Phase 2
constraint engine is what rejects and reallocates, and it should be built
against real numbers from a real dataset.

## Development

```bash
.venv/bin/python -m pytest          # 287 tests, no dataset required
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format .
```

The renderer has no reference recordings to diff against, so its tests measure
the audio instead: pitch by autocorrelation, duty ratios off the waveform,
LFSR period and balance, noise bandwidth, mixer nonlinearity, and aliasing
versus oversampling.

## Data governance

Training sources carry a `use_class` of `research_only`, `commercially_cleared`
or `unknown`, recorded per file in the manifest and copied into every intake
record. NES-MDB is `research_only`.

A software repository's licence does not grant rights over the compositions in
its dataset. Before any commercial model or asset distribution, training-data
rights, sample-bank licences and output similarity all need review.

## References

- [NES Music Database](https://github.com/chrisdonahue/nesmdb)
- [MidiTok](https://miditok.readthedocs.io/en/latest/)
- [FamiStudio](https://famistudio.org/doc/)
- [Furnace Tracker](https://github.com/tildearrow/furnace)
