# Retro Synthesis Engine (RSE)

Symbolic music generation for vintage game-console sound hardware. RSE composes
structured musical events and renders them through hardware-aware synthesis,
rather than generating audio waveforms directly.

**Compose symbolically. Validate against hardware. Render through chip-aware synthesis.**

Initial target: the NES APU's four channels (two pulse, one triangle, one
noise). Genesis and SNES profiles come later.

## Status

**Phase 1 — data foundation — is implemented.** Everything downstream of it is
scaffolding with a docstring naming the phase that fills it in.

| Phase | Scope | State |
|---|---|---|
| 1 | Repository, environment, dataset ingestion, statistics, provenance | **done** |
| 2 | Canonical RSE Event Format, hardware validator | not started |
| 3 | Rule-based baseline generator, NES renderer, exports | not started |
| 4 | Tokenizer, Transformer training and inference | not started |
| 5 | Gradio interface | not started |
| 6 | Regression tests, originality screening, alpha release | not started |

What Phase 1 gives you: a reproducible path from a dataset archive to
per-composition intake records, dataset statistics, a duplicate report, and a
provenance manifest — with every source file traceable and every measurement
written to disk.

## Install

Python 3.11 or 3.12. Phase 4 pins the upper bound: torch, miditok and symusic
have no 3.13+ wheels yet. The Phase 1 pipeline needs only `mido`.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"     # Windows: .venv\Scripts\python
```

Optional extras, added when the phase that needs them arrives:
`.[ml]` (training), `.[audio]` (rendering), `.[ui]` (Gradio).

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
```

The files it produces are not music anyone would want to listen to. They exist
to exercise parsing, normalization, measurement and de-duplication.

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
  generation/  phrase planning and sampling          — Phase 3, 4
  renderers/   chip-aware audio synthesis            — Phase 3
  exporters/   MIDI, WAV, JSON, tracker formats      — Phase 3
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
.venv/bin/python -m pytest          # 95 tests, no dataset required
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format .
```

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
