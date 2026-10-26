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
| 2 | Canonical RSE Event Format, hardware validator | **done** |
| 3 | Rule-based generator, NES renderer, exports | **done** |
| 4 | Tokenizer, Transformer training and inference | **pipeline done**; no model trained on real music |
| 5 | Gradio interface | **done** |
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

The Phase 4 machine learning is **wired up but untrained on real music**: the
tokenizer, Transformer, training loop, grammar-constrained sampler and
evaluation harness all work end to end, and the only corpus available so far
is output from the rule-based generator. That validates the pipeline and
nothing about musical quality — see [docs/model.md](docs/model.md), which says
so at more length. Training on real NES music is blocked on fetching NES-MDB.

## Install

Python 3.11 or 3.12. Phase 4 pins the upper bound: torch, miditok and symusic
have no 3.13+ wheels yet. Ingestion and rendering need only `mido` and `numpy`.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"     # Windows: .venv\Scripts\python
```

Optional extras: `.[ml]` for training and sampling (PyTorch), `.[ui]` for the
Gradio interface once it exists. Ingestion, rendering and the rule-based
generator all work without them.

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
rse validate check a composition against the hardware profile and its loop
rse train    train a composition model (needs the ml extra)
rse sample   generate music from a trained checkpoint
rse evaluate measure a checkpoint against the rule-based baseline
rse render   synthesize NES audio from a MIDI file or an intake record
rse ui       launch the local web interface (needs the ui extra)
rse windows  tokenize a corpus to disk, for training elsewhere
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
  hardware/    profiles and the constraint engine     — working
  events/      canonical RSE Event Format (RSE-EF)    — working
  tokenizers/  REMI-style vocabulary and grammar      — working
  models/      decoder-only Transformer               — working
  training/    corpus, windows, training, evaluation  — working
  generation/  planning, rule-based composer, sampler — working
  renderers/   NES APU synthesizer                   — working
  exporters/   WAV and MIDI; tracker formats later   — working
  ui/          local Gradio application               — working
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
.venv/bin/python -m pytest          # 534 tests, no dataset required
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

## The interface

```bash
pip install -e ".[ui]"
rse ui                 # http://127.0.0.1:7860
```

Pick a source (rule-based composer, or a trained checkpoint), set mood, energy,
length, tempo and seed, press Generate, and you get playable audio plus a
per-channel breakdown, the hardware and loop reports, and WAV/MIDI/JSON
downloads. It works with no checkpoint at all — the rule-based composer is a
complete generator on its own — and gains the model as a second source once
one exists.

The UI is a client of `rse.compose`, which the CLI uses too, so neither can
drift from the other about what a composition is.

**It binds `127.0.0.1` and nothing else.** Generated music is yours and the
server has no authentication, so it must not be reachable from off the machine.
A non-loopback `--host` is refused unless you also pass `--allow-remote`, and
Gradio's public `share` tunnel is never used. To reach it from another machine,
forward the port over SSH rather than widening the bind.

## The generation API

The public-facing path: an HTTP service that turns a prompt and a few controls
into a finished NES soundtrack. It is the backend the browser generator calls,
and it exists so the thing a person uses is the same thing the command line
uses — both call `rse.song.generate`, so neither can produce different music
from the same request.

```bash
pip install -e ".[api]"
rse-api                      # then open http://127.0.0.1:8000
```

That is the whole generator: `rse-api` serves the page at `/` as well as the
API, so there is nothing to build and no second process. **`npm install` is not
required to use it** — it is only for editing the page with hot reload:

```bash
npm install
npm run dev                  # http://127.0.0.1:5173, proxies /api to :8000
```

The dev server serves the page only. Generation happens in Python either way,
so `rse-api` must be running for the Generate button to do anything.

On Windows, from a checkout:

```
py -m venv .venv
.venv\Scripts\activate
pip install -e ".[api]"
rse-api
```

then open <http://127.0.0.1:8000>. No dataset, no checkpoint and no model
download are needed — the procedural composer is the whole engine.

```bash
curl -X POST http://127.0.0.1:8000/api/generate \
  -H 'content-type: application/json' \
  -d '{"prompt": "A dark and dramatic boss battle theme with driving percussion.",
       "seconds": 60, "energy": 75, "loop": true}'
```

The response carries the song's links, the seed that produced it, what
performed it, and the requirement report. `GET /api/capabilities` returns the
styles, duration limits, formats, tempo ladder and instrument profiles, read
from the engine rather than duplicated in the frontend.

**Two ways in, one engine.** *Quick Generate* is settings only and involves no
language processing at all. *Describe a Song* adds a rule-based parser on top
of the same controls — it recognises styles, scenarios, moods, tempos, keys and
durations, and reports exactly what it could not use. Neither mode routes to a
model: both drive the procedural composer, asserted by test.

**Only connected controls are offered.** A control that changes nothing is
worse than a missing one. `mood` was exactly that for most of this project --
the interpreter read it into `key_mode`, which the composer never consults, so
"a dark theme" and "a bright theme" produced byte-identical output once a style
was chosen. It now selects the instrument profile. A mood or profile the engine
cannot produce is refused with a 422 rather than silently ignored.

### Sound design

Composition decides what is played; instrumentation decides which sounds play
it; expression decides how they behave. Only the first was ever worked on, and
it showed: four composition experiments in a row returned no audible
preference, while the first change to the *performance* was accepted
immediately.

Each song picks a complementary pair of pulse timbres and shapes note velocity
from musical position. Three hardware facts constrain it:

- **Three pulse timbres, not four.** The fourth duty setting is the 25%
  waveform inverted — identical in magnitude spectrum, so it is offered
  nowhere. 50% has no even harmonics (hollow), 25% is the classic lead, 12.5%
  is bright and nasal.
- **Volume is a 4-bit register**, so velocity is quantised to fifteen steps and
  velocities 98–104 all land on the same one. Dynamics are authored in register
  space, not in velocity.
- **The triangle has no volume register.** Bass dynamics do not exist on this
  hardware and are not faked.

**No model, no GPU.** A 60-second song costs about 1.2 s of CPU, 97% of it in
the audio renderer and 3 ms in composition. That is short enough to answer
inside the request, so there is no job queue and no polling: one POST returns a
finished song. The service imports neither torch nor transformers.

**Reproducing and varying.** Omit `seed` for a new variation; the response
tells you which seed it used. Send that seed back to get the same song, byte
for byte — the same one `scripts/prompt_to_song.py` produces for that seed.

**It binds `127.0.0.1` by default**, and `rse-api` refuses a non-loopback host
unless `RSE_API_ALLOW_REMOTE=1` says you meant it. A container platform routes
to the service instead, and states its own wildcard bind in
`deploy/render.yaml` where the choice is visible.

**Caps, because it is anonymous and public.** There is no authentication — the
audience is anyone who wants retro music, and a gate would lock out exactly
those people — so the controls are on cost instead: a duration ceiling
(`RSE_API_MAX_SECONDS`, default 180), a concurrency ceiling
(`RSE_API_MAX_CONCURRENT`, default 2), a per-caller token bucket
(`RSE_API_RATE_PER_MINUTE`, default 20, burst 5), and a retention window after
which generated songs are deleted (`RSE_API_RETENTION_SECONDS`, default 1800).
CORS is same-origin until `RSE_API_ALLOWED_ORIGINS` names the site; it is never
`*`.

**SNES returns 501.** Only the 8-bit NES engine exists. Sample-based
instruments, eight DSP voices, envelopes and stereo are a separate milestone,
and playing NES compositions through different instruments would not be SNES
output — so the API refuses rather than returning NES audio under an SNES
label. `GET /api/capabilities` marks it unavailable so the interface can grey
the option out.
