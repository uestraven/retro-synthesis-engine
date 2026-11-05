# A SNES music database — design, shelved 2026-11-05

**Status: SHELVED.** Nothing here is built. Written down while the reasoning
was fresh so picking it up later does not mean re-deriving it. Read §7 before
§1: the first question is whether to do this at all, and the answer today is
"not yet, and probably only in the reduced form."

The goal would be the SNES counterpart to NES-MDB: a corpus of symbolic SNES
music in RSE's own event schema, extracted from SPC files rather than
transcribed by hand.

---

## 1. Why NES-MDB was mechanical and this is not

NES-MDB works because the 2A03 gives away everything for free. Emulate the
NSF, log the register writes, and you are done — no per-game knowledge enters
the pipeline at any point.

| what NES-MDB relies on | the 2A03 | the S-DSP |
|---|---|---|
| channel identity | **5 fixed roles** — pulse 1, pulse 2, triangle, noise, DMC, the same in every game | **8 generic voices**, no fixed role, assignment varies per game and can change mid-song |
| timbre vocabulary | **closed set of 4** duty values, meaningful across games | **open set** — "BRR sample #N in this game's ARAM", an arbitrary index with no cross-game meaning |
| pitch | a period register, **one universal formula** to Hz | a 14-bit **resample ratio** relative to the sample's own base frequency, which lives in the game's private instrument table |
| time grid | **60.0988 Hz frames**, universal; music advances on frames | per-driver tick rate; N-SPC, AKAO, Rare, Konami, Capcom, Hudson all differ |

So the four things NES hands over as constants are, on SNES, either per-game
or absent. That is the whole difficulty, and it is why no SNES equivalent of
NES-MDB has been published while NSF datasets are commonplace.

## 2. Source data

`snesmusic.org` — 1,519 SPC sets covering 2,631 of 3,127 known games, plus a
whole-archive torrent.

An SPC is **not a score**: it is a 64 KB ARAM dump plus SPC700 and DSP
register state — a snapshot, with no note list anywhere in it. Symbolic data
has to be recovered by emulating and observing.

**Licensing is narrower than NES-MDB's.** The site's disclaimer: files are
copyrighted by their owners, "for educational and personal non-profit use
only," offered "as fair use samples for preservation purposes only." NES-MDB
is at least explicitly research-licensed. Anything ingested gets
`use_class: research_only` at minimum, never enters the public repo, and never
ships in a product artifact. See the existing NES-MDB constraints — the same
rules apply more strictly, not less.

## 3. Pipeline

Three of the five stages already exist in this repository.

```
SPC files ──┐
            ├─(2) emulate + log DSP writes ──▶ per-voice event log
            │     blargg snes_spc                (t, KON/KOFF, SRCN,
            │     [BUILT AND VALIDATED]           PITCH, VOL L/R, ADSR, ENVX)
            │
            ├─(3) recover absolute pitch ──▶ note numbers
            │     f0 from the BRR sample itself
            │
            ├─(4) recover the grid ──▶ rows
            │     onset-interval histogram
            │
            ├─(5) assign voice roles ──▶ melody / bass / comp / percussion
            │     clustering [THE RESEARCH RISK]
            │
            └─(6) emit ──▶ rse Composition
                  composition_from_intake
```

**(2) Emulate and log — driver-agnostic, no per-game work.**
blargg's `snes_spc` is already built and validated against RSE's BRR codec to
0.011–0.020% relative RMS (see `research/README.md`, the BRR overflow finding;
harness was in `/tmp/snes_oracle/`, rebuild from the clone). Add DSP-write
logging to its run loop and emit, per voice:
`(t, KON/KOFF, SRCN, PITCH, VOL L/R, ADSR, ENVX)`. This is the structural
analogue of NES-MDB's `exprsco` and needs nothing game-specific. **LGPL-2.1 —
keep it as an external binary, never linked into RSE.**

**(3) Recover absolute pitch — the trick that avoids instrument tables.**
Don't try to read the game's base-frequency table. Decode each ARAM-resident
BRR sample and estimate its fundamental directly (autocorrelation / FFT), then

    note = f0_sample · PITCH / 4096

`rse/snes/samples.py` already has periodicity and partial analysis to reuse.
Percussion and noise will fail f0 estimation; label them unpitched rather than
forcing a number.

**Optional shortcut:** skip (3) entirely if transposition-invariant melody is
acceptable. PITCH *ratios* within a voice give exact intervals with no base
frequency at all — enough for most analysis, and it removes the stage's whole
failure mode.

**(4) Recover the grid — already solved for NES.**
Histogram the key-on inter-onset intervals; the driver's tick period falls out
as the dominant spacing. `rse/events/pulse.py:222` (`detect_pulse`) already
makes this move on NES compositions. Driver-agnostic.

**(5) Assign voice roles — the genuinely new part, and the soft spot.**
This replaces the 2A03's fixed channel semantics, which is the one thing with
no SNES equivalent. Cluster voices by pitch range, sample length, loop
presence and envelope shape: low + sustained → bass; short + unlooped +
unpitched → percussion; the rest by register and density. Expect this to be
mushy, and expect it to be where the corpus's quality is ultimately judged.

**(6) Emit** into `composition_from_intake` (`rse/events/convert.py:51`) so
the result lands in the same event schema as the NES corpus and every
downstream consumer — windowing, analysis, the renderer — works unchanged.

## 4. The validation gate

This is the part that would make the corpus trustworthy, and it is cheap:

> Re-render each extracted score with RSE's SNES renderer and compare against
> `snes_spc`'s own audio output for that SPC.

Automatable per track, and it yields a real quality score per extraction
rather than a global hope. Note *why* the reference must be the emulator: this
project has already shipped a BRR codec validated only against its own
decoder, which hid a 67% hardware wrap rate through three rounds of green
tests and a full release validation. A dataset validated against RSE's own
assumptions would repeat that mistake at corpus scale.

## 5. Prior art worth not duplicating

- **VGMTrans** (zlib) — supports the major SNES drivers (N-SPC, AKAO, Rare,
  Konami, Capcom, Hudson, Chunsoft…) and exports MIDI **plus** instrument data
  as SF2/DLS. Does this well, per-driver, GUI-batch with no documented CLI.
- **`loveemu/spc_converters_legacy`** (MIT) — 14 per-driver SPC→MIDI tools
  (`akaospc`, `nintspc`, `rarespc`, `konspc`, `capspc`, `hudspc`…), explicitly
  "not 100% accuracy," superseded by VGMTrans.

So "SPC to notes" is not novel. What would be novel is being
**driver-agnostic** (steps 2–4 need no driver identification at all) and the
**role assignment** in step 5. If the goal is only a few dozen games, using
VGMTrans by hand is strictly cheaper than building this.

## 6. Effort

| stage | effort | risk |
|---|---|---|
| (1) acquire | hours | licensing, not technical |
| (2) emulate + log | days — emulator exists and is validated | low |
| (3) f0 recovery | days | medium; skippable via the ratio shortcut |
| (4) grid detection | days — the NES version exists | low |
| (5) role assignment | weeks | **high, and won't come out clean** |
| (6) emit + validate | days | low |

## 7. Should this be built — read this first

**Not for model training.** That path is closed: a clean retrain beat the old
checkpoint by 11.4% on validation loss and still produced "barely music," and
JRPG-only fine-tuning was worse. The blocker was never data volume, corpus
purity, or training hygiene — all three were addressed — it is that next-token
loss does not optimise for anything a listener values. A SNES corpus does not
change that argument; it just makes it more expensive to re-test.

**Possibly for deriving rules — but note the record.** Composition-rule
changes derived from measurement are 0 for 4 with this listener: harmonic
rhythm (24 pairs, null), harmonic vocabulary (24 pairs, null),
key/tempo/form/arrangement (null), phrase structure (dropped undecided). The
one axis that has ever landed is how material is *played* — per-song duty and
position-shaped velocity were accepted immediately. So if this is built for
analysis, extract **performance** conventions (voice counts over time,
velocity shaping, ADSR, echo usage, note-length distributions) before
progressions and form lengths.

**And SNES is not currently shipped** — the deployed API runs
`RSE_API_CONSOLES nes` on a free plan, because a 60 s render peaks at 340 MB
against a 512 MB ceiling. Better SNES material improves a console no user can
reach today.

### The reduced version, which is what to actually do first

Build **step 2 only**, point it at **20–50 SPCs**, and report the performance
conventions. That answers the live question — what real SNES music does with
its eight voices, its envelopes and its echo — for days of work instead of
weeks, with no role assignment, no f0 recovery, and no corpus to license,
store or maintain. Build steps 3–6 only if that sample surfaces something
worth acting on.

## 8. Loose ends held over

- The 512-entry S-DSP gaussian interpolation table is in hand.
  `rse/snes/sampler.py:9` notes RSE uses Catmull-Rom and does not have it.
  Swapping it in is independent of this initiative and would re-pin the SNES
  baselines — needs an A/B, since the real gaussian is a known low-pass and
  "more authentic" is not the same as "preferred."
- `hardware_overflow=True` exists on `encode`/`decode`/`decode_looping` and
  takes hardware wraps to zero for ≤2.2 dB of round-trip accuracy. Default is
  still the saturating path so baselines hold. Flipping the default is a
  prerequisite for any real-hardware target.
- `snesmod` (Impulse Tracker → SPC) and `snesgss` are the export path if
  real-hardware playback ever becomes a goal, and are reference material for
  the `UNMODELLED` allocation classes in `rse/snes/aram.py` that keep
  `hardware_budgeted` False.
