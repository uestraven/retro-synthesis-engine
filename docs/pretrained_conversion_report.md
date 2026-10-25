# Phase 2: conversion fidelity through a pretrained model's grid

Deterministic, no GPU, frozen D4 codec untouched. Reproduce with

```bash
python scripts/transfer_audit.py --split train --limit 400 --out transfer_audit_train.json
python scripts/transfer_audit.py --split train --limit 400 --drum-key-shift 1 \
  --out transfer_audit_train_shift1.json
```

Scale: **400 train pieces, 827 windows, 223,925 events**; 39 pieces had no
fitted row. Validation is reserved for confirmation and the test split is
untouched.

The reference is the **D4-decoded composition**, not the raw MIDI, because D4
is what RSE actually holds and renders. The loss measured is therefore the
external representation's alone.

## Headline: I was wrong about the grid

Phase 1 argued from grid arithmetic that a 24-TPQN tokenizer injects up to
**11.1 ms** of onset error and therefore made fine-tuning (path B) inferior.
That used the worst case as though it were typical. Measured:

| TPQN | multiple of beat/16? | event recall | onset MAE | onset p95 | duration MAE | collisions | density in → out |
|---|---|---|---|---|---|---|---|
| 8 | no | **0.923** | 12.05 ms | 23.83 ms | 17.11 ms | **17,221** | 181 → 178 |
| **24** (Fermata) | no | **1.000** | **4.47 ms** | 8.78 ms | 5.99 ms | **0** | 181 → 181 |
| 48 | yes | 1.000 | 2.32 ms | 4.37 ms | 0.37 ms | 0 | 181 → 181 |

**At Fermata's actual resolution, nothing is lost.** Recall is 1.000 across
every event class and every channel, there are no onset collisions, and
melodic density is identical. The 4.47 ms mean onset error is *smaller than
D4's own 5.70 ms grid error* — the quantization is below the noise floor of
the representation it is converting from.

So the grid does not argue against path B. My Phase 1 conclusion on that point
is withdrawn.

Resolution does still matter, and 8 TPQN shows what a genuinely coarse grid
costs — and costs selectively:

| TPQN | attack | arpeggio | vibrato | retrigger | passage |
|---|---|---|---|---|---|
| 8 | 0.999 | **0.934** | 0.998 | 0.949 | **0.902** |
| 24 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

At 8 TPQN the losses fall on exactly the hardware-specific material — arpeggio
frames and fast passages — while plain attacks survive. That is the shape of
loss Phase 2 was built to detect, and 24 TPQN simply does not exhibit it.

## The real finding: the drum defect destroys all percussion

Fermata's model card documents that `midi-unpitched` values are 1-based but
were read as 0-based, so drum notes come out one GM key high. Modelled
explicitly:

| | event recall | **noise-channel recall** | percussion events lost |
|---|---|---|---|
| no shift | 1.000 | 1.000 | 0 |
| **+1 GM key** | **0.737** | **0.000** | **58,991** |

**Every percussion event dies**, at every resolution. Percussion is 27.2% of
NES-MDB note-ons, so overall recall falls from 1.000 to 0.737 from this one
off-by-one — far more damage than any grid choice.

It is correctable (subtract 1 at read time, as the card says), so this is an
integration requirement rather than a barrier. But it must be corrected
*explicitly*, and the failure mode deserves emphasis: **a naive integration
would silently produce NES music with no percussion at all** — which is
indistinguishable, from the output alone, from the zero-noise behaviour the
native foundation showed at 10k steps. Two unrelated causes, identical
symptom. Had we integrated before testing, we would have blamed the model.

### A mapping choice that makes the defect fail loudly

The 16 noise periods map to GM drum keys **spaced two apart** rather than
adjacent. Under a +1 shift:

| mapping | exact | silently wrong period | detectably lost |
|---|---|---|---|
| adjacent (kick, snare, toms…) | 0 | **15** | 1 |
| **spaced by 2** | 0 | **0** | **16** |

Adjacent keys would have made the shift land on another valid entry, so 15 of
16 periods would come back as a *different drum* with nothing reporting a
problem — invisible to every onset and duration metric in this report. Spacing
converts that into 16 detectable losses. The cost is that a GM-trained model
sees some less-common kit pieces; while the shift is unfixed, that is the
right trade.

## Losses, kept separate

| source | magnitude at 24 TPQN | notes |
|---|---|---|
| quantization | 4.47 ms onset, 5.99 ms duration, **zero events** | below D4's own 5.70 ms grid error |
| expressiveness | **not yet isolated** | see below |
| arrangement | **not measured** | needs a general polyphonic score, i.e. Phase 4 |

The expressiveness column is the honest gap in this report. The round trip
preserves monophony, channel identity and arpeggios-as-sequences **because
this conversion was written to preserve them** — `Track.monophonic`,
`Track.channel_id` and `ScorePercussion.source_value` carry exactly the
information a general representation would drop. What is not yet established
is whether Fermata's own tokenizer preserves them when *it* reads a score:

- Would it keep three rapid pitches on one track as a sequence, or re-voice
  them as a chord across tracks? A chord implies polyphony the APU cannot play.
- Does its bar-major interleaving survive a track that is silent for most of a
  bar, as the noise channel often is?
- Does a GM program number survive round-tripping, or is track identity
  inferred from content?

Those need the actual checkpoint and tokenizer, which Phase 4 provides. Until
then, this report establishes only that the **grid** is adequate — not that
Fermata's representation is.

## Rendered comparisons

`transfer_audio/` holds three 8-bar excerpts, each as `d4_reference`,
`tpqn08`, `tpqn24`, `tpqn48`, all through the same RSE renderer at the same
musical timing, so a difference is the conversion and not the synthesis.
Expect `tpqn24` and `tpqn48` to be indistinguishable from the reference and
`tpqn08` to be audibly looser. 23 MB, kept out of the repo under the
research-only corpus rule.

## Contamination uncertainty

Unresolved for every candidate, and the report should not imply otherwise.

- **Moonbeam**: its paper's Appendix A does name the pretraining sets
  (AriaMIDI, MetaMIDI, MAESTRO and others), correcting the Phase 1 claim that
  provenance was undisclosed. Some carry non-commercial source licences.
  Composition-level overlap with NES-MDB remains unchecked, and MetaMIDI in
  particular is a broad scrape where game music is plausible.
- **MIDI-GPT**: GigaMIDI is named but not audited here.
- **Fermata**: PDMX, KernScores/Humdrum, OpenScore Lieder and The Session are
  classical, choral and folk notation, which makes overlap *unlikely* — but
  disclosed sources are not proof of zero overlap, and no composition-level
  check has been run.

A memorization screen against NES-MDB is a Phase 4 prerequisite, not an
optional extra, because the existing evaluation already has
`MEMORIZATION_THRESHOLD` and corpus fingerprinting for exactly this.

## What this changes

**Path B is back in contention.** The grid objection does not hold: NES-MDB
converts into a 24-TPQN representation with zero event loss and sub-D4 timing
error, so fine-tuning would not be training on a degraded corpus.

**Path C remains preferable for a different reason** — Fermata has never seen a
chiptune, so its value is as a composer of musical ideas, not as a source of
NES arrangement patterns. But that is now a judgement about musical content
rather than a measured representational barrier, and Phase 4 is what tests it.

**The decision should wait for two things**: the native 100,000-step baseline,
and an unmodified Fermata inference prototype. Neither is available yet.
