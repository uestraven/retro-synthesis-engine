# Pretrained symbolic music models: feasibility audit

Phase 1 of the pretrained-model study. Verified against the live sources on
2025-10-25; every claim below is either quoted from the source or marked as
not stated there. No GPU was provisioned and nothing was downloaded.

The question this phase answers is narrow: **which candidate, if any, is
practical enough to prototype against?** It does not claim any of them compose
better NES music — that needs Phase 4.

## Candidates

### Moonbeam MIDI Foundation Model

| | |
|---|---|
| weights | `moonbeam_309M.pt`, `moonbeam_839M.pt`, plus a CommU LoRA adapter |
| size | 2.39 GB repo total; parameter counts inferred from filenames, not stated |
| licence | Apache-2.0 on both code and the Hugging Face weights repo |
| pretraining | "81.6K items, 18 billion tokens" of unnamed MIDI |
| fine-tuning | LoRA documented: `--use_peft True --peft_method lora` |
| representation | "a novel domain-knowledge-inspired tokenization"; no field-level detail published |

**The GitHub README's checkpoint links are `https://TODO` placeholders**, which
initially looked like a blocker. The files do exist on Hugging Face — only the
README is unfinished. The Hugging Face model card is empty.

The real problem is **provenance**. The pretraining corpus is not named, so we
cannot check whether it overlaps NES-MDB. Without that, any apparent quality
advantage on NES-style material is unfalsifiable: it could be recall. That
matters more here than for most transfer tasks, because NES-MDB is itself a
widely mirrored MIDI collection.

### MIDI-GPT

| | |
|---|---|
| weights | `yellow_small`, `yellow_medium`, `prism_medium`, `expressive_medium` |
| licence | code MIT; **weights under the Metacreation Non-Commercial Research License** |
| pretraining | GigaMIDI |
| context | 2048 tokens (`yellow`), 4096 (`prism`, `expressive`) |
| controls | note density, polyphony min/max, note duration min/max, key, pitch range, silence, pitch-class set, genre |
| multitrack | yes, with track generation and infilling |
| drums | yes, attributes tagged melodic / drum / both; `note_density` is drum-only on `yellow` |

The controls are the most relevant of any candidate. **Note density and
polyphony bounds map almost directly onto the two failures we measured** — the
density overshoot and the channel collapse — so a prompt could ask for
NES-plausible density rather than our having to post-hoc correct it.

Two caveats. The weights are **non-commercial research only**, and a
fine-tuned derivative does not escape that; the user's own brief is explicit
that this makes it a research candidate, not a production dependency.
And **GigaMIDI's contents must be checked for NES-MDB overlap** before any
quality claim — it is a large aggregation and game music is plausible in it.

### Fermata 0.2B

| | |
|---|---|
| weights | 201,398,272 parameters, bf16, Apache-2.0 on weights *and* card |
| architecture | 17 layers, d_model 1024, 16 query / 4 KV heads, RoPE, RMSNorm, SwiGLU |
| context | 4096 max, trained on 2048-token windows (~35 bars) |
| representation | `tokenizer_v2`, bar-major REMI+, vocabulary 9507, **24 TPQN** |
| multitrack | yes, GM program per track, bar-interleaved so tracks advance together |
| drums | yes, GM percussion numbers with velocity |
| tasks | GENERATE, CONTINUE (12/12), INFILL (12/12) |
| pretraining | ~1.18M windows from PDMX, KernScores/Humdrum, OpenScore Lieder, The Session |

Fermata is the only candidate with **both permissive weights and fully
disclosed training data**. Its corpus is classical, choral and folk notation —
so **no plausible NES-MDB overlap**, which makes a quality comparison
interpretable in a way the other two are not.

It is also the only one that documents its own defects, which is a point in
its favour rather than against:

1. **BPE merges were never applied** — a builder bug meant training ran on the
   atomic token stream.
2. **Drum pitches are shifted +1 GM key** — `midi-unpitched` is 1-based and was
   read as 0-based, so kicks and snares can render as neighbouring kit pieces
   or whistles. Subtracting 1 at render time recovers most of it.
3. Dissonance 0.29 against ~0.15 for real scores; in-key ratio 0.81 against
   ~0.96.

The flip side of its clean provenance is that **it has never seen chiptune**.
Classical and folk notation is a long way from NES idiom: no arpeggiated chords
standing in for polyphony, no noise-channel percussion, no register-write
articulation.

## The finding that constrains all three

**None of these tokenizers can express our onset grid exactly.**

RSE's frozen representation places onsets on `beat/16` — four tracker rows per
beat, four sub-row slots per row. Mapping that onto a tokenizer with `T` ticks
per quarter note is exact only when `T` is a multiple of 16:

| tokenizer | TPQN | exact? | worst onset error at the corpus median 112.87 BPM |
|---|---|---|---|
| common MidiTok default | 8 | no | **33.2 ms** |
| MidiTok fine | 12 | no | 22.2 ms |
| **Fermata `tokenizer_v2`** | **24** | **no** | **11.1 ms** |
| (hypothetical) | 48 | yes | 0 ms |

For scale: Branch B's whole purpose was reducing grid onset error, and D4
achieved a **5.70 ms** median against codec A's 15.57 ms. Round-tripping
through a 24-TPQN tokenizer injects up to **11.1 ms** — roughly double the
error we just spent the entire branch removing, and on half of all onsets,
since `beat/16` lands on half-integer 24-TPQN ticks.

This does not rule out the approach. It does mean the honest architecture is
**C, not B**: use the pretrained model as a *composer* whose output is
re-quantized onto our grid, rather than fine-tuning it on NES-MDB converted
into its coarser representation. Path B would train the model on a degraded
copy of the corpus and then generate music we degrade again on the way back.

Phase 2 must measure this rather than assume it — the figures above are
arithmetic on the grids, not a measured round trip of real pieces.

## Scored comparison

Weighted for what actually decides this: interpretable comparison first
(provenance), then licence headroom, then the controls that address our
measured failures.

| criterion | weight | Moonbeam | MIDI-GPT | Fermata |
|---|---|---|---|---|
| weights obtainable now | 3 | 3 | 2 (gated, licence accept) | 3 |
| licence permits eventual commercial use | 3 | 3 | **0** | 3 |
| training-data provenance disclosed | 3 | **0** | 2 (GigaMIDI, needs overlap check) | 3 |
| NES-MDB overlap checkable | 2 | 0 | 1 | 3 |
| representation documented | 2 | 0 | 1 | 3 |
| onset resolution vs our grid | 2 | ? | ? | 1 (24 TPQN, inexact) |
| multitrack + drums | 2 | ? | 3 | 3 |
| controls matching our failures | 2 | 0 | **3** | 1 |
| fine-tuning path documented | 1 | 3 | 2 | 1 |
| defects disclosed | 1 | 0 | 1 | 3 |
| **weighted total** | | **27** | **38** | **54** |

## Recommendation

**Prototype against Fermata 0.2B first.** Not because it is the strongest
composer — it is the smallest candidate and has never seen a chiptune — but
because it is the only one where a result would *mean* something: permissive
weights, disclosed corpus with no NES-MDB overlap, documented representation,
and self-reported defects we can correct for rather than discover later.

**Keep MIDI-GPT as the control on controllability.** Its density and polyphony
bounds target our two measured failures directly, and it is worth running even
under a non-commercial licence to learn whether explicit density control is the
missing ingredient. Treat anything it produces as research evidence only.

**Deprioritise Moonbeam** until its pretraining corpus is named. It is the
largest and has the cleanest fine-tuning story, but an unfalsifiable quality
claim is not worth GPU time ahead of the other two.

## What this phase does not establish

- That any candidate composes better than the native model. The native
  100,000-step baseline is still running and is the control.
- That the conversion survives. The 11.1 ms figure is grid arithmetic, not a
  measured round trip — that is Phase 2.
- Fermata's drum defect interacts with the NES noise channel in a way nobody
  has tested: GM percussion numbers must map onto 16 noise periods, and a +1
  key shift could silently change which percussion sound is chosen.
