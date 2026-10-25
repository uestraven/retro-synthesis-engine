# Architecture decision: native RSE, with a named failure to fix

**Decision: keep the native D4 Transformer. Do not adopt a pretrained model,
and do not continue step-scaling it either.**

Both halves matter. The pretrained study did not find a better composer, and
the native study found that more training makes the native model *worse* on
the measures that distinguish music from a drum machine. The next work is
diagnostic and cheap, not another foundation run.

Evidence is from `generation_battery.json` (36 samples per checkpoint, 12 seeds
× 3 temperatures), `fermata_nes_results.json` (12 generations through the NES
arranger), and `docs/codec_comparison.md`. The official test split is still
untouched.

## The decision in one table

| option | verdict | why |
|---|---|---|
| **A — native D4 Transformer** | **adopted** | competitive with a 201M model at 5.7M params; failure is named and local |
| B — fine-tune Fermata on NES-MDB | rejected for now | representationally viable, but 77% of its vocabulary is untrained dead BPE |
| C — Fermata composer + NES arranger | retained as a component | works end to end at 0.844 retention; better rhythm, worse motif |
| continue native to 300k | **rejected** | 100k→150k moved every rhythm measure the wrong way |

## Why not more native training

100k → 150k cost 33 GPU-minutes and moved validation loss by **0.0006**
(0.5685 → 0.5679). Generation moved as follows:

| | 100k | 150k | direction |
|---|---|---|---|
| 8-bar completion | 66.7% | 83.3% | up, but see below |
| all four channels | 33.3% | 33.3% | flat |
| percussion present | 38.9% | 38.9% | flat |
| pitch-class entropy | 2.021 | 2.312 | up |
| **IOI entropy** | 0.675 | **0.387** | **down** |
| **distinct IOIs per sample** | 2.5 | **2.0** | **down** |
| **top duration share** | 0.530 | **0.674** | **down** |
| **motif participation** | 0.535 | **0.000** | **down** |

The completion gain is not what it appears. Saturated samples (below) rose
from roughly 7 to 10 of 36 while early-EOS fell 12 → 6, and a saturated run
trivially reaches eight bars. So the headline improvement shares a cause with
the regressions.

## The failure, named

**The model fills every row on every active channel.** Measured on the dense
samples:

| sample | events/bar | events/row | percussion/bar | IOI entropy |
|---|---|---|---|---|
| 100k t0.7 s0 | 64.0 | **4.00** | **16.0** | −0.00 |
| 100k t0.7 s11 | 63.8 | 3.98 | 16.0 | −0.00 |
| 150k t0.7 s0 | 64.0 | **4.00** | **16.0** | −0.00 |
| 150k t0.7 s2 | 48.0 | **3.00** | 0.0 | −0.00 |

Exact integers. A bar is 16 rows, so 16.0 percussion events per bar is one
noise hit on *every row*, and 4.00 events per row is all four channels firing
on every row. IOI entropy is zero because every inter-onset interval is
identical. That is a constant pulse at maximum density, against a corpus
median near 24 notes per bar.

This reframes an earlier positive reading of mine: the "128 percussion" samples
I had taken as the native model finally learning drums are the noise channel
saturated, not played.

### Why it happens

Next-token cross-entropy on this representation is minimised by emitting the
modal token at every opportunity, and the modal duration is extreme:
`Duration_1` is **38.6%** of all duration tokens in the training cache
(top-3 = 66.5%, entropy 3.02 bits of a possible 6.97). Meanwhile **a rest has
no token**. An empty row emits nothing at all, so sparsity is only ever the
*absence* of a prediction and receives no direct gradient. Filling rows with
short notes is the locally optimal policy, and more training finds it more
reliably.

### A measurement trap worth recording

Aggregate `Position_` entropy is 4.45 bits against the corpus 5.61 — a 1.2-bit
gap that reads as mild. It is not: **within a sample the model uses all 16
positions of every bar**, which is maximal token entropy and zero rhythmic
information. Pooled token entropy cannot distinguish "uses every position
selectively across pieces" from "uses every position in every piece".
`n_distinct_iois`, at a median of 2.0 against the corpus's far higher spread,
is the statistic that sees it.

I reported the pooled figure first and it understated the problem. Any future
rhythm claim should be made per sample.

## Native versus Fermata, on what survives scrutiny

Two columns from the earlier comparison must be struck. Fermata's percussion
and four-channel participation were **prompted and arranged**: every non-sparse
scenario planned a drum track, and `arrange_for_nes` maps roles to channels
deterministically. Those rows measured the prompt and the arranger, not the
composer.

What remains, with the corrected like-for-like references:

| metric | corpus (8-bar D4) | native 150k | Fermata → NES |
|---|---|---|---|
| IOI entropy | **1.479** (IQR 0.84–1.81) | **0.387** | **1.021** |
| motif participation | 0.509–0.754 band* | 0.000 median | 0.417 |
| notes per bar | ~24 | 24.96 | 18.41 |
| stepwise fraction | 0.370 | 0.304 | 0.536 |
| n-gram repetition | 0.393, bimodal | 0.322, bimodal | 0.385, bimodal |

\* Those bounds were fitted as motif *label* thresholds, not as a reference
distribution. They indicate a band, and a single recurrence score cannot
separate varied thematic development from repeated notes. Treat the motif row
as suggestive.

The IOI reference also needed correcting: the widely-quoted **2.358** came from
full songs of original MIDI. Like-for-like on 8-bar D4 round-trips it is
**1.479**, because the D4 representation itself costs 34% of IOI entropy. On
that basis Fermata sits inside the corpus interquartile range and the native
model is eightfold below the median and outside the range.

**So Fermata's one clear advantage is rhythmic variety**, and the native
model's is note density and motif recurrence. Neither is a better composer
overall, and the native model achieves parity at 5.7M parameters against 201M.

## Why not path B

Representationally it is viable — Phase 2 found **zero event loss** converting
NES-MDB at Fermata's 24 TPQN, with 4.47 ms onset error, below D4's own 5.70 ms
grid error. The objection is different: **7,348 of Fermata's 9,507 tokens are
`<BPE_*>` that a documented builder bug left untrained**. Fine-tuning would
inherit an embedding table and output softmax three quarters of which the model
has never seen. Its effective vocabulary is 2,159.

There is also no published encoder or decoder. The repository contains no
Python at all, so all Fermata results here rest on a reconstruction
(`rse-fermata-reconstruction/1.0.0`), validated by perplexity on the
repository's own promo samples (NLL 0.2254 correct against 10.75 shuffled and
9.22 with two fields transposed) but not by the author's code.

## What to do instead, in order

**1. Sampling and decoding.** Cheapest and untested. The saturated mode appears
at t=0.7; at t=1.0 IOI entropy reaches 1.1–2.8, inside the corpus IQR, but
output is short and sparse. A density- or bar-aware decoding constraint may
reach the middle with no retraining at all. No GPU required.

**2. Density conditioning.** The control the model lacks, and RSE's
conditioning machinery already exists for it. This is a *different* axis from
the `driving` rhythm class that failed Gate 2 at 0.316 recall: `notes_per_bar`
is directly measurable and perfectly reliable, which is precisely what makes it
a better conditioning target than a derived class. Note that Gate 3's
conditioning result was invalidated by a stale cache and must be re-run on the
frozen representation regardless.

**3. An explicit rest or row-advance token, or an auxiliary sparsity
objective.** This addresses the root cause rather than its symptom, and it is a
change to `rse-repr/1.0.0-d4k4-tempo1.01`, which is frozen. It needs an
explicit decision, and it would invalidate the current cache and foundation.

**4. Fermata as a rhythm planner.** Keep paths B and C available. Its IOI
entropy is where the native model's is not, and a hybrid that takes high-level
rhythmic or structural planning from a pretrained model while RSE arranges and
renders is the shape the evidence supports — not a replacement.

## Not recommended

- **More capacity.** Nothing here indicates the 5.7M model is too small. It
  matches a 201M model on density and beats it on motif recurrence; it fails on
  a specific pathology that scaling would likely sharpen, since more training
  already did.
- **More steps at the same objective.** Measured, twice.
- **Replacing the native architecture.** No evidence supports it.

## Costs, for the record

| | |
|---|---|
| native 100k | 3,883 s on an RTX PRO 4500, ~$0.78 |
| native 100k→150k | 1,969 s, ~$0.39 |
| Fermata inference | CPU only, 17–34 tokens/s, ~60 s per 8-bar sample |
| Fermata checkpoint | 403 MB, Apache-2.0 on weights and card |
| Phase 1–4 study | no GPU |

## Open and unresolved

- **Contamination.** Unchecked for all three candidates. Fermata's disclosed
  corpora (PDMX, KernScores, OpenScore Lieder, The Session) make NES-MDB
  overlap unlikely but unproven; Moonbeam's paper names its sets, MetaMIDI
  among them, where game music is plausible. A memorization screen is a
  prerequisite to any published quality claim, and the evaluation already has
  corpus fingerprinting for it.
- **13% of train pieces have no fitted pulse** and are encoded on the legacy
  grid. Independent of everything above, and untouched by the tempo widening.
- **Fermata never terminates.** All 12 generations were cut off by the token
  budget, with bar counts from 4 to 19. Its weakness is control; the native
  model's is content. That asymmetry is the strongest argument for a hybrid.
