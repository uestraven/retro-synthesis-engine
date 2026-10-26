# NES-MDB → Fermata representation audit

120 composer-disjoint pieces from the train and valid splits (test reserved),
through the existing Fermata encoder and decoder. No GPU, no training.

**Verdict: conditionally suitable.** Fermata's vocabulary loses nothing
relative to the frozen D4 representation. The losses that exist are in the
NES → score conversion that precedes it, and in sequence length.

## The arithmetic objection was wrong, and it was mine

I claimed encoding into Fermata "discards about half the sub-row precision"
because a D4 slot is 1.5 Fermata ticks. Verified computationally: all 16 D4
positions per quarter map to **16 distinct** ticks
(0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23), the mapping is
injective, and **every slot recovers exactly**. The cost is displacement
alone — at most half a tick, 10.4 ms at 120 BPM, 16.5 ms at 76.

A correct division led to a wrong musical conclusion that would have argued
against this milestone over a non-problem.

## Arm B: what Fermata's vocabulary costs relative to D4

| tolerance | recall | missing | spurious |
|---|---|---|---|
| 0 ms | 0.0066 | 102,524 | 102,524 |
| 5 ms | 0.5937 | 41,489 | 41,489 |
| **10 ms** | **1.0000** | **0** | **0** |
| 25 ms | 1.0000 | 0 | 0 |
| 50 ms | 1.0000 | 0 | 0 |

**Perfect recall at 10 ms, on every channel.** Not one event lost, not one
invented, across 120 pieces. Onset error median 4.24 ms, worst-piece median
5.70 ms — well inside the half-tick bound. Duration error 6.4–7.8 ms per
channel against the 811 available duration tokens.

The 0 ms row is the displacement, not loss: matching at exact equality fails
because onsets move by a few milliseconds, which is why the tolerance is
stated beside every number. An event moved 4 ms is not a missing event.

## Arm A: total fidelity against the original NES events

| tolerance | recall | missing | spurious |
|---|---|---|---|
| 10 ms | **0.9232** | 11,227 | **0** |
| 25 ms | 0.9232 | 11,227 | 0 |

**Nothing is invented at any tolerance.** 7.7% of original events do not
survive the full chain, and since arm B is lossless, that loss is in the
NES → score conversion ahead of Fermata, not in Fermata's vocabulary.

Per channel, at 25 ms:

| channel | pieces | recall | onset | duration | velocity |
|---|---|---|---|---|---|
| P1 | 117 | 0.9231 | 4.22 ms | 0.54 ms | 2 |
| P2 | 118 | 0.9115 | 4.15 ms | 0.57 ms | 3 |
| **TR** | 107 | **1.0000** | 4.15 ms | 1.93 ms | 3 |
| **NO** | 78 | **1.0000** | 4.13 ms | 1.00 ms | 3 |

Triangle and noise are lossless. The loss is confined to the two pulse
channels, which is where NES composers share one timbre between melody and
counter-melody and where simultaneous notes have to be reduced. Velocity
quantisation to 16 levels costs 2–3 units — percussion and bass keep their
accents.

Spread: median recall 0.9232, p10 0.7586, min 0.5408, with **19 of 120 pieces
below 0.80**. Dense pulse writing is where it hurts.

## Two findings that are properties of the representation, not bugs

**Channel identity does not survive on its own.** Fermata's `<TRK_n>` carries
a GM program, not a role. `decode` therefore gives every pitched track the
same role, and `score_to_composition` — which assigns channels *by role* —
put all three pitched voices on P1: 302 reconstructed events against 151
original, with P2 and TR empty, and recall stuck at 0.34. Restoring channels
from the program numbers fixed it. **106 of 120 pieces needed two roles
corrected**, 13 needed one, 1 needed none. Any fine-tune must carry the
program mapping explicitly; the round trip will not recover the four NES
roles by itself.

**Sequence length is the real constraint.** Median 1,740 tokens, p95 9,996,
max 208,674.

| over | pieces |
|---|---|
| 1,024 | 83/120 (69%) |
| 2,048 | 56/120 (47%) |
| 4,096 | 32/120 (27%) |
| 8,192 | 8/120 (7%) |

Nearly half the corpus exceeds 2,048 tokens. Whole-score training is not
viable without windowing, which the project already does for D4.

## Recommended pilot, if authorised

Smallest useful LoRA:

- **Windowed examples, not whole scores.** 8-bar windows at bar boundaries,
  matching the existing D4 cache policy, which keeps sequences near the
  median and sidesteps the 47% over 2,048.
- **Carry the program mapping in the training data** (80/81/38 + drums), since
  roles are otherwise unrecoverable.
- **Composer-disjoint splits**, using the corpus's own, test untouched.
- **Judge on listening, not validation loss.** The native Transformer's loss
  plateaued at 0.5685 → 0.5679 while its rhythm got worse.
- **Benchmark: 60% NES-game-music pass rate (12/20)**, with Best/Good 5/20 as
  a stricter secondary. Not the superseded 25%.

## One measurement not trusted

The `d4_vs_original` cross-comparison returns recall 0.057 with sane-looking
8–15 ms onset errors, which does not add up. It is flagged in the code as not
quotable. Arm B already isolates Fermata's cost relative to D4, which is the
attribution question that field was meant to answer, so nothing in this
report depends on it.

An earlier version of this audit also hard-coded `GridFit(step_ticks=120)`
while the corpus carries `ticks_per_beat=22050`, making a 1.719 s piece report
63.599 s — a 37× error whose loss would have been blamed on Fermata. The fit
is now derived per composition.
