# Tokenizer, Model and Evaluation

Phase 4. `rse-vocab/1.0.0`, `rse-checkpoint/1.0.0`.

## The honest headline

**The training pipeline works end to end. No model has been trained on real
NES music.** NES-MDB is not in hand, so the only corpus available is output
from the Phase 3 rule-based generator.

Training on procedural output proves that tokenization, windowing, training,
checkpointing, sampling and evaluation are all correct and connected. It
proves nothing about musical quality, because the model's ceiling is a
rule-based generator whose limitations are already documented. Any numbers
from that corpus should be read as a pipeline test and labelled as one —
`CorpusInfo.synthetic` is set to `True` so reports say so themselves.

## Tokenizer

REMI-inspired, with the families a general MIDI tokenizer lacks (proposal 6.2).
Around 200 tokens, which is the point: a General MIDI vocabulary would spend
hundreds on programs the NES cannot play.

```
<bos> Hardware_nes Mood_dark Energy_medium Tempo_120 Length_8 Loop_start
  Bar Position_0 Channel_P1 Pitch_81 Velocity_7 Duration_3
                 Channel_P2 Pitch_69 Velocity_7 Duration_3
                 Channel_TR Pitch_45 Velocity_7 Duration_3
                 Channel_NO Noise_13 Velocity_7 Duration_1
      Position_2 Channel_NO Noise_2  Velocity_4 Duration_1
  Bar ...
Loop_end <eos>
```

| Family | Contents |
|---|---|
| Structure | `Bar`, `Position_0..15`, `Loop_start`, `Loop_end` |
| Conditioning | `Hardware_*`, `Mood_*`, `Energy_*`, `Tempo_60..220`, `Length_4..32` |
| Channel | `Channel_P1/P2/TR/NO` |
| Value | `Pitch_21..108` for pitched channels, `Noise_1..16` for unpitched |
| Expression | `Velocity_0..7` (bucketed), `Duration_1..32` (in grid steps) |

`Pitch_*` and `Noise_*` are disjoint families, and the grammar refuses a pitch
token on an unpitched channel. The separation that RSE-EF enforces in the data
is enforced again in the vocabulary.

A `Bar` token is emitted for **every** bar including empty ones, so bar count
and phrase position are recoverable from the token stream alone.

### Where quantization becomes destructive

The intake stage treats quantization as additive and never overwrites source
ticks (proposal 5.3). Tokenization cannot: a model needs a finite vocabulary,
so note starts snap to the 16th grid, durations snap to whole steps, and
velocity is bucketed eight ways. This is a deliberate change of contract, and
`encode` reports what it cost:

```text
n_tokens 753   n_events 168   n_encoded 168   n_dropped 0
max_start_error_ticks 0   mean_start_error_ticks 0.0
n_duration_clamped 0   n_out_of_range 0
```

Round-trip fidelity is checked against an explicit budget — one grid step of
timing, one step of duration, one velocity bucket — using the same `Tolerance`
machinery as Phase 2.

### Controllers are not tokenized

The data preserves controllers; the first model does not read them.
`include_controllers=True` raises `NotImplementedError` rather than silently
doing nothing. The reason is sequence length: a composition's controller
stream dwarfs its note stream, and velocity already carries the main
expressive axis. This is the first thing to revisit if generated music sounds
dynamically flat.

## The grammar

One state machine, used by both the decoder and the sampler. A sampler that
could emit sequences the decoder cannot read would produce silent corruption,
and two hand-written state machines would drift apart.

For sampling it doubles as the cheapest possible hardware awareness — the kind
proposal 3.2 wants *during* generation rather than after it. A small model will
happily emit two pitches in a row; refusing to sample one costs nothing.
`test_grammar_constrained_output_always_decodes_cleanly` asserts that even an
*untrained* model produces a fully decodable sequence.

`--unconstrained` turns the mask off, which is how you see what the model has
actually learned rather than what the grammar is propping up.

## Model

A plain decoder-only Transformer: learned positional embeddings, pre-norm
blocks, GELU, tied output weights. Nothing here should be interesting — the
interesting parts of this project are the tokenizer and the hardware
constraints.

Size is a config value saved with the checkpoint, not a fixed decision
(proposal 6.1). A checkpoint carries its weights, model config **and**
tokenizer vocabulary, because a model without the exact vocabulary it was
trained on is not usable; loading refuses a vocabulary that does not match the
one rebuilt from its own config.

Tested properties worth naming: the model cannot see the future
(`test_the_model_cannot_see_the_future`), and a real position's output does not
depend on what sits in the padding (`test_padding_is_excluded_from_attention`).
The second needed a hand-built mask, because `scaled_dot_product_attention`
will not combine `is_causal` with an explicit mask.

## Training

Everything proposal 6.3 asks for: train/validation separation, checkpointing,
gradient accumulation, learning-rate warmup and cosine decay, validation-loss
monitoring, best-checkpoint selection, early stopping, reproducible seeds, and
a logged config. Mixed precision when CUDA is present.

Independent of the UI and of any particular corpus, so the same command runs
on a laptop CPU or a rented GPU (proposal 8.4).

Two details that matter:

- **Validation loss is measured on a seeded sample** of windows, so it is
  comparable across runs rather than jittering with whichever windows were
  drawn.
- **Splits are assigned by composition, never by window.** Windows from one
  composition overlap, so a window-level split would put near-identical
  sequences in train and validation and make the validation loss a lie. For a
  real dataset the splits come from the release's own index, because NES-MDB's
  are composer-disjoint and replacing them would invite exactly the
  memorization the originality screening looks for.

## Evaluation

Proposal 11.2 lists six measurable qualities. Validation loss lives in the
trainer; the rest are proxies, and each docstring says what it actually
measures — the gap between "in-scale note fraction" and "sounds harmonically
coherent" is where this kind of evaluation misleads people.

| Quality | Proxy |
|---|---|
| Harmonic consistency | Best-fitting key over 12 tonics × 3 scales, and the share of lead notes inside it; pitch-class entropy |
| Repetition balance | Share of bars repeating an earlier bar; unique-bar share; transposition-invariant melodic n-gram repetition |
| Musical coherence | Stepwise vs large-leap interval fractions |
| Loop quality | The Phase 3 loop validator, musical and acoustic |
| Originality | Max Jaccard against corpus melodic fingerprints, reusing the Phase 1 de-duplication machinery |
| Technical | Hardware compliance via the Phase 2 constraint engine; grid alignment |

Originality screening closes MVP criterion 10 for *model output* rather than
just for the dataset: a sample at or above 0.6 similarity to the corpus is
flagged for review.

### The comparison that matters

Proposal 11.3 makes beating the rule-based generator a requirement, not a
courtesy. `rse evaluate` samples from the checkpoint and from the baseline,
measures both, and prints the deltas side by side.

It deliberately produces **no single verdict**, because these proxies do not
justify one. What it does assert is a technical bar: every sample must be
hardware-compliant, must loop cleanly, and must not be flagged as memorized.
Failing that bar is an objective failure; the rest is for human ears.

## Commands

```bash
# Synthetic corpus — a pipeline test, and labelled as one
rse train --source procedural --corpus-size 300 --steps 600 --out checkpoints/m.pt

# Real corpus, once NES-MDB is ingested
rse train --source dataset --dataset nesmdb --release midi --steps 20000

rse sample   --checkpoint checkpoints/m.pt --mood dark --bars 16 --seed 7 --stems
rse evaluate --checkpoint checkpoints/m.pt --n 16
```

`rse sample` runs model output through the constraint engine before rendering,
because unlike the rule-based generator the model has no compliance guarantee.
Anything it had to repair is reported.

## Known limitations

- **No model trained on real music.** The headline limitation, and it is
  blocked on the dataset rather than on code.
- **Mood and energy conditioning are untrained.** The tokens exist and the
  plumbing works, but a dataset file has no mood label, so it encodes as
  `Mood_unknown`. Proposal 6.4 is right that this needs labels first;
  conditioning on them now would be theatre.
- **Controllers, effects and instruments are not modelled.** Reserved in the
  vocabulary design, absent from the implementation.
- **No hierarchical generation.** Proposal 6.5 describes a phrase plan driving
  generation; Phase 3 does that with rules, but the model samples flat from a
  conditioning header. Combining them — the model composing *inside* a plan —
  is the obvious next step and is not implemented.
- **Evaluation has no human-listening harness.** Proposal 11.2 asks for
  listening evaluations; nothing here collects or stores them.
