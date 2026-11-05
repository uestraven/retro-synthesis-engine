# Listening records

The listener's own judgements, which are the one thing in this project that
cannot be regenerated. Everything else a listening batch produced — the audio,
the token dumps, the per-piece JSON — is reproducible from a seed and is not
kept.

`ratings/` holds one file per batch, named for the directory it came from.
`keys/` holds the blinding keys, so a rating stays interpretable: an A/B
verdict means nothing without knowing which arm was which.

Batches, roughly in order:

| file | what was being decided |
|---|---|
| `listen_batch2-RATINGS.md` | the 30-piece blind batch that came back "99% not good" |
| `listen_timing-RATINGS.md` | the arranger time-conversion fix, blind A/B |
| `listen_arch-RATINGS.md` | Fermata vs native RSE-100k |
| `listen_corrected-RATINGS.md` | baseline after the timing correction |
| `audit_source_vs_nes-AUDIT.md` | GM source vs NES render, where the melody was lost |
| `listen_themes-THEMES.md`, `listen_themes_2bar-THEMES.md` | theme bank, 4-bar and 2-bar openings |
| `listen_guided-*-RATINGS.md` | theme-guided continuation |
| `listen_profile_ab-RATINGS.md` | the NES composition-profile planning layer |
| `listen_procedural-RATINGS.md` | the procedural composer — "fantastic, no complaints" |
| `listen_rhythm-RATINGS.md` | harmonic rhythm, 24 pairs, no preference |
| `listen_vocab-RATINGS.md` | harmonic vocabulary, 24 pairs, no preference |
| `listen_paired-RATINGS.md` | paired 8-vs-16-bar material |
| `listen_prompt_to_song-README.md` | the first complete Prompt-to-Song track |
| `listen_tempo-RATINGS.md` | tempo snapped to the NES frame grid — accepted |
| `listen_phrase-RATINGS.md` | phrase-level melody — dropped, undecided at n=2 |
| `listen_variety-RATINGS.md` | key/tempo/form/arrangement variety — **null** |
| `listen_sound-RATINGS.md` | duty + velocity sound design — **accepted** |
| `listen_snes-RATINGS.md` | first SNES sampled stereo boss theme — **accepted** |

A third: tempo on this hardware is a **frame count**, not a BPM. The console
advances a tracker row every whole video frame, so only a ladder of tempos
exists (180.30, 150.25, 128.78, 112.69, 100.16, 90.15 …). Four of the six
presets had asked for tempos no NES could play.

**Byte-uniqueness is not perceptual distinctness.** The variety experiment
moved every number it set out to move — harmony and bass from 5-6 unique
across 20 seeds to 17-20, forms from 1 to 2-5, keys from 1 to 5 — and the
listener heard no difference. The melody had always measured 20 unique out of
20 while sounding the same, because all 20 are the same five motif shapes
re-anchored. Counting distinct note sequences counts the anchoring, not the
idea. Do not use it as a diversity metric again.

**Trace a rendered artifact to the SCORE before the synthesiser.** A buzz in
the first SNES track took three diagnoses. The first two -- a loop
discontinuity at 32.7 Hz and a bass approach note with no harmonic target --
were real defects found by examining the renderer, and fixing them did not
remove what the listener was hearing. The cause was a one-row grace note on an
odd row in the MELODY, from `vary_motif` ornamenting a note too short to
subdivide: an NES defect in 54 of 240 songs. The MIDI is cheaper to read than
the audio and it is unambiguous.

**A test can measure the data and miss what the hardware would play.** SNES V1
release validation found a six-sample dropout under every sustained note, at
16.3 Hz, in a bank whose loop tests were green. The tests measured the
discontinuity across the *asset's declared* loop region, where the waveform
really was seamless; the encoded sample that the player actually loops was
zero-padded to a block boundary and its loop point rounded down. The question
a loop test has to ask is not "is this data periodic" but "what comes out of
the decoder on pass two".

**A fix is a change, so measure what else it moved.** Block-aligning those
loops removed the dropout and detuned the whole bank by 4.4 cents, because
`loop_partials` derives the fundamental from the loop length. It was caught
only because the same milestone independently checked tuning. Two defects in
this project have now been introduced by the fix for a previous one.

**Repeatability is not absence of a click.** The same milestone proved the
loop decoded bit-identically across five passes and called looping verified.
It was not: `_interpolate` clamped its four taps to the buffer, so the last
two reads before every wrap repeated the final sample instead of reading the
loop start -- a step on every held note, in the rendered output, downstream of
three rounds of green loop tests. Nothing was wrong with the loop. The player
did not know there was one. Periodic data says nothing about a player that
reads it wrong, and the only test that catches this renders a note long enough
to wrap and measures the step at the boundary.

**Count the skips.** `test_the_mix_has_little_sub_bass` read a gitignored path
and skipped when it was absent, which in a clean checkout is always -- a test
sitting in the suite looking like coverage of the drone regression without
once executing. Found by reading skip reasons rather than the pass count.

**When a threshold fails, suspect the threshold.** That test then failed twice
for reasons that were not the code: its band was hard-coded 20-60 Hz and
described as below the bass's range, when the bass plays 55 Hz; and the fix
measured the whole mix including a kick, which is legitimately sub-bass. The
right discriminator separates artifact from content by an order of magnitude
-- sub-bass DURATION, 0.15 s for a kick transient against 6.17 s for the drone
-- not by a limit tuned to sit just past the current measurement.

**The thin NES pad is not a defect. Two fixes for it were rejected by ear.**
`town` and `tension` use the `pad` figure: one note per bar, the root,
sustained. The third never sounds, so the accompaniment states no chord
quality at all -- measured, a full triad in 0% of bars. A musician who
listened said the engine "doesn't really have harmony", which seemed to
confirm it.

It did not. Four NES renders at ascending accompaniment density were compared
blind-ish by the listener: `pad` (1 note/bar) "sounds fine"; `pad_triad`
(3 sustained notes stating the triad) "jumbled, not good"; `arpeggio`
(8/bar, the figure that already existed) "sounds good"; a frame-rate arpeggio
at 60 note changes per second "weird, like a buzzing in the background". Both
new figures were removed.

`town` and `tension` both moved to `arpeggio` on that verdict -- "better and
more authentic NES" for town, and preferred over the pad for tension on a
second listen -- and their eight baseline hashes were re-pinned deliberately,
which is the only way an accepted piece should move.

**Then `alternating` won, and it never states the third at all.** A figure
audit flagged it as the one measurable red flag left: eight notes a bar, as
busy as `arpeggio`, but root and fifth only, so a full triad in 0% of bars.
Same no-chord-quality shape as the SNES rootless dyad with a busy rhythm over
it. Rendered against `arpeggio` and `stab` on boss with rhythm held constant,
it was judged the BEST of the three -- and then preferred over `offbeat`,
boss's incumbent, which spells the chord completely. `boss` moved to
`alternating`; `offbeat` stays in the pool because both were judged to have
merit for different scenarios.

So three separate times now, "the accompaniment does not state the third" has
predicted a problem that was not there, and twice a fix for it made things
worse. The measurement is real and it does not track what a listener hears --
the chord's quality is evidently carried well enough by melody and bass, and
a comp that competes with them is worse than one that leaves them room. Stop
using triad completeness as a quality signal for NES accompaniment.

Two lessons under that. A harmonic deficit visible in the score is not
necessarily audible as one -- the quality was being carried by melody and
bass, and adding it to the comp made the texture worse, not clearer. And the
answer was already in the codebase: `arpeggio` was the good option all along,
so the whole exercise produced a one-word default change and two deletions.

**A chiptune arpeggio is a pitch change, not a sequence of notes.** The
frame-rate arpeggio buzzed, and the mechanism is in the renderer rather than
the rate. `_render_note` starts each note at `index = 0` and derives the duty
sequencer position from it, so every note-on resets the waveform phase. The
implementation emitted 112 note-ons per bar; hardware writes a new period to
the pulse timer while the channel keeps running. `rse/events/classify.py`
already says this -- it classifies arpeggios as within-row EFFECTS, not
attacks -- and the renderer has no way to express one. The row grid was never
the limit (a frame is representable in ticks); the per-note phase reset is.

**An 11.4% validation-loss improvement produced unusable music.** The
foundation checkpoint was retrained from scratch on the byte-identical corpus
(same corpus_hash, same 11,151 windows, same 5,667,840 parameters) and reached
0.5030 against the original's 0.56789 -- a real gain, traced to the original
having been trained from warm weights with a cold optimizer. The listener's
verdict on four samples: "horrible and can barely be considered music."

So validation loss on next-token prediction does not measure music, and the
gap is not small -- it is the difference between a clear numerical win and
output that does not qualify as music. Two independently trained models now
lose decisively to the rules-based composer.

**Training on one genre did not help either.** The seven NES JRPGs give 233
compositions and 307k tokens. Fine-tuning changed nothing at lr 3e-5 and made
validation WORSE at 2e-4, where train loss rose too -- disruption rather than
memorization. The reason was predicted from the data before the run: 10% of
tracks supply 40% of training windows because they are long, and RPG tracks
are long, so the foundation had already over-learned the idiom.

**The corpus was not the problem, and the evidence that said it was came
from a broken string match.** Doubt about the model line turned to the data:
if NES-MDB lacks Zelda II, SMB3 and Castlevania III, the dataset is not
musically valid. An availability check reported 19 of 50 curated titles
absent. All three named games were in fact present, and so were 31 of 35
spot-checked classics; the release has 392 games, not the 374 first counted.

The cause was the filename grammar. NES-MDB writes
`NNN_Game[_Part]_TT_NNTrackName.mid`, and the game spans a VARIABLE number of
underscore fields: a sequel number (`SuperMarioBros_3`), a subtitle
(`ZeldaII_TheAdventureofLink`), or punctuation in the real title (`Dr_Mario`).
Reading field 1 -- which `_matches_title` did in shipped code -- collapsed
SMB1, SMB2 and SMB3 into one game and returned nothing at all for Dr. Mario,
whose field 1 is "Dr". The first hand-check compounded it by spelling sequels
with digits (`Castlevania3`) where the release uses Roman numerals. Two
independent naming assumptions, two wrong answers, pointing the same way.

So data purity never explained the output: the 9,702 training windows that
produced "barely music" already contained SMB3, Castlevania III and Zelda II.
Before a dataset-availability claim is allowed to justify a decision, print
the actual keys.

**A codec validated against its own decoder is not validated.** The BRR
encoder had only ever been checked against `rse.snes.brr.decode`, and three
rounds of green loop tests plus a full V1 release validation all passed. Built
as an external oracle, blargg's `snes_spc` found that 67-72% of the sustained
instruments' samples would WRAP on a real S-DSP: the hardware ends each
decoded sample with `CLAMP16(s); s = (int16_t)(s * 2)` -- saturate at half
scale, then double with 16-bit wraparound -- where `_clamp` saturates at full
scale and never wraps. On a full-scale sine the two decoders disagree by 96%.

The cause was the encoder scoring filter choices with the saturating decoder,
so the overflow-prone predictors 2 and 3 looked free at full amplitude: bass
is filter 3 in 602 of 642 blocks. Scoring against the hardware's arithmetic
instead takes every wrap to zero for 2.2 dB of round-trip accuracy on bass and
under 0.6 dB elsewhere. The first fix tried -- banning filters 2 and 3 --
cost three times as much, because the ban applies to the whole bank where
correct scoring picks a safe filter per block. Fix the measurement, not the
symptom.

Nothing here was audible: the preview renderer decodes with the same
saturating decoder it encodes for, so the output was self-consistent, and
`aram.py` already declined to claim hardware compatibility. The defect is in a
capability that was never claimed -- which is exactly why only an independent
implementation could find it, and why the fix is opt-in with the default left
byte-identical.

Two process notes. The harness earned trust only after feeding the hardware
decoder's own samples back through it reproduced the oracle to 0.02%; before
that check it was just a second opinion. And two diagnoses were wrong first --
overflow saturation (a counter that measured RSE's predictor chain rather than
the hardware's, and so reported zero) and then LSB parity. Validate the
instrument before believing the reading.

**The model has no sense of where a phrase ends.** Every sample set both
`underran_length` and `overran_length`: it tried to stop before the eighth bar,
was held to length by the grammar, then tried to start a ninth. Its form is
imposed by the sampler, not chosen. Trained on 8-bar windows with a 3072-token
context, a 32-bar request silently exceeds both.

**Measure the resource that fails, not the one you were thinking about.** The
first deployed NES request returned 502. The duration cap had been sized by
careful CPU measurement -- NES renders at 0.022x realtime, so 180 s of audio
is ~40 s of work even on a tenth of a core -- and memory was estimated in my
head as "21 MB of samples, should be fine". Peak RSS for a 60-second render
was 521 MB against a 512 MB instance. A CPU budget degrades; a memory ceiling
kills the process, and the platform reports it as a bad gateway with nothing
in the application log.

The cause was ordinary numpy spelling, not an algorithm: `pulse_sum =
pulse_sum + buffer` allocates a new 45 MB array per channel, and
`np.where(cond, f(x), 0.0)` allocates one per arithmetic step. Rewriting the
mixer in place and letting the product path skip per-channel stems cut peak
memory 35% with bit-identical output. Peak memory is a product constraint
wherever the thing is deployed, so it belongs in the same measurements as
speed.

**Advertising a setting is a claim that it does something.** The SNES echo
control reserved the right number of ARAM bytes and produced byte-identical
audio at every preset, because nothing reads the per-voice echo sends. Metadata
existing is not the capability existing — which was already written down as a
rule here, and still shipped. For a user-facing control the check is cheap:
generate at two settings and diff the bytes.

**Composition, instrumentation and expression are separable, and only the
first had ever been worked on.** Four experiments that rearranged material
returned no preference; the first that changed how the material is *played* --
per-song pulse duty and position-shaped velocity, with every note identical --
was accepted immediately. Before reaching for another composition change, ask
whether the complaint is actually about performance.

**Four arrangement-level experiments have now returned null**: harmonic rhythm
(24 pairs), harmonic vocabulary (24 pairs), key/tempo/form/arrangement
(6 per arm), and phrase structure (undecided, dropped at 2 pairs). Changing how
the material is arranged has not once moved a listener. The untested axis is
the material itself.

Two more results worth not re-deriving: the quality measure is **"would this pass as
NES game music"** (60%), not "is this good" (25%) — the wrong question
understated the system for weeks. And harmony expansion is **closed**: 48 pairs
across two experiments returned zero preference.
