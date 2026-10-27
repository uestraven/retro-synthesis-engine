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
