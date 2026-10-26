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
