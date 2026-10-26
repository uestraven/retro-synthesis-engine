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

A third: tempo on this hardware is a **frame count**, not a BPM. The console
advances a tracker row every whole video frame, so only a ladder of tempos
exists (180.30, 150.25, 128.78, 112.69, 100.16, 90.15 …). Four of the six
presets had asked for tempos no NES could play.

Two more results worth not re-deriving: the quality measure is **"would this pass as
NES game music"** (60%), not "is this good" (25%) — the wrong question
understated the system for weeks. And harmony expansion is **closed**: 48 pairs
across two experiments returned zero preference.
