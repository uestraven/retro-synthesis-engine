# Prompt-to-Song, procedural engine -- one complete track

**Listen to `boss_procedural.wav`.** One song, not a batch. 56.5 s, loops.

This is the first track produced by the public entry point reaching the
procedural composer. Nothing was hand-picked: one prompt, one recorded seed,
one run.

```
python scripts/prompt_to_song.py \
  "Compose a dark, energetic, 60-second NES boss battle theme with a memorable \
   main melody, driving percussion, a contrasting middle section, and a dramatic \
   return to the opening theme. Make it loop seamlessly." \
  --seed 20261026 --name boss_procedural
```

| | |
|---|---|
| engine | procedural (no model, no weights, no torch) |
| style | `boss`, from scenario `boss_battle` |
| music | 170 BPM, harmonic_minor, tonic 57 |
| form | A-A'1-A'2-B-A'' (40 bars, 56.471 s) |
| hardware violations | 0 |
| loop | timing continuous, nothing ringing past the boundary |

## The one question

Would this pass as NES game music? yes / no -- and if the loop or the middle
section is where it falls down, say which.

## What the prompt could not reach

"memorable" is reported unsupported, not satisfied: nothing in the composer
takes memorability as input, so the song either has a hook or it does not and
the report does not pretend otherwise. That is the thing to judge.
