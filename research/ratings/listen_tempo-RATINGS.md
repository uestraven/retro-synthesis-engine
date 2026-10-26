# Tempo snapped to the NES frame grid — 6 before/after pairs

Not blind, and not a preference test. **A is the tempo RSE used to produce;
B is the nearest tempo an NES can actually play.** Same style, same seed, same
notes — except `action` and `victory`, which also gain one section because a
different number of bars now fits 60 seconds.

Why B exists: the console has no tempo register. A music driver runs once per
video frame (60.0988 Hz) and advances the tracker one row every *whole* number
of frames, so only a ladder of tempos is reachable. RSE composed in float
seconds and produced tempos off that ladder — `action` asked for 168 BPM,
which is 5.37 frames per row. No NES ever played it.

| style | A (was) | B (now) | frames/row | change |
|---|---|---|---|---|
| boss | 150 | 150.25 | 6 | +0.17%, notes identical |
| town | 128 | 128.78 | 7 | +0.6%, notes identical |
| tension | 104 | 100.16 | 9 | −3.7%, notes identical |
| adventure | 137 | 128.78 | 7 | −6.0%, notes identical |
| victory | 144 | 150.25 | 6 | +4.3%, **4 → 5 sections** |
| action | 168 | 180.30 | 5 | +7.3%, **5 → 6 sections** |

`boss` and `town` barely move and are a sanity check — if those sound
different, something other than tempo changed and I need to know.

`adventure` and `action` are the real question: 6% slower and 7% faster are
both plausibly audible.

## The question

For each pair: **does B still pass as NES game music?** And if A and B differ
enough that you prefer one, say which.

| pair | B passes? | prefer | comment |
|---|---|---|---|
| boss | | | |
| town | | | |
| tension | | | |
| adventure | | | |
| victory | | | |
| action | | | |

## Outcome

Confirmed 2026-10-26: all tempos corrected, B accepted everywhere. The 8
changed baseline hashes (`action` and `victory`, four seeds each) were
re-pinned; the other 16 stayed byte-identical.
