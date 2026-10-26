# Source versus arrangement

Six compositions, each in three forms. Same music throughout -- the
scores are rebuilt from the stored token streams, with no cleanup,
transposition or requantisation.

- `*.source.mid` -- Fermata's own score, before NES conversion, at
  its own tempo with its declared GM instruments.
- `*.lead.mid` -- the track Fermata DECLARED as the lead (program 80).
  In most of these pieces it carries under 17% of the notes, so it is a
  thin line over the top rather than the tune.
- `*.busiest.mid` -- the track the arranger picked as the melody instead,
  where that differs. Often the voice actually carrying the material.
- `*.nes.wav` -- the NES rendition you already rated.

**Play the `.mid` files through Windows' built-in GM synth** (just
open them) and the `.wav` with the player. The timbres are not
comparable and are not meant to be.

For each piece, answer the three questions **separately for the MIDI
and the NES version** -- that contrast is the whole point:

1. **Melody** -- can you follow a tune? (yes / barely / no)
2. **Harmony** -- does it hold together? (yes / barely / no)
3. **Direction** -- does it go somewhere? (yes / barely / no)

| piece | MIDI: mel / harm / dir | NES: mel / harm / dir | notes |
|---|---|---|---|
| piece_E |  |  |  |
| piece_B |  |  |  |
| piece_D |  |  |  |
| piece_C |  |  |  |
| piece_A |  |  |  |
| piece_F |  |  |  |

Labels are neutral and shuffled. They carry no information about
which piece this is, what it scored before, or how much the arranger
changed it -- the mapping is held outside this directory until your
answers are recorded.

## Which voice is the tune?

The declared lead holds only a small share of each piece:

| piece | source notes | declared lead | arranger's pick |
|---|---|---|---|
| piece_A | 148 | trk 0 | (same) |
| piece_B | 326 | trk 0 | trk 1 |
| piece_C | 339 | trk 0 | trk 1 |
| piece_D | 196 | trk 0 | trk 1 |
| piece_E | 111 | trk 0 | (same) |
| piece_F | 199 | trk None | trk 1 |

If `busiest.mid` sounds more like the melody than `lead.mid` does,
that is worth noting -- it means Fermata is not putting the tune where
its own track plan says it will be.
