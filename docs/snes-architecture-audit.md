# SNES foundation — architecture audit

What exists, what is reusable, what is NES-specific, and the shortest route to
a playable SNES-style track. Every claim below was checked against the code,
not inferred from the directory layout.

NES V1 is frozen: its composer, sound-design defaults and 24 regression
baselines do not move.

---

## 1. Reusable unchanged

| abstraction | where | why it transfers |
|---|---|---|
| `Composition` | `rse/events/schema.py` | `hardware_profile` is a free string, `ticks_per_beat` is generic, and tempo/meter/loop/section data carries no console assumption |
| `Note.instrument_id`, `Note.effects` | same | **already present and unused by NES.** The score model can already name an instrument and attach modulation; nothing had to be invented |
| `HardwareProfile`, `ChannelSpec` | `rse/hardware/profile.py` | JSON-driven, already carries `instrument_params`, `effects`, `volume_levels`, `max_simultaneous_notes`, `value_min/max`. A SNES profile is a new file, not a new loader |
| `Renderer` protocol, `RenderNote`, `RenderResult`, `EnvelopePoint` | `rse/renderers/base.py` | the protocol is `render(notes, settings) -> RenderResult`, with no channel-count assumption, and `RenderResult.stems` is already a per-channel dict. `EnvelopePoint(sec, gain)` and the `envelopes` argument already exist and are already honoured per channel |
| motif model | `rse/procedural/compose.py` `Motif` | scale-degree offsets, not pitches, so it re-harmonises against any chord vocabulary |
| form archetypes | `rse/procedural/variety.py` | ternary, rounded binary, pop32, linear, theme-and-variation are era-general; they came from the 8/16-bit repertoire in the first place |
| phrase planning | `rse/procedural/phrase.py` | antecedent/consequent, cadence placement, rests — no hardware in it |
| `rse/analysis/restatement.py` | — | generic bar/contour analysis |
| `rse/hardware/tempo.py` | — | already parameterised by `frame_rate` and `rows_per_beat`; the NTSC constants are defaults, not assumptions |

**Conclusion:** the score representation and the renderer interface do not need
replacing. This is the single most important audit result — it means the SNES
work is additive.

---

## 2. NES-specific — must not be forced onto SNES

**Hard-coded channel identifiers.** `"P1" | "P2" | "TR" | "NO"` appear in
**17 modules**, including the composer itself:

```
rse/procedural/compose.py      rse/procedural/sound.py
rse/transfer/arranger.py       rse/transfer/converters.py
rse/transfer/score.py          rse/transfer/ensemble.py
rse/renderers/nes_apu.py       rse/analysis/restatement.py
rse/compose.py                 rse/generation/procedural.py
rse/cli.py                     rse/conditioning.py
rse/data/exprsco.py            rse/data/synthetic.py
rse/tokenizers/vocab.py        rse/training/evaluate.py
rse/ui/pianoroll.py
```

The composer writes `channel_id="P1"` directly in `render_melody`, `"P2"` in
`render_harmony`/`render_counter`, `"TR"` in `render_bass`, `"NO"` in
`render_drums`. **Musical role and physical channel are the same thing today.**
That is precisely what Phase 1 says must be separated, and it is the main
refactor this milestone needs.

**Other NES-bound pieces:**

| thing | problem for SNES |
|---|---|
| `Noise` event type | a dedicated noise channel. SNES percussion is sampled, so it is a `Note` with a drum instrument |
| triads only | `QUALITY_INTERVALS` has major/minor/diminished/augmented and `Chord` has `degree`/`quality`/`root_shift`. **No sevenths, ninths, elevenths, inversions, or omission priority** |
| `DUTY_RMS`, `TIMBRES`, duty presets | pulse-waveform concepts with no SNES meaning |
| 16 rows/bar on the NTSC frame grid | SNES timing is driver-defined, not video-frame-locked |
| `write_wav` | `setnchannels(1)` — **mono only.** Stereo is a real gap |
| `RenderSettings` | no stereo, pan or echo fields |

---

## 3. New model needed: intended harmony vs sounding notes

The architectural addition called essential. Today `Voicing` conflates two
things; SNES needs three distinct layers.

```
ChordIntent          what the composer means
  degree, quality, extensions=(7,9,11), inversion, borrowed_from
       |
       v  voicing planner: priority, omission, register, common tones
Voicing              what actually sounds
  pitches, omitted, rationale
       |
       v  voice allocator: 8 voices, roles, priority, stealing
VoiceAssignment      which physical voice, and what was dropped
  voice_index, role, stolen_from, dropped
```

Worked example, Cmaj9 — intent is five pitch classes, and the fifth is the one
to drop because the third and seventh carry the quality and the ninth carries
the colour:

| pitch | role in the chord | kept? |
|---|---|---|
| C | root | bass states it |
| E | **third** — defines major | keep |
| G | fifth | **omit** |
| B | **seventh** — defines maj7 | keep |
| D | ninth — the colour | keep |

Chord-tone priority therefore: root (via bass) > third > seventh > ninth >
eleventh > fifth. That ordering is the rule that lets dense jazz harmony fit
without exhausting eight voices.

---

## 4. Eight-voice allocation

Roles are logical and never pinned to a voice index:

```
lead · countermelody · chordal harmony · sustained pad · bass
arpeggio · percussion · accent/effect
```

Allocation per tick: sort active parts by configurable priority (lead, bass,
percussion, harmony, countermelody, decoration — overridable per style and
passage), assign in order, and when more than eight are live, steal from the
lowest priority. Every steal is **reported**, never silent: voice index, role,
tick, what displaced it, and whether a release tail was cut.

Two render modes: music-only (all 8) and game-ready (configurable reservation
for effects, or a deterministic stealing policy).

---

## 5. 64 KiB ARAM budget

Total must not exceed **65,536 bytes**. Accounted as:

```
driver code + sample directory (4 bytes/entry)
  + BRR sample data (9 bytes per 16 decoded samples)
  + sequence data
  + DSP echo buffer (delay-dependent, 2 KiB per 16 ms step)
  + runtime state
```

The echo buffer is charged to the budget, so a long reverb costs instrument
memory — which is the real tradeoff the hardware imposes. The report states
total, sample, driver/sequence, echo, and remaining.

**SPC700 is the audio CPU running the driver; the S-DSP does the eight-voice
sample playback, envelopes, mixing and echo.** Vibrato and tremolo are
driver-written pitch and volume changes, *not* an S-DSP LFO; the DSP's
pitch-modulation feature is a separate thing (voice N modulated by voice N−1)
and will be modelled separately if at all.

---

## 6. Two modes, never conflated

| mode | what it is |
|---|---|
| **SNES-inspired preview** | sample-based synthesis, 8-voice limit, hardware-informed envelopes/pan/echo, WAV out. **Not** hardware-exact and never labelled as such |
| **SNES hardware-compatible** | real BRR blocks, 9-byte/16-sample alignment, valid loop points, representable envelopes, a real ARAM allocation, and emulator verification *before* any compatibility claim |

---

## 7. Shortest route to the first playable track

The vertical slice, deliberately thin. No BRR, no SPC, no large bank.

1. **`configs/hardware/snes.json`** — 8 generic sampled voices, stereo, echo
   params. New file; the existing loader reads it unchanged.
2. **Role/voice split** — introduce a `part` field (role name) on the score and
   a `VoiceAllocator` that maps parts to voice indices. NES keeps its 1:1
   mapping so nothing changes there.
3. **Extended chords** — add `extensions` and `inversion` to `Chord`, plus a
   voicing planner with the priority order above. NES never requests them, so
   the frozen baselines cannot move.
4. **Instrument model + a handful of samples** — synthesised in-repo
   (sine/saw/noise envelopes shaped into brass, strings, bass, drums) so the
   slice needs no third-party assets and no licensing question. Real sample
   provenance comes later.
5. **`snes_sampler.py` renderer** — sample playback, ADSR, per-voice pan, 8-voice
   cap, stereo out. Needs `write_wav` extended to stereo.
6. **One boss theme**, 60 s, A–A′–B–A″, through
   shared composer → SNES arranger → sampler → WAV.

Items 1–3 are prerequisites; 4–6 are the slice. Nothing here requires touching
`render_melody`, `render_harmony`, `render_bass` or `render_drums` for NES —
they keep writing to their fixed channels.

---

## 8. What this audit does **not** claim

- That the shared composer is adequate for SNES composition. It plans one
  motif, one harmony voice, one bass and drums. Royal Road harmony, jazz
  extensions, modal interchange and voice leading are **new composition work**,
  not an arranger concern, and Phase 3 is where they land.
- That mono-to-stereo is trivial. `write_wav`, `RenderSettings`, `RenderResult`
  and the normalisation helper all assume one channel.
- That four composition nulls on NES mean the composer is finished. They mean
  those four dimensions were already adequate *for NES*; SNES asks different
  questions of it.
