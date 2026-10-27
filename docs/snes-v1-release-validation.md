# SNES V1 release validation

Run 2026-10-27 against the integrated SNES generation path, under a feature
freeze: no new composition, synthesis or orchestration work. Everything below
is either a measurement, a documentation correction, or a defect the
validation itself found.

Three defects were found, and two of them were audible. That is the argument
for running a validation milestone rather than shipping on a green suite:
every one of these was sitting behind passing tests.

## 1. Full regression suite

Run from a clean `git clone` of the committed tree, so the result describes
what is committed rather than what happens to be in the working directory.

| | |
|---|---|
| Baseline commit | `995f577` "Website 16-bit support: SNES through the API and the page" |
| Clean-checkout result | 1577 passed, 5 skipped, exit 0 |
| After this milestone's fixes | see the commit message for the final count |
| Nondeterministic tests | none observed |
| Lint | 2 findings, both pre-existing `E501` in the untouched `scripts/timing_fix_report.py` |

The clean-checkout run initially printed no summary line at all: `addopts`
in `pyproject.toml` already carries `-q`, and passing `-q` again makes `-qq`,
which suppresses it. Worth knowing before reading a "silent" green run as a
green run.

**NES baselines.** Unchanged, and verified the strong way rather than by
assertion: the same seed through the website produced a NES WAV with digest
`5549b6b6d690d9c6` both before and after every change in this milestone. The
sample-bank, BRR and ARAM work does not reach the NES path.

## 2. BRR loop verification

### The documentation was wrong, and the correction matters

This repository said the block at a loop point must use filter 0 because the
predictor has no history across a wrap. That is not what the hardware does.
The S-DSP keeps its two decoded-sample registers running continuously, and
history carries across the loop exactly as it does between any two blocks.

What filter 0 at the loop point actually buys is **repeatability**. The
history arriving at the loop block differs between passes — on the first pass
it is the attack, on every later pass it is the end of the loop region — so a
filter-1/2/3 block there decodes to slightly different samples each time
round, and filters 2 and 3 have poles close enough to the unit circle that
the difference need not decay. Forcing filter 0 makes the block depend on
nothing but its own nibbles.

So it is a conservative encoding policy, not a hardware requirement. Real
drivers rely on the history and sound fine. RSE keeps the policy, and now
documents it as a policy. Corrected in `rse/snes/brr.py`, both the module
docstring and the comment at the decision.

### Defect found: the loop was periodic but not block-aligned

The sustained instruments were generated 2,202 samples long with the loop
starting at 245. BRR works in 16-sample blocks, so:

- the length was padded to 2,208 with **six zero samples**, which landed
  **inside the loop region**;
- the loop point rounded down to 240, restarting **five samples early**.

Every repetition therefore played a six-sample dropout. The loop period was
61.5 ms, so that is a dropout at **16.3 Hz** under every sustained note.

This is the third loop defect in this bank, and the reason the first two
tests did not catch it is worth recording: `test_loops_still_wrap_seamlessly_after_encoding`
measured the discontinuity in the *asset's* declared loop region, not in what
the encoded sample actually plays. The data was seamless. The thing the
hardware would play was not.

Fixed in `rse/snes/samples.py` by aligning the lead-in and the loop length to
whole blocks before generating the waveform. The encoder cannot fix this —
by the time it sees the data, padding or truncating is all it can do.

| instrument | wrap step before | wrap step after | largest in-loop step |
|---|---|---|---|
| brass | 0.12543 | 0.05872 | 0.14344 |
| strings | 0.12500 | 0.00079 | 0.25001 |
| choir | 0.06259 | 0.01718 | 0.19807 |
| bass | 0.06275 | 0.05893 | 0.06250 |

### Defect found: the alignment fix detuned the bank

Caught by this milestone's own tuning check, which is the only reason it is
not shipping. `loop_partials` derives the fundamental *from* the loop length,
so moving the length moves the pitch: rounding brass from 1,957 samples to
the nearest block, 1,952, left room for exactly 16 cycles, and 16 cycles in
1,952 samples is 262.295 Hz against a wanted 261.626 — **+4.4 cents**, on
every sustained instrument.

`align_loop` now searches block counts within ±25% of the requested length
and keeps the one whose quantised fundamental is closest, instead of rounding
to the nearest.

| instrument | wanted | before alignment | after `_align` | after `align_loop` |
|---|---|---|---|---|
| brass / strings / choir | 261.626 Hz | +0.02 ct | +4.42 ct | **+0.38 ct** |
| bass | 65.406 Hz | +0.88 ct | +0.88 ct | **−0.05 ct** |

### Defect found: the float predictor was not the hardware's arithmetic

Cross-checking the decoder against an independently written one — same
algebra, integer shifts instead of float fractions — showed them diverging by
up to **0.46% of full scale** on brass, which is 150 times the per-sample
rounding difference that produced it. The prediction feeds back, so a
half-unit disagreement does not stay a half-unit disagreement.

The hardware truncates each shift toward negative infinity; float
multiplication rounds to nearest. Since the hardware is the thing being
modelled, `rse.snes.brr.predict` now implements the integer forms and the
float `FILTERS` table is kept only as documentation. Both the encoder and the
decoder call it, which also closes a subtler gap: the encoder scores
candidate filters by the error the decoder would produce, and that scoring is
only meaningful if it models the decoder exactly.

### Verified

| check | result |
|---|---|
| Block alignment | every asset's length is a multiple of 16; the encoder pads nothing |
| Loop-point adjustment | every loop start is a multiple of 16, so the round-down is a no-op |
| Header END / LOOP flags | END on the last block; LOOP set iff the sample loops |
| Repeated loop decoding | five passes decode **bit-identically** (drift exactly 0.0) |
| Discontinuity at the wrap | below the largest in-loop step for all four looping instruments |
| Decoded sample stability | the loop block's peak error is no worse than the predictive blocks' |
| Non-looping percussion | END without LOOP ends the voice; the tail after it is exactly silent |
| Cross-check | agrees with an independent integer decoder |

**On the cross-check.** It is an independent implementation, not a
third-party one — written from the published integer forms, in this
repository, by the same author as the thing it checks. It catches a wrong
coefficient or wrong truncation. It does not substitute for an emulator, and
no emulator has run any of this.

## 3. ARAM validation

### The echo buffer was already in the total

The review worked out that adding a 240 ms echo to the reported 30,985 bytes
would reach 61,705 and leave 3,831 free. The echo was already inside that
figure — the boss style defaults to `room`, 64 ms, 8,192 bytes — so cavern
replaces it rather than adding to it. Measured across the presets on one
track:

| preset | delay | echo bytes | total | of 64 KiB |
|---|---|---|---|---|
| dry | 0 ms | 0 | 22,861 | 34.9% |
| room | 64 ms | 8,192 | 31,053 | 47.4% |
| hall | 128 ms | 16,384 | 39,245 | 59.9% |
| cavern | 240 ms | 30,720 | 53,581 | 81.8% |

The report gave no way to tell, which is the real defect. The caveat now says
"do not add it again" and a test asserts it.

### Itemized report, with each line's evidence class

The deliverable SNES track (`boss`, seed 20261027, 75 s requested, `hall`):

```
ARAM 40056 / 65536 bytes (61.1%), 25480 free
  samples    17532  (measured)
  directory     28  (measured)
  sequence    3552  (modelled)
  echo       16384  (measured)
  driver      2048  (assumed)
  runtime      512  (assumed)
  -- of which measured 33944 bytes
```

Per instrument, encoded: bass 5,778 · kick 3,960 · snare 3,240 · hat 1,260 ·
brass 1,098 · strings 1,098 · choir 1,098.

**Four evidence classes, and every line now carries one.**

- **measured** — the size of something that exists. Sample bytes are `len()`
  of real encoded data. The echo buffer follows exactly from the delay
  register at 2 KiB per 16 ms step. Directory entries are 4 bytes by the
  format's definition.
- **modelled** — a defensible number for something RSE does not produce.
  Sequence data at 4 bytes per note event is a plausible driver encoding;
  RSE emits no sequence bytes.
- **assumed** — a stated allowance. Driver code and runtime state. RSE has no
  SPC700 driver.
- **unmodelled** — named, counted as zero: instrument table, pattern and
  order tables, page-alignment padding (the DIR and ESA registers address
  pages, so both regions must start on a 256-byte boundary), the SPC700's
  stack and direct page, and the DSP register shadow most drivers keep.

So **`hardware_budgeted` is False**, and it is deliberately not a function of
whether the total fits. Fitting means the represented allocations fit; the
flag means the representation is complete. Only the second licenses the
phrase "hardware-compatible memory budgeting", and it is not true yet.

The API response carries `hardware_budgeted` alongside `hardware_exact`, and
the page now reads "audio RAM accounted" with a separate "of which measured"
figure, rather than presenting one authoritative number.

## 4. Stereo and rendering

| check | result |
|---|---|
| Left/right separation | correlation 0.9180, channels not identical, side energy 43% of mid |
| Unintended clipping | 0 clipped samples in the output; peak L 0.8900, R 0.8744 |
| Sample tuning | within 0.38 cents of equal temperament at every pitch tested |
| ADSR / key-off | release decays monotonically to exactly 0.0 — no residual at key-off |
| Attack rate | fixed rate, not scaled to note length: brass 35 ms, strings 160 ms, bass 6 ms, truncated for notes shorter than the attack |
| Release-tail occupancy | 8 of 8 voices peak, 0 dropped, 0 stolen, 0 displacements |
| Eight-voice maximum | never exceeded; `reserved_voices=3` correctly drops to 5 and sheds 15 notes |
| Deterministic rendering | same settings and seed → byte-identical WAV, verified through two separate HTTP requests |
| WAV metadata | 2 ch, 44,100 Hz, 16-bit for SNES; 1 ch for NES |
| Interpolation overshoot | measured; see below |

**Catmull-Rom overshoot.** Real, bounded, and labelled. Its kernel has
negative lobes, so it exceeds the range of the samples it interpolates
between — the price of reproducing them exactly at integer positions, which
the earlier Gaussian-shaped low-pass avoided by not being an interpolator at
all. Worst case over seven pitches per instrument:

| instrument | overshoot |
|---|---|
| snare | 6.51% |
| brass | 0.31% |
| strings | 0.25% |
| kick | 0.14% |
| bass | 0.03% |
| choir | 0.02% |
| hat | none (−0.28%) |

Largest on the snare, whose spectrum is mostly noise, which is the expected
shape. It cannot clip the output: the mix is scaled by one scalar after
summing, so the output peak is set by the normaliser. (The previous clipping
report counted pre-normalisation overshoot and was therefore measuring
something the listener never hears; `audio.warnings` now says so explicitly.)

The S-DSP's four-point Gaussian deliberately attenuates high frequencies
instead. That is a difference in **character** as well as accuracy, so
Catmull-Rom stays labelled as preview interpolation and exact Gaussian
interpolation is deferred to the hardware-accuracy milestone.

## 5. Website generation

Generated through the HTTP API on `127.0.0.1:8077`, the same path the browser
uses. Listener confirmed loopback-only (`ss -ltn`: `127.0.0.1:8077`).

| check | result |
|---|---|
| Console selection honoured | `console: "snes"` → `renderer: snes_sampler`; NES → mono with `snes: null` |
| Uses the SNES sampler | `notes_rendered: 1282` across 8 voices, stereo |
| Stereo in the browser | 2-channel 16-bit WAV with non-identical channels; plays in the page's `<audio>` element |
| Downloaded WAV valid | 13,639,924 bytes, parses as 2 ch / 44.1 kHz / 16-bit, 77.32 s |
| Reproducible | two requests with the same settings and seed → identical WAV digest `2fa52bd85e949d5f` |
| `hardware_exact=false` | present, with `hardware_budgeted=false` beside it |
| Unsupported features not advertised | no SPC export, no emulator claim, no Gaussian-table claim |

### Blocker found: the echo control is inaudible

Generating one song at `dry`, `room`, `hall` and `cavern` produced **four
byte-identical WAVs**. The setting reserves the echo buffer in the memory
account, and `VoiceSettings.echo` records a per-voice send, but no code reads
either when rendering — there is no echo processor. This is precisely the
"the metadata exists, therefore the capability exists" failure.

Under the feature freeze the fix is disclosure, not implementation. The
control is now labelled budget-only in the request model, in
`capabilities.echo_presets[].affects_audio`, in the report's `echo` block,
and on the page itself ("**This will not change how the song sounds.**"). A
test asserts the audio does *not* change, which looks backwards and is the
point: when an echo processor is written, that test fails and whoever writes
it has to come back and update every label.

**Your call** whether that is acceptable for V1, or whether the control
should be removed, or whether echo is worth implementing. The three options
cost very differently and only you can weigh them against the roadmap.

Everything else in the panel does affect the output: `reserved_voices=3`
changes the allocation (8 voices → 5, 15 notes shed) and the audio.

## 6. Musical acceptance

Not done by me, and not substitutable. One complete SNES boss theme is in
`listen_snes/web_snes_boss.wav` with a NES rendering of the **same
composition** beside it for comparison — same seed, same 48 bars, same form,
identical MIDI digest, so the only difference is the renderer.

Reproduce with:

```
POST /api/generate
{"console": "snes", "style": "boss", "seed": 20261027, "seconds": 75, "echo": "hall"}
```

What the measurements can and cannot tell you: they say the melody is
present, the voices are allocated, nothing clips and the tuning is right.
They cannot say whether it sounds like SNES-era game music. That is the
question this milestone exists to ask you.

## Remaining blockers

1. **The echo control is inaudible** (above). Disclosed, not fixed. Needs a
   product decision.
2. **No emulator verification.** The BRR cross-check is an independent
   implementation, not a third-party one, and nothing here has been played by
   an SPC700 emulator.
3. **`hardware_budgeted` is False.** Five allocation classes are unmodelled.
   No hardware-compatible memory claim until that list is empty.
4. **No SPC export.** Out of scope for this milestone and still out of scope.
