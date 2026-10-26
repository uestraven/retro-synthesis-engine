# Source versus arrangement: the NES pipeline is not the problem

Six pieces from the corrected baseline, two per rating band, each heard as
Fermata's own score in General MIDI and as the NES render. Labels were
neutral and shuffled; the mapping was held back until the answers were in.

| piece | more musical | either good? | prior NES rating | arp% | prior comment |
|---|---|---|---|---|---|
| piece_B | **NES** | yes | BEST | 40.5 | "cool" |
| piece_E | **NES** | yes | GOOD | 0.0 | — |
| piece_C | **NES** | not bad not good | OK | 39.2 | "has potential" |
| piece_D | **NES** | no | OK | 5.1 | "too repetetive" |
| piece_A | SAME | not bad not good | BAD | 0.0 | "harmonies make no sense" |
| piece_F | **NES** | no | BAD | 45.2 | — |

## The NES version won 5 of 6; it never lost

This is the opposite of the arrangement-damage hypothesis. The NES rendition
sounded *more* musical than Fermata's own score in five pieces and equal in
the sixth. Not once did the General MIDI source sound better.

So the branch is settled: **do not work on the arranger.** The NES conversion
is not degrading good music — on this evidence it is improving it. Likely
mechanisms, none tested:

- the four-channel monophonic reduction thins a cluttered texture, forcing one
  note per voice where Fermata had piled several up
- arpeggiating a chord across sub-row slots turns a vertical smear into a
  figure with rhythm
- NES timbres are bright and separated, so voices stay distinguishable where
  GM patches blur together

The three most heavily arpeggiated pieces (45.2%, 40.5%, 39.2%) all went to
NES, and the two with no arpeggiation at all went NES and SAME. Arpeggiation
is not harming these pieces and may be helping.

## But only 2 of 6 are any good

| answer | n |
|---|---|
| yes | 2 |
| not bad, not good | 2 |
| no | 2 |

That is the fifth outcome from the pre-registered table: coherence is not the
binding constraint, and neither is arrangement. The pieces that are not good
are not good **as compositions**, in both renderings.

## The prior ratings held up

Five of six pieces agree with the blind baseline judgements made in a separate
session with different labels and a different scale: BEST and GOOD both now
"yes", one OK now "not bad not good", BAD now "no".

The one movement is `piece_A`, previously BAD with "harmonies make no sense"
and now "not bad not good" — and it is also the only SAME. Its arranger
intervention was zero on every axis: 0% arpeggiated, 0% truncated, nothing
lost, nothing merged. So the piece whose harmony complaint was sharpest is the
piece the arranger never touched, and hearing it on GM instruments did not
reveal a hidden coherent composition.

## Decision

Per the pre-registered rule: the source MIDI is not better than the NES
version, so **no arrangement intervention**. The remaining failure is
composition quality, and the next experiment is theme-guided continuation —
supplying Fermata a short coherent opening and testing whether it can develop
given material, rather than asking it to invent eight bars unprompted.

One correction to the record: `docs/lead_continuity_diagnosis.md` reported
`assign_roles` overriding Fermata's declared lead as an arranger defect
"demoting the real tune". The declared lead carries under 17% of the notes in
four of these six pieces — 21 of 326 in `piece_B` — so the override was
selecting the voice that actually carries the material. That finding should be
read as a property of Fermata's track usage, not as an arranger bug.
