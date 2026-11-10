"""A language-model experiment over published snesmdb releases.

Pre-registered in the snesmdb repository at
`docs/EXPERIMENT_PREREGISTRATION.md`, before any model was trained.

This lives in the engine rather than in the dataset repository because the two
have no code dependency on each other, by design: snesmdb publishes artifacts,
this reads them through the documented file format. The direction is one-way
and stays that way.

Nothing here touches `rse.tokenizers.frozen`, which holds the frozen
NES-specific `rse-repr/1.0.0-d4k4-tempo1.01`.
"""
