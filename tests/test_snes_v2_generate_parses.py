"""Every retained outcome must parse. The test whose absence hid a defect.

`generate()` appended `EOS` unconditionally at the stopping boundary. A delta
is the PREFIX of its group, so that boundary has an open group with no
actions, and `EOS` closes an open group -- so every soft-stop and hard-cap
excerpt was unparseable. Twelve generator tests passed anyway, because none
of them took a long excerpt through `parse()`.

The property, stated once: for any model, any seed, any sampling setting, a
retained outcome that is not `invalid` decodes. The harness may not be the
reason an excerpt fails.
"""
import json
import random
from pathlib import Path

import pytest

from rse.snes_v2 import delta as dx
from rse.snes_v2 import generate as gen
from rse.snes_v2 import grammar as gx
from rse.snes_v2 import statistics as sx


@pytest.fixture(scope="module")
def vocab():
    raw = json.loads((Path.home() /
                      "workplace/v2_artifacts/delta_vocab_k512.json").read_text())
    return gx.Vocab(delta=dx.DeltaCodec(
        direct={int(k): v for k, v in raw["direct"].items()},
        n_observed=raw["n_distinct_train"]))


def uniform_model(vocab):
    """A model with no musical knowledge at all: uniform over the vocabulary.

    Deliberately the worst case. The mask is the only thing shaping its
    output, so any grammar defect in the harness shows up immediately
    instead of being hidden by a model that happens to sample EOS early.
    """
    flat = [1.0 / vocab.size] * vocab.size
    return lambda context: list(flat)


@pytest.mark.parametrize("seed", [1000, 1001, 1002, 1003, 7, 99])
@pytest.mark.parametrize("top_p", [1.0, 0.95])
def test_a_retained_outcome_decodes(vocab, seed, top_p):
    excerpt = gen.generate(
        uniform_model(vocab), vocab,
        gen.SamplingConfig(temperature=1.0, top_p=top_p, seed=seed),
        arm="uniform")
    assert excerpt.outcome in ("complete", "short", "budget_exhausted",
                              "invalid")
    if excerpt.outcome == "invalid":
        assert excerpt.invalid_reason, "an invalid outcome needs its reason"
        return
    parsed, _ = gx.parse(excerpt.tokens, vocab)
    notes = gx.notes_from_groups(parsed)
    assert notes, f"{excerpt.outcome} excerpt decoded to no notes"
    assert sx.excerpt_statistics(notes) is not None


def test_a_soft_stop_excerpt_decodes_and_is_reported_with_its_duration(vocab):
    """The specific case that was broken: termination by the clock, not EOS."""
    found = None
    for seed in range(40):
        excerpt = gen.generate(
            uniform_model(vocab), vocab,
            gen.SamplingConfig(temperature=1.0, top_p=1.0, seed=seed),
            arm="uniform")
        if excerpt.outcome == "complete" and "EOS at" not in excerpt.termination_reason:
            found = excerpt
            break
    assert found is not None, "no clock-terminated excerpt in 40 seeds"
    assert "soft stop" in found.termination_reason or \
           "hard cap" in found.termination_reason
    parsed, _ = gx.parse(found.tokens, vocab)
    assert gx.notes_from_groups(parsed)
    assert found.duration_sec > 0


def test_the_terminal_group_is_closed_with_actions_not_an_appended_eos(vocab):
    """The stream must end `... action(s) EOS`, never `... DELTA EOS`."""
    for seed in range(20):
        excerpt = gen.generate(
            uniform_model(vocab), vocab,
            gen.SamplingConfig(temperature=1.0, top_p=1.0, seed=seed),
            arm="uniform")
        if excerpt.outcome == "invalid":
            continue
        assert excerpt.tokens[-1] == vocab.EOS
        assert excerpt.tokens[-2] < vocab.delta_base, (
            f"seed {seed}: EOS follows a delta token, which closes an empty "
            f"group; last tokens {excerpt.tokens[-4:]}")


def test_terminal_releases_are_counted_rather_than_hidden(vocab):
    """Closing the group is an intervention, so it appears in the record."""
    excerpts = [gen.generate(
        uniform_model(vocab), vocab,
        gen.SamplingConfig(temperature=1.0, top_p=1.0, seed=seed),
        arm="uniform") for seed in range(12)]
    clock_stopped = [e for e in excerpts
                     if e.outcome == "complete"
                     and "EOS at" not in e.termination_reason]
    assert clock_stopped, "no clock-terminated excerpt to check"
    for excerpt in clock_stopped:
        record = excerpt.to_dict()
        assert "terminal_releases" in record
        assert "exceeded_hard_cap" in record
        if excerpt.terminal_releases:
            assert "terminal release" in excerpt.termination_reason


def test_an_overshooting_delta_is_flagged_rather_than_silently_in_target(vocab):
    """Section 5 stops at the NEXT boundary, so one sampled delta can jump
    past 60 s. Conformant, but it must not read as an in-target excerpt."""
    for seed in range(60):
        excerpt = gen.generate(
            uniform_model(vocab), vocab,
            gen.SamplingConfig(temperature=1.0, top_p=1.0, seed=seed),
            arm="uniform")
        if excerpt.duration_sec > gen.HARD_CAP_SEC:
            assert excerpt.exceeded_hard_cap is True
            assert excerpt.outcome == "complete"
            return
    pytest.skip("no overshoot in 60 seeds")


# --- budget exhaustion is a DECODABLE outcome, per Amendment A1 -------------

@pytest.mark.parametrize("budget", [40, 120, 400])
@pytest.mark.parametrize("seed", [1000, 1001, 1002, 1003])
def test_a_budget_exhausted_outcome_decodes(vocab, seed, budget):
    """A1: `short` and `budget_exhausted` remain decodable outcomes.

    They were not. Breaking out of the loop on the budget left the stream
    with no EOS, `parse` rejected it, and the candidate would have been
    ineligible for the harness's reason. The budget is overridden here
    rather than in the procedure, so the frozen 6,000 is untouched.
    """
    excerpt = gen.generate(
        uniform_model(vocab), vocab,
        gen.SamplingConfig(temperature=1.0, top_p=1.0, seed=seed),
        arm="uniform", token_budget=budget)
    if excerpt.outcome == "invalid":
        assert excerpt.invalid_reason
        return
    if excerpt.outcome == "budget_exhausted" and excerpt.invalid_reason:
        # The one honest non-decodable case: the stop fell inside the very
        # first group, so there is no completed group to terminate after.
        assert "invent" in excerpt.invalid_reason
        return
    parsed, _ = gx.parse(excerpt.tokens, vocab)
    assert gx.notes_from_groups(parsed)


def test_budget_exhaustion_is_actually_reachable_in_this_test(vocab):
    """Guards against the above passing because the branch never ran."""
    outcomes = {gen.generate(
        uniform_model(vocab), vocab,
        gen.SamplingConfig(temperature=1.0, top_p=1.0, seed=seed),
        arm="uniform", token_budget=120).outcome for seed in range(12)}
    assert "budget_exhausted" in outcomes, outcomes
