"""Milestone 4 generation: eligibility, endpoint separation, feasibility."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from rse.snes_v2 import generate as gen  # noqa: E402


@pytest.fixture(scope="module")
def m4():
    spec = importlib.util.spec_from_file_location(
        "m4_gen", ROOT / "scripts/snes_v2_m4_generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def workspace(tmp_path):
    source = Path.home() / "workplace/v2_artifacts"
    (tmp_path / "delta_vocab_k512.json").write_text(
        (source / "delta_vocab_k512.json").read_text())
    (tmp_path / "sel.json").write_text(json.dumps({
        "arm": "model", "successful": True,
        "selected": {"temperature": 1.0, "top_p": 1.0}}))
    return tmp_path


def excerpt(duration, outcome="complete", tokens=(), seed=0):
    return gen.Excerpt(
        seed=seed, arm="model", tokens=list(tokens), outcome=outcome,
        termination_reason="stub", duration_sec=duration, n_groups=3,
        n_rollovers=0, illegal_mass_mean=0.01, illegal_mass_max=0.1,
        forced_closures=0, allocator_steals=0)


def run(m4, workspace, monkeypatch, *, durations, decodable=True,
        report="sel.json"):
    """Drive main() with a stubbed model. `durations` cycles."""
    from rse.snes_v2 import delta as dx, grammar as gx, scripted
    raw = json.loads((workspace / "delta_vocab_k512.json").read_text())
    vocab = gx.Vocab(delta=dx.DeltaCodec(
        direct={int(k): v for k, v in raw["direct"].items()},
        n_observed=raw["n_distinct_train"]))
    good_tokens = scripted.reference_performance(vocab)

    supply = iter(durations)
    monkeypatch.setattr(
        m4.gen, "generate",
        lambda d, v, c, **k: excerpt(
            next(supply), tokens=good_tokens if decodable else [],
            seed=c.seed))
    monkeypatch.setattr(
        m4.rx, "transformer_distribution",
        lambda checkpoint, device: (
            lambda context: [],
            {"step": 5400, "n_parameters": 5_646_336, "vocab_size": 2544,
             "hardware": {"device": device}}))
    monkeypatch.setattr(sys, "argv", [
        "snes_v2_m4_generate.py",
        "--selection-report", str(workspace / report),
        "--delta-vocab", str(workspace / "delta_vocab_k512.json"),
        "--out", str(workspace / "m4_out")])
    try:
        m4.main()
        code = 0
    except SystemExit as info:
        code = info.code
        if isinstance(code, str):
            raise
    path = workspace / "m4_out/m4_generation.json"
    return code, (json.loads(path.read_text()) if path.exists() else None)


def test_the_frozen_seed_population_is_disjoint_from_examined_seeds(m4):
    assert m4.M4_SEEDS == tuple(range(3000, 3036))
    assert m4.N_M4_SEEDS == 36
    assert not set(m4.M4_SEEDS) & m4.FORBIDDEN_SEEDS
    assert m4.FORBIDDEN_SEEDS >= set(range(24)) | {1000, 1001, 1002, 1003}
    assert m4.REQUIRED_PRIMARY_PAIRS == 24
    assert m4.WINDOW_SEC == 20.0 and m4.TARGET_SEC == 45.0


def test_an_unregistered_setting_is_refused(m4, workspace, monkeypatch):
    (workspace / "drift.json").write_text(json.dumps({
        "arm": "model", "successful": True,
        "selected": {"temperature": 0.9, "top_p": 1.0}}))
    with pytest.raises(SystemExit, match="Refusing to generate"):
        run(m4, workspace, monkeypatch, durations=[30.0] * 36,
            report="drift.json")


def test_endpoint_a_is_unconditional_over_every_seed(m4, workspace,
                                                     monkeypatch):
    """12 reach 45s, 12 more reach 20s, 12 fall short."""
    durations = [46.0] * 12 + [25.0] * 12 + [3.0] * 12
    code, payload = run(m4, workspace, monkeypatch, durations=durations)
    a = payload["endpoint_a"]
    assert a["denominator"] == 36 and a["unconditional"] is True
    assert a["reaching_45s"]["count"] == 12
    assert a["reaching_20s"]["count"] == 24
    assert a["decodable"]["count"] == 36
    assert a["reaching_45s"]["rate"] == pytest.approx(12 / 36)
    low, high = a["reaching_45s"]["ci95"]
    assert low < 12 / 36 < high, "the exact interval must bracket the rate"
    assert len(payload["excerpts"]) == 36


def test_an_undecodable_long_excerpt_is_not_eligible(m4, workspace,
                                                     monkeypatch):
    """A 25s stream that will not parse is not a stimulus."""
    code, payload = run(m4, workspace, monkeypatch,
                        durations=[30.0] * 36, decodable=False)
    assert payload["endpoint_a"]["reaching_20s"]["count"] == 36
    assert payload["endpoint_a"]["decodable"]["count"] == 0
    assert payload["endpoint_b_stimulus_set"]["eligible_count"] == 0
    assert payload["endpoint_b_stimulus_set"]["feasible"] is False
    assert code == 4


def test_the_stimulus_set_is_the_first_24_eligible_by_seed_order(
        m4, workspace, monkeypatch):
    """Not the longest, not the best-scoring: the first, by seed."""
    durations = [5.0] + [46.0] * 35          # seed 3000 falls short
    code, payload = run(m4, workspace, monkeypatch, durations=durations)
    b = payload["endpoint_b_stimulus_set"]
    assert b["eligible_count"] == 35
    assert b["selected_seeds"] == list(range(3001, 3025))
    assert b["feasible"] is True
    assert code == 0


def test_too_few_eligible_exits_nonzero_and_does_not_relax_anything(
        m4, workspace, monkeypatch):
    durations = [46.0] * 23 + [2.0] * 13     # 23 eligible, 24 required
    code, payload = run(m4, workspace, monkeypatch, durations=durations)
    b = payload["endpoint_b_stimulus_set"]
    assert b["eligible_count"] == 23
    assert b["required_pairs"] == 24, "the requirement must not move"
    assert b["feasible"] is False
    assert "NOT FEASIBLE" in b["verdict"]
    assert "without redefining success" in b["verdict"]
    assert code == 4
    # Endpoint A still reported in full, not hidden by the failure.
    assert payload["endpoint_a"]["denominator"] == 36
    assert len(payload["excerpts"]) == 36


def test_a_seed_mismatch_is_refused_rather_than_mis_attributed(
        m4, workspace, monkeypatch):
    """The identifier and the recorded seed must come from one source."""
    from rse.snes_v2 import delta as dx, grammar as gx, scripted
    raw = json.loads((workspace / "delta_vocab_k512.json").read_text())
    vocab = gx.Vocab(delta=dx.DeltaCodec(
        direct={int(k): v for k, v in raw["direct"].items()},
        n_observed=raw["n_distinct_train"]))
    monkeypatch.setattr(
        m4.gen, "generate",
        lambda d, v, c, **k: excerpt(
            46.0, tokens=scripted.reference_performance(vocab), seed=999))
    monkeypatch.setattr(
        m4.rx, "transformer_distribution",
        lambda checkpoint, device: (
            lambda context: [],
            {"step": 5400, "n_parameters": 5_646_336, "vocab_size": 2544,
             "hardware": {}}))
    monkeypatch.setattr(sys, "argv", [
        "snes_v2_m4_generate.py",
        "--selection-report", str(workspace / "sel.json"),
        "--delta-vocab", str(workspace / "delta_vocab_k512.json"),
        "--out", str(workspace / "m4_out")])
    with pytest.raises(SystemExit, match="mis-attributed"):
        m4.main()


def test_nothing_is_rendered_or_heard(m4, workspace, monkeypatch):
    code, payload = run(m4, workspace, monkeypatch, durations=[46.0] * 36)
    assert payload["no_audio_rendered"] is True
    assert payload["nothing_heard"] is True
