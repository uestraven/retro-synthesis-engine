"""The driver's own exit path and manifest, exercised end to end.

The library tests in `test_snes_v2_selection.py` cover the eligibility rule.
These cover the part a library test cannot: that the SCRIPT exits nonzero on
a declared selection failure and on a short attempted population, and that
its manifest keeps every outcome and both denominators.

The model is stubbed by monkeypatching `generate`, not by a flag in the
shipped driver -- the frozen procedure's driver gets no debug path.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from rse.snes_v2 import generate as gen  # noqa: E402
from rse.snes_v2 import selection as sel  # noqa: E402


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location(
        "select_driver", ROOT / "scripts/snes_v2_select_sampling.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(tmp_path):
    """The three inputs the driver reads, minimal but real in shape."""
    # Prefer the bundle's own copies when this test runs from an extracted
    # bundle, so bundle validation is about the bundled bytes.
    source = (ROOT if (ROOT / "delta_vocab_k512.json").exists()
              else Path.home() / "workplace/v2_artifacts")
    (tmp_path / "delta_vocab_k512.json").write_text(
        (source / "delta_vocab_k512.json").read_text())
    (tmp_path / "reference_stats.json").write_text(
        (source / "reference_stats.json").read_text())
    (tmp_path / "MANIFEST.json").write_text(
        (source / "windows/MANIFEST.json").read_text())
    return tmp_path


def excerpt(outcome, *, tokens=(), reason=""):
    return gen.Excerpt(
        seed=0, arm="model", tokens=list(tokens), outcome=outcome,
        termination_reason="stub", duration_sec=47.5, n_groups=3,
        n_rollovers=0, illegal_mass_mean=0.01, illegal_mass_max=0.2,
        forced_closures=0, allocator_steals=0, invalid_reason=reason)


def run(driver, bundle, monkeypatch, *, make_excerpt):
    """Run the real main() with a stubbed model and generator."""
    calls = {"n": 0}

    def fake_generate(distribution, vocab, config, **kwargs):
        calls["n"] += 1
        return make_excerpt(config)

    monkeypatch.setattr(driver.gen, "generate", fake_generate)
    monkeypatch.setattr(
        driver, "baseline_distribution",
        lambda window_dir, vocab_size: (lambda context: [], {"family": "stub"}))
    monkeypatch.setattr(sys, "argv", [
        "snes_v2_select_sampling.py", "--arm", "baseline",
        "--windows", str(bundle),
        "--delta-vocab", str(bundle / "delta_vocab_k512.json"),
        "--reference", str(bundle / "reference_stats.json"),
        "--out", str(bundle / "out")])
    try:
        driver.main()
        code = 0
    except SystemExit as exit_info:
        code = exit_info.code
    report_path = bundle / "out/sampling_selection_baseline.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    return code, report, calls["n"]


def test_an_entirely_invalid_grid_exits_nonzero_from_the_driver(
        driver, bundle, monkeypatch):
    code, report, n_calls = run(
        driver, bundle, monkeypatch,
        make_excerpt=lambda config: excerpt("invalid", reason="no legal token"))
    assert code == 3, "the driver must not exit 0 on a declared failure"
    assert n_calls == 40
    assert report["successful"] is False
    assert report["selected"] is None
    assert "declared unsuccessful" in report["failure_reason"]


def test_the_failure_manifest_keeps_both_denominators_and_every_outcome(
        driver, bundle, monkeypatch):
    code, report, _ = run(
        driver, bundle, monkeypatch,
        make_excerpt=lambda config: excerpt("invalid", reason="no legal token"))
    assert code == 3
    assert report["denominators"]["attempted_outcomes"] == 40
    assert report["denominators"]["expected_attempted_outcomes"] == 40
    assert report["denominators"]["eligible_candidates"] == 0
    assert report["denominators"]["candidates"] == 10
    assert report["outcome_tally"] == {"complete": 0, "short": 0,
                                       "budget_exhausted": 0, "invalid": 40}
    assert len(report["excerpts"]) == 40
    assert len({e["identifier"] for e in report["excerpts"]}) == 40
    # Every retained outcome is on disk under its predetermined name.
    for record in report["excerpts"]:
        assert (bundle / f"out/{record['identifier']}.json").exists()


def test_a_short_attempted_population_exits_nonzero(
        driver, bundle, monkeypatch):
    """An outcome silently not retained must fail the accounting, not pass."""
    monkeypatch.setattr(sel, "DEVELOPMENT_SEEDS", (1000, 1001, 1002))
    code, report, n_calls = run(
        driver, bundle, monkeypatch,
        make_excerpt=lambda config: excerpt("invalid"))
    assert n_calls == 30
    assert code != 0
    assert report["denominators"]["attempted_outcomes"] == 30
    assert report["denominators"]["expected_attempted_outcomes"] == 40


def test_a_successful_grid_exits_zero_and_records_the_selection(
        driver, bundle, monkeypatch):
    """A decodable scripted performance per seed, so selection succeeds."""
    from rse.snes_v2 import delta as dx
    from rse.snes_v2 import grammar as gx

    raw = json.loads((bundle / "delta_vocab_k512.json").read_text())
    vocab = gx.Vocab(delta=dx.DeltaCodec(
        direct={int(k): v for k, v in raw["direct"].items()},
        n_observed=raw["n_distinct_train"]))

    from rse.snes_v2.scripted import reference_performance
    tokens = reference_performance(vocab)

    code, report, n_calls = run(
        driver, bundle, monkeypatch,
        make_excerpt=lambda config: excerpt("complete", tokens=tokens))
    assert code == 0, f"exited {code} on a fully eligible grid"
    assert n_calls == 40
    assert report["successful"] is True
    assert report["denominators"]["eligible_candidates"] == 10
    # Every candidate scored identically, so the tie-break decides: lowest
    # temperature, then highest top-p.
    assert (report["selected"]["temperature"],
            report["selected"]["top_p"]) == (0.8, 1.0)
    assert report["no_audio_heard"] is True
    assert report["no_test_data_touched"] is True
