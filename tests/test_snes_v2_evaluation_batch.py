"""The evaluation batch refuses to run at an unregistered setting.

§9: "No temperature or top-p change. No alternative seeds. No handpicked
examples. No regeneration of an unflattering excerpt." A script that takes
the settings as flags can violate the first of those with a typo and leave no
trace. So the settings are read from the arm's selection report and asserted
against the registered values, and these tests drive that refusal.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from rse.snes_v2 import generate as gen  # noqa: E402


@pytest.fixture(scope="module")
def batch():
    spec = importlib.util.spec_from_file_location(
        "eval_batch", ROOT / "scripts/snes_v2_evaluation_batch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def workspace(tmp_path):
    source = Path.home() / "workplace/v2_artifacts"
    (tmp_path / "delta_vocab_k512.json").write_text(
        (source / "delta_vocab_k512.json").read_text())
    (tmp_path / "MANIFEST.json").write_text(
        (source / "windows/MANIFEST.json").read_text())
    return tmp_path


def write_report(path, *, arm="baseline", temperature=0.8, top_p=0.95,
                 successful=True):
    path.write_text(json.dumps({
        "arm": arm, "successful": successful,
        "selected": {"temperature": temperature, "top_p": top_p},
    }))
    return path


def excerpt(duration, outcome="complete", tokens=()):
    return gen.Excerpt(
        seed=0, arm="baseline", tokens=list(tokens), outcome=outcome,
        termination_reason="stub", duration_sec=duration, n_groups=3,
        n_rollovers=0, illegal_mass_mean=0.01, illegal_mass_max=0.1,
        forced_closures=0, allocator_steals=0)


def run(batch, workspace, monkeypatch, report, *, make_excerpt,
        statistics=None):
    calls = {"seeds": []}

    def fake_generate(distribution, vocab, config, **kwargs):
        calls["seeds"].append(config.seed)
        return make_excerpt(config)

    monkeypatch.setattr(batch.gen, "generate", fake_generate)
    monkeypatch.setattr(sys, "argv", [
        "snes_v2_evaluation_batch.py", "--arm", "baseline",
        "--selection-report", str(report),
        "--windows", str(workspace),
        "--delta-vocab", str(workspace / "delta_vocab_k512.json"),
        "--out", str(workspace / "eval_out")])
    try:
        batch.main()
        code = 0
    except SystemExit as exit_info:
        code = exit_info.code
        if isinstance(code, str):          # a refusal message, not a status
            raise
    path = workspace / "eval_out/evaluation_batch_baseline.json"
    payload = json.loads(path.read_text()) if path.exists() else None
    return code, payload, calls["seeds"]


@pytest.fixture(autouse=True)
def _stub_baseline(batch, monkeypatch):
    """The n-gram refit reads 4.5 MB of windows; irrelevant to these tests.

    Patchable only because the loaders live in `rse.snes_v2.runners` now. The
    earlier `importlib` copy inside main() could not be substituted, which is
    the same shape as a test that silently never exercises its target.
    """
    monkeypatch.setattr(batch.rx, "baseline_distribution",
                        lambda window_dir, vocab_size:
                        (lambda context: [], {"family": "stub"}))


def test_the_registered_seed_list_is_zero_through_twentythree(batch):
    assert batch.EVALUATION_SEEDS == tuple(range(24))
    assert len(batch.EVALUATION_SEEDS) == 24
    assert not set(batch.EVALUATION_SEEDS) & {1000, 1001, 1002, 1003}
    assert batch.WINDOW_SEC == 20.0
    assert batch.N_EVALUATION_SEEDS == 24
    assert batch.REGISTERED_SETTINGS == {
        "model": {"temperature": 1.0, "top_p": 1.0},
        "baseline": {"temperature": 0.8, "top_p": 0.95}}


def test_a_report_with_an_unregistered_setting_is_refused(
        batch, workspace, monkeypatch):
    """The §9 guard: a drifted temperature stops the run, it does not
    silently produce a batch nobody selected."""
    report = write_report(workspace / "sel.json", temperature=1.1)
    with pytest.raises(SystemExit) as info:
        run(batch, workspace, monkeypatch, report,
            make_excerpt=lambda c: excerpt(30.0))
    assert "Refusing to generate at an unregistered setting" in str(info.value)


def test_a_report_for_the_wrong_arm_is_refused(batch, workspace, monkeypatch):
    report = write_report(workspace / "sel.json", arm="model",
                          temperature=1.0, top_p=1.0)
    with pytest.raises(SystemExit) as info:
        run(batch, workspace, monkeypatch, report,
            make_excerpt=lambda c: excerpt(30.0))
    assert "is for arm" in str(info.value)


def test_an_unsuccessful_selection_cannot_be_generated_from(
        batch, workspace, monkeypatch):
    """A1 §6: if selection failed, no fallback is chosen. That must also mean
    no batch is produced at a setting that was never selected."""
    report = write_report(workspace / "sel.json", successful=False)
    with pytest.raises(SystemExit) as info:
        run(batch, workspace, monkeypatch, report,
            make_excerpt=lambda c: excerpt(30.0))
    assert "declared unsuccessful" in str(info.value)


def test_every_seed_yields_exactly_one_retained_outcome(
        batch, workspace, monkeypatch):
    report = write_report(workspace / "sel.json")
    code, payload, seeds = run(
        batch, workspace, monkeypatch, report,
        make_excerpt=lambda c: excerpt(30.0, outcome="invalid"))
    assert seeds == list(range(24)), "seeds must be the registered list, in order"
    assert len(payload["excerpts"]) == 24
    assert payload["denominators"]["attempted_outcomes"] == 24
    assert len({e["identifier"] for e in payload["excerpts"]}) == 24
    for record in payload["excerpts"]:
        assert (workspace / f"eval_out/{record['identifier']}.json").exists()
    assert code == 0, "a batch of retained failures is still a complete batch"


def test_window_reachability_is_recorded_per_excerpt(
        batch, workspace, monkeypatch):
    report = write_report(workspace / "sel.json")
    durations = iter([5.0, 20.0, 19.999, 45.0] * 6)
    code, payload, _ = run(
        batch, workspace, monkeypatch, report,
        make_excerpt=lambda c: excerpt(next(durations), outcome="invalid"))
    reach = [e["reaches_window"] for e in payload["excerpts"]]
    assert reach[:4] == [False, True, False, True], "20.0s must be inclusive"
    assert payload["denominators"]["reaching_window"] == 12
    assert all(e["window_sec"] == 20.0 for e in payload["excerpts"])


def test_a_short_population_exits_nonzero(batch, workspace, monkeypatch):
    """Truncating the seed list must fail the accounting.

    It did not, originally: `intended=len(EVALUATION_SEEDS)` reconciled
    against itself, so the check could never fire. N_EVALUATION_SEEDS is a
    literal for exactly this reason, and is left at 24 here.
    """
    monkeypatch.setattr(batch, "EVALUATION_SEEDS", tuple(range(20)))
    assert batch.N_EVALUATION_SEEDS == 24
    report = write_report(workspace / "sel.json")
    code, payload, seeds = run(
        batch, workspace, monkeypatch, report,
        make_excerpt=lambda c: excerpt(30.0, outcome="invalid"))
    assert len(seeds) == 20
    assert code != 0
    assert payload["denominators"]["attempted_outcomes"] == 20
    assert payload["denominators"]["expected_attempted_outcomes"] == 24
