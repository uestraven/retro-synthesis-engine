"""Interface tests.

The server is never started here. What matters is testable without one: that
the bind defaults to loopback and refuses anything else, and that the display
helpers render real results.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rse.compose import ComposeRequest, compose
from rse.ui.app import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    UiConfig,
    UiError,
    channel_rows,
    check_bind,
    describe,
    find_checkpoints,
    is_loopback,
)

# -- binding ------------------------------------------------------------------


def test_the_default_host_is_loopback():
    """The default is the deliverable, not a placeholder."""
    assert DEFAULT_HOST == "127.0.0.1"
    assert is_loopback(DEFAULT_HOST)
    assert UiConfig().host == "127.0.0.1"
    assert UiConfig().allow_remote is False


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "127.0.0.5", "::1"])
def test_loopback_addresses_are_recognized(host):
    assert is_loopback(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "10.0.0.1", "example.com"])
def test_non_loopback_addresses_are_not(host):
    assert not is_loopback(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10"])
def test_a_non_loopback_bind_is_refused(host):
    with pytest.raises(UiError, match="refusing to bind"):
        check_bind(host)


def test_the_refusal_explains_the_consequence_and_the_alternative():
    with pytest.raises(UiError) as caught:
        check_bind("0.0.0.0")
    message = str(caught.value)
    assert "reach this machine" in message
    assert "--allow-remote" in message
    assert "SSH" in message


def test_exposure_requires_an_explicit_opt_in():
    assert check_bind("0.0.0.0", allow_remote=True) == "0.0.0.0"


def test_loopback_needs_no_opt_in():
    assert check_bind("127.0.0.1") == "127.0.0.1"
    assert DEFAULT_PORT == 7860


# -- checkpoint discovery -----------------------------------------------------


def test_no_checkpoints_is_not_an_error(tmp_path):
    assert find_checkpoints(tmp_path) == []
    assert find_checkpoints(tmp_path / "missing") == []


def test_checkpoints_are_listed_newest_first(tmp_path):
    import os
    import time

    older = tmp_path / "old.pt"
    newer = tmp_path / "new.pt"
    older.write_bytes(b"x")
    time.sleep(0.01)
    newer.write_bytes(b"y")
    os.utime(older, (1, 1))
    assert find_checkpoints(tmp_path) == [str(newer), str(older)]


def test_only_checkpoints_are_listed(tmp_path):
    (tmp_path / "a.pt").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("hi")
    assert find_checkpoints(tmp_path) == [str(tmp_path / "a.pt")]


# -- display ------------------------------------------------------------------


@pytest.fixture
def result(nes_profile, tmp_path):
    return compose(
        ComposeRequest(seed=5, bars=8, mood="dark"),
        out_base=tmp_path / "p",
        profile=nes_profile,
    )


def test_describe_reports_the_essentials(result):
    text = describe(result)
    assert "notes" in text and "bars" in text
    assert "playable as written" in text
    assert "Loop: clean" in text
    # The rule-based plan is shown, since it is the thing a user tweaks.
    assert "dark" in text


def test_describe_handles_a_model_result(result):
    result.sample_info = {"n_tokens": 400, "hit_token_limit": True, "overran_length": True}
    text = describe(result)
    assert "Sampled 400 tokens" in text
    assert "token limit" in text
    assert "held to it" in text


def test_channel_rows_cover_every_channel(result):
    rows = channel_rows(result)
    assert [row[0] for row in rows] == ["P1", "P2", "TR", "NO"]
    assert all(len(row) == 6 for row in rows)
    assert all(row[4] == 1 for row in rows)  # max polyphony


def test_the_ui_adds_no_musical_logic():
    """It is a client of rse.compose; drift between the two is the risk."""
    import inspect

    import rse.ui.app as app

    source = inspect.getsource(app)
    for forbidden in ("make_plan(", "NesApuRenderer(", "ConstraintEngine("):
        assert forbidden not in source, f"{forbidden} belongs in rse.compose"


def test_output_dir_is_under_the_repo_by_default():
    from rse.ui.app import DEFAULT_OUTPUT_DIR

    assert DEFAULT_OUTPUT_DIR.name == "ui"
    assert "out" in DEFAULT_OUTPUT_DIR.parts


def test_building_the_app_requires_gradio_or_says_so(nes_profile, tmp_path):
    gradio = pytest.importorskip("gradio")
    assert gradio is not None
    app = __import__("rse.ui.app", fromlist=["build_app"]).build_app(
        UiConfig(output_dir=tmp_path / "out", checkpoints_dir=tmp_path), nes_profile
    )
    assert app is not None
    # Building must not start a server.
    assert not getattr(app, "is_running", False)


def test_config_accepts_a_custom_output_dir(tmp_path):
    config = UiConfig(output_dir=tmp_path / "custom")
    assert config.output_dir == tmp_path / "custom"
    assert isinstance(config.checkpoints_dir, Path)
