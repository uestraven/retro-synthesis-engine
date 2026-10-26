"""The generator page, and its contract with the API.

There is no browser in the test environment, so these check the two things a
browser would have caught and that static review would not.

**Field names.** Every `song.x` and `caps.x` the page reads must exist in a
real response. A typo there is invisible until someone clicks Generate.

**No second source of truth.** The page must not carry its own copy of the
style list, the duration limits or the format list. A frontend that hard-codes
them drifts from the composer the first time a preset changes, and then the
dropdown offers music the engine cannot write.

The page was additionally driven in headless Chromium by hand during
development, which is how the tempo-slider defect below was found.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rse.api.settings import Settings

PAGE = Path("frontend/index.html")


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    from rse.api.app import create_app

    return TestClient(
        create_app(
            Settings(
                storage_dir=tmp_path, rate_per_minute=100000, rate_burst=1000
            )
        )
    )


@pytest.fixture(scope="module")
def source() -> str:
    return PAGE.read_text()


def test_the_api_serves_the_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Retro Synthesis Engine" in response.text


def test_the_page_is_self_contained(source: str) -> None:
    """No build step and no third-party fetch: one file, offline-capable."""
    remote = re.findall(r'(?:src|href)="(https?://[^"]+)"', source)
    assert remote == [], f"page fetches remote assets: {remote}"
    assert "<script" in source and "import " not in source.split("<script")[1][:200]


def test_every_song_field_the_page_reads_exists(
    client: TestClient, source: str
) -> None:
    """The contract check. A `song.durationSec` typo would pass review."""
    song = client.post(
        "/api/generate", json={"prompt": "a boss theme", "seed": 1}
    ).json()
    referenced = set(re.findall(r"\bsong\.([a-z_]+)", source))
    missing = sorted(field for field in referenced if field not in song)
    assert missing == [], f"page reads fields the API does not return: {missing}"


def test_every_link_and_requirement_field_exists(
    client: TestClient, source: str
) -> None:
    song = client.post(
        "/api/generate", json={"prompt": "a boss theme", "seed": 1}
    ).json()
    for field in set(re.findall(r"song\.links\.([a-z]+)", source)):
        assert field in song["links"], field
    for field in set(re.findall(r"song\.requirements\.([a-z]+)", source)):
        assert field in song["requirements"], field


def test_every_capability_field_the_page_reads_exists(
    client: TestClient, source: str
) -> None:
    caps = client.get("/api/capabilities").json()
    for field in set(re.findall(r"\bcaps\.([a-z_]+)", source)):
        assert field in caps, field


def test_the_page_does_not_hard_code_the_style_list(source: str) -> None:
    from rse.procedural.compose import STYLES

    for name in STYLES:
        assert f'value="{name}"' not in source, (
            f"style {name!r} is hard-coded; read it from /api/capabilities"
        )
    assert "/api/capabilities" in source


def test_the_page_does_not_hard_code_the_console_list(source: str) -> None:
    """Whether SNES is available is the server's answer, not the page's."""
    assert "caps.consoles" in source
    assert "console_.available" in source


def test_the_tempo_slider_can_reach_its_style_default(source: str) -> None:
    """The defect a browser found.

    The auto position is one step below the real tempo floor. With `min="60"`
    in the markup the browser clamps the initial value up to 60 at parse time,
    so the auto position was unreachable and a slider nobody touched silently
    overrode every preset's tempo with 60 BPM.
    """
    tempo = re.search(r'<input type="range" id="tempo"[^>]*>', source)
    assert tempo, "tempo slider not found"
    tag = tempo.group(0)
    minimum = int(re.search(r'min="(\d+)"', tag).group(1))
    value = int(re.search(r'value="(\d+)"', tag).group(1))
    assert minimum < 60, "the auto position must be below the tempo floor"
    assert value == minimum, "the slider must start at the auto position"
    # And the request must omit tempo entirely when it sits there.
    assert "if (tempo >= 60) request.tempo_bpm = tempo" in source


def test_the_duration_slider_is_bounded_by_the_server(source: str) -> None:
    assert "duration.min = caps.min_seconds" in source
    assert "duration.max = caps.max_seconds" in source


def test_interpolated_text_is_escaped(source: str) -> None:
    """The prompt comes back in the report; it must not be injected as HTML."""
    assert "const escape_" in source
    for interpolation in re.findall(r"\$\{([^}]+)\}", source):
        if "item." in interpolation or "note" in interpolation:
            assert "escape_" in interpolation, interpolation


def test_the_page_shows_what_the_request_could_not_reach(source: str) -> None:
    """RSE's reporting is the point; burying the unsupported bucket would make
    the page claim more than the composer did."""
    assert "requirements.unsupported" in source
    assert "Nothing takes this as input" in source


def test_the_page_offers_both_seed_actions(source: str) -> None:
    assert 'id="again"' in source and 'id="same"' in source
    assert "Generate another variation" in source


def test_errors_are_shown_from_the_api_not_invented(source: str) -> None:
    """The API's refusals are specific; a generic "failed" hides them."""
    assert "payload.message" in source
    assert "retry_after_seconds" in source


def test_the_page_names_no_amazon_internal_system(source: str) -> None:
    """This repository is public. Nothing internal belongs in a shipped page."""
    lowered = source.lower()
    for term in ("amazon", "a2z", "midway", "brazil", "apollo", "corp."):
        assert term not in lowered, term
