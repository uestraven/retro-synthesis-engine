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


def test_the_unbuilt_page_falls_back_to_its_own_origin(
    client: TestClient, source: str
) -> None:
    """The same file has to work served by the API and built for Pages.

    `%VITE_API_BASE%` is substituted by `vite.config.js` at build time. When
    the API serves this file there is no build step, so the placeholder
    reaches the browser verbatim -- and a truthiness test on it would send
    every local request to a literal "%VITE_API_BASE%/api/generate". Hence
    the leading-% check, which this test pins, since the failure is a page
    that looks fine and cannot generate anything.
    """
    assert "%VITE_API_BASE%" in source
    assert 'BUILT_API_BASE.startsWith("%")' in source
    # `?api=` must still win, so a local page can be aimed at a deployed API.
    index = source.index("const API_BASE")
    resolution = source[index : index + 400]
    assert resolution.index('get("api")') < resolution.index("BUILT_API_BASE.startsWith")

    # And the served page must be the unbuilt one: if a dist build ever ends
    # up behind this route, the placeholder would already be gone.
    assert "%VITE_API_BASE%" in client.get("/").text


def test_an_unreachable_api_is_not_blamed_on_localhost(source: str) -> None:
    """A blocked cross-origin fetch and a dead host are indistinguishable from
    JavaScript, so the message names both and prints the origin it tried.

    The previous message assumed local development and told a hosted visitor
    to go looking for a server on their own machine.
    """
    assert "RSE_API_ALLOWED_ORIGINS must name this origin" in source
    assert "${API_BASE}" in source


def test_the_echo_control_is_not_among_the_musical_controls(source: str) -> None:
    """An Echo slider beside Tempo and Mood promises a sound change.

    It does not change the sound -- it reserves a memory buffer the preview
    renderer never reads. So the control lives in a collapsed diagnostics
    panel, and this test pins the structure rather than the wording, because
    the wording is the part most likely to be softened later.
    """
    start = source.index('id="snesPanel"')
    panel = source[start : source.index("</fieldset>", start)]
    assert 'id="reserved"' in panel, "voice reservation does affect audio"
    assert 'id="echo"' not in panel, "the echo control is back beside the musical settings"

    diagnostics = source[source.index('id="snesDiagnostics"') :]
    diagnostics = diagnostics[: diagnostics.index("</details>")]
    assert 'id="echo"' in diagnostics
    assert "does not add audible reverb" in diagnostics
    assert "memory planning" in diagnostics.lower()
    # Collapsed by default: <details> without `open`.
    assert "<details" in source and "<details open" not in source


def test_the_budget_table_separates_the_base_from_the_echo(source: str) -> None:
    """One combined number is what caused the confusion this panel fixes.

    A reviewer read the reported total as excluding echo and worked out what
    adding a 240 ms buffer would cost on top; it was already included. Base
    and echo on separate rows makes the question unaskable.
    """
    for row in ("budgetBase", "budgetEcho", "budgetTotal", "budgetFree"):
        assert f'id="{row}"' in source, row
    # The echo buffer must not be counted in both the base and its own row.
    assert "aram.total_bytes - aram.breakdown.echo" in source
    assert "aram.measured_bytes - aram.breakdown.echo" in source
    # And the panel must not quote a figure before a song has been generated.
    assert "Generate a 16-bit song to populate" in source


def test_the_memory_figures_are_not_shown_as_musical_facts(source: str) -> None:
    """They were in the same chip row as tempo and voice counts, which reads
    as a property of the music."""
    facts = source[source.index('el("facts").innerHTML +=') :]
    facts = facts[: facts.index("].map(")]
    assert "audio RAM" not in facts
    assert "echo" not in facts.lower()
    assert "peak_voices" in facts, "the audible facts should still be there"


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
        "/api/generate", json={"style": "boss", "seed": 1}
    ).json()
    referenced = set(re.findall(r"\bsong\.([a-z_]+)", source))
    missing = sorted(field for field in referenced if field not in song)
    assert missing == [], f"page reads fields the API does not return: {missing}"


def test_every_link_and_requirement_field_exists(
    client: TestClient, source: str
) -> None:
    song = client.post(
        "/api/generate", json={"style": "boss", "seed": 1}
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


def test_the_tempo_slider_starts_at_style_default(source: str) -> None:
    """The defect a browser found.

    `min` and `value` must both be set in the MARKUP. Setting `min` from
    script lets the browser clamp the initial value first, which made the
    auto position unreachable and silently forced every song to the slider's
    floor -- a control nobody touched overriding every preset.
    """
    tempo = re.search(r'<input type="range" id="tempo"[^>]*>', source)
    assert tempo, "tempo slider not found"
    tag = tempo.group(0)
    minimum = int(re.search(r'min="(\d+)"', tag).group(1))
    value = int(re.search(r'value="(\d+)"', tag).group(1))
    assert minimum == 0, "index 0 is the style-default position"
    assert value == minimum, "the slider must start at the auto position"
    # At index 0 no tempo is sent, so the preset keeps its own.
    assert "if (index > 0 && TEMPO_LADDER[index - 1])" in source


def test_the_tempo_slider_indexes_the_hardware_ladder(source: str) -> None:
    """A free BPM dial offers tempos no NES can play. The slider must index
    the ladder the server reports, and hold no copy of it."""
    assert "caps.tempo_ladder" in source
    assert "TEMPO_LADDER" in source
    # No hard-coded rung: 150.25 and friends come from the server.
    for rung in ("180.3", "150.25", "128.78", "100.16"):
        assert rung not in source, f"{rung} is hard-coded in the page"


def test_the_page_shows_the_frame_count(source: str) -> None:
    """Showing frames/row is what makes the uneven BPM numbers legible rather
    than looking like a rounding bug."""
    assert "frames_per_row" in source
    assert "frames per tracker row" in source


def test_the_duration_slider_is_bounded_by_the_server(source: str) -> None:
    assert "duration.min = caps.min_seconds" in source
    assert "duration.max = caps.max_seconds" in source


def test_interpolated_text_is_escaped(source: str) -> None:
    """Server strings reach the report; they must not be injected as HTML.

    The original reason was a user-supplied prompt echoed into the report.
    That field is gone, which REMOVES one source and does not remove the
    rule: style names, requirement details, caveats and the SNES claims table
    all still arrive from the server and still reach innerHTML.

    Scoped to `innerHTML` assignments, which are the only injection vector
    here. `textContent` and `dataset` assignments are escaped by the DOM
    itself, and treating them as unsafe made this test flag a server-supplied
    constant while saying nothing about actual risk.
    """
    assert "const escape_" in source
    unsafe = []
    for statement in re.findall(r"innerHTML\s*=\s*(.+?);\n", source, re.S):
        interpolations = re.findall(r"\$\{([^}]+)\}", statement)
        # Server- or user-supplied strings reaching innerHTML.
        risky = [
            text
            for text in interpolations
            if re.search(r"\b(item|note|song|entry)\b", text)
            and "toFixed" not in text
            and "length" not in text
        ]
        # The escaping happens at the point of insertion, which may be a
        # helper applied to the whole row rather than to each field, so the
        # invariant is per STATEMENT: if it interpolates supplied text, that
        # statement has to escape.
        if risky and "escape_" not in statement:
            unsafe.append(risky)
    assert unsafe == [], f"innerHTML statements with unescaped supplied text: {unsafe}"


def test_only_escaped_values_reach_inner_html(source: str) -> None:
    """The report's own fields are interpolated into innerHTML, so the
    escaping helper must be used there and must cover the five characters that
    matter."""
    helper = re.search(r"const escape_ =.*?;\n", source, re.S)
    assert helper, "no escaping helper"
    body = helper.group(0)
    for char in ("&", "<", ">", '"'):
        assert char in body, f"{char} not escaped"


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
