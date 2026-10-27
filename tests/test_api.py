"""The web API, including the parts that must refuse.

A public generator's defects are not musical. They are a control that moves
nothing, a format advertised that the server cannot write, an unimplemented
console quietly answered with NES audio, a seed that does not reproduce, a
filename that escapes the storage directory, and a cost cap that is not
actually enforced. Each of those has a test here.

The bind default is tested too. `RSE_API_HOST` exists so a container platform
can route to the service, but what ships in the repository is loopback, and a
default is the thing most likely to be left alone.
"""

from __future__ import annotations

import hashlib
import json
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rse.api.limits import RateLimiter
from rse.api.models import ENERGY_LADDER, GenerateRequest, energy_from_percent
from rse.api.settings import DEFAULT_HOST, Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # Rate limiting off by default so a test does not fail for making a second
    # request; the limiter has its own tests.
    return Settings(
        storage_dir=tmp_path / "songs",
        rate_per_minute=100000,
        rate_burst=1000,
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    from rse.api.app import create_app

    return TestClient(create_app(settings))


def generate(client: TestClient, **body) -> dict:
    response = client.post("/api/generate", json=body)
    assert response.status_code == 200, response.text
    return response.json()


# --- the connection -------------------------------------------------------


def test_health_reports_no_model_requirement(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["engine"] == "procedural"
    assert body["model_required"] is False
    assert set(body["consoles_available"]) == {"nes", "snes"}


def test_capabilities_are_read_from_the_engine(client: TestClient) -> None:
    """The frontend must not carry its own copy of the style list."""
    from rse.procedural.compose import STYLES

    body = client.get("/api/capabilities").json()
    assert {style["name"] for style in body["styles"]} == set(STYLES)
    for style in body["styles"]:
        assert style["tempo_bpm"] == STYLES[style["name"]].tempo_bpm
        assert style["mode"] == STYLES[style["name"]].mode


def test_an_english_prompt_produces_a_playable_song(client: TestClient) -> None:
    body = generate(client, style="boss", seed=20261026)
    assert body["engine"] == "procedural"
    assert body["style"] == "boss"
    assert body["hardware_violations"] == 0
    assert body["loop_timing_continuous"] is True
    assert body["bars"] >= 24

    wav = client.get(body["links"]["wav"])
    assert wav.status_code == 200
    assert wav.headers["content-type"] == "audio/wav"
    assert len(wav.content) > 100_000


def test_the_api_and_the_cli_share_one_generation_path() -> None:
    """Guard against the duplication that has already cost this project once.

    A silence fix once went into `rse/song/build.py` while the script kept its
    own copy of the function, so the shipped path stayed broken and the tests
    passed. Two front ends make that failure likely, so both must call the
    library.
    """
    api = Path("rse/api/app.py").read_text()
    cli = Path("scripts/prompt_to_song.py").read_text()
    assert "from rse.song.generate import" in api
    assert "from rse.song.generate import" in cli
    # Neither may reimplement composition.
    for source in (api, cli):
        assert "from rse.procedural.compose import compose" not in source


def test_the_served_song_matches_the_served_report(client: TestClient) -> None:
    body = generate(client, style="boss", seed=5)
    report = client.get(body["links"]["report"]).json()
    assert report["seed"] == body["seed"]
    assert report["duration_sec"] == body["duration_sec"]
    assert report["hardware_violations"] == body["hardware_violations"]
    assert report["composition"]["form"] == body["form"]


# --- what must refuse -----------------------------------------------------


def test_snes_is_never_nes_audio_under_a_snes_label(client: TestClient) -> None:
    """The constraint this test has always protected. It used to be satisfied
    by a 501; now SNES really generates, so it is satisfied by the output
    actually being different -- stereo, eight voices, sampled instruments --
    and by the preview label travelling with it."""
    snes = generate(client, console="snes", style="boss", seconds=30, seed=11)
    nes = generate(client, console="nes", style="boss", seconds=30, seed=11)
    assert snes["snes"] is not None
    assert nes["snes"] is None
    # Different renderers, different audio.
    digest = lambda url: hashlib.sha256(  # noqa: E731 - local, single use
        client.get(url).content
    ).hexdigest()
    assert digest(snes["links"]["wav"]) != digest(nes["links"]["wav"])
    assert snes["snes"]["hardware_exact"] is False


def test_capabilities_marks_both_consoles_available(client: TestClient) -> None:
    consoles = {c["id"]: c for c in client.get("/api/capabilities").json()["consoles"]}
    assert consoles["nes"]["available"] is True
    assert consoles["snes"]["available"] is True
    # NES is not a preview; SNES is, and says so.
    assert not consoles["nes"].get("preview")
    assert consoles["snes"]["preview"] is True


def test_no_format_is_advertised_that_cannot_be_written(client: TestClient) -> None:
    """OGG is in the roadmap and not in the image; a dead download button is
    worse than a missing one."""
    body = client.get("/api/capabilities").json()
    assert "ogg" not in body["formats"]
    song = generate(client, style="boss", seed=1)
    assert song["links"].get("ogg") is None


def test_duration_cap_is_enforced(client: TestClient, settings: Settings) -> None:
    over = client.post(
        "/api/generate",
        json={"style": "boss", "seconds": settings.max_seconds + 1},
    )
    assert over.status_code == 422
    assert over.json()["error"] == "duration_out_of_range"
    under = client.post(
        "/api/generate",
        json={"style": "boss", "seconds": settings.min_seconds - 1},
    )
    assert under.status_code == 422


def test_unknown_style_is_rejected(client: TestClient) -> None:
    response = client.post("/api/generate", json={"prompt": "x", "style": "dubstep"})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "filename",
    ["../../../etc/passwd", "song.exe", "../song.wav", "song.wav/../../x"],
)
def test_file_fetch_cannot_leave_the_storage_directory(
    client: TestClient, filename: str
) -> None:
    body = generate(client, style="boss", seed=2)
    response = client.get(f"/api/songs/{body['song_id']}/{filename}")
    assert response.status_code == 404


def test_song_id_must_be_a_hex_uuid(client: TestClient) -> None:
    assert client.get("/api/songs/..%2f..%2fetc/song.wav").status_code == 404
    assert client.get("/api/songs/notauuid/song.wav").status_code == 404


def test_rate_limit_answers_429_with_a_retry_hint(tmp_path: Path) -> None:
    from rse.api.app import create_app

    client = TestClient(
        create_app(
            Settings(storage_dir=tmp_path, rate_per_minute=60, rate_burst=1)
        )
    )
    first = client.post("/api/generate", json={"style": "boss", "seconds": 10})
    assert first.status_code == 200
    second = client.post("/api/generate", json={"style": "boss", "seconds": 10})
    assert second.status_code == 429
    assert second.json()["error"] == "rate_limited"
    assert int(second.headers["Retry-After"]) >= 1


# --- the controls must do something ---------------------------------------


@pytest.mark.parametrize(
    ("percent", "bucket"),
    [(0, "very_low"), (20, "low"), (40, "medium"), (60, "medium_high"), (100, "very_high")],
)
def test_energy_slider_covers_every_bucket(percent: float, bucket: str) -> None:
    assert energy_from_percent(percent) == bucket


def test_every_energy_bucket_is_reachable_from_the_slider() -> None:
    """Equal bands, so no rung needs an impossible slider position."""
    reached = {energy_from_percent(p) for p in range(0, 101)}
    assert reached == set(ENERGY_LADDER)


def test_energy_actually_changes_the_style(client: TestClient) -> None:
    """The defect this guards: a slider the interface shows and the composer
    never sees. Energy is what selection falls back to, so with a prompt that
    names no scenario or mood it must move the result."""
    calm = generate(client, energy=0, seed=7)
    intense = generate(client, energy=100, seed=7)
    assert calm["style"] != intense["style"]
    assert "energy" in calm["style_reason"]
    assert "energy" in intense["style_reason"]


def test_energy_says_so_when_it_did_not_steer(client: TestClient) -> None:
    """A named scenario outranks energy; the note must admit that rather than
    implying the slider was applied."""
    body = generate(client, style="boss", energy=5, seed=7)
    assert body["style"] == "boss"
    assert any("did not steer" in note for note in body["notes"])


def test_tempo_control_reaches_the_composer(client: TestClient) -> None:
    """The defect this guards: `select()` discards a tempo it thinks was
    guessed, which silently turned a 90 BPM request into a 150 BPM song and
    reported "the prompt named no tempo"."""
    slow = generate(client, style="boss", tempo_bpm=90, seed=7)
    fast = generate(client, style="boss", tempo_bpm=180, seed=7)
    # Snapped to the NES frame grid: 90 -> 90.15 (10 frames), 180 -> 180.30 (5).
    assert slow["tempo_bpm"] == pytest.approx(90.15, abs=0.01)
    assert fast["tempo_bpm"] == pytest.approx(180.30, abs=0.01)
    assert (slow["frames_per_row"], fast["frames_per_row"]) == (10, 5)
    # The composer fits the requested duration and varies the bar count, so
    # tempo shows up in the bars rather than the seconds. Checking the seconds
    # would pass whether or not the tempo arrived.
    assert fast["bars"] > slow["bars"]
    # The note must name the frame count, which is what actually changed.
    assert any("frames per tracker row" in note for note in slow["notes"])
    assert any("overrides" in note for note in slow["notes"])


def test_an_explicit_tempo_is_not_called_absent(client: TestClient) -> None:
    """The report must not claim the prompt named no tempo when a control
    named one."""
    body = generate(client, style="boss", tempo_bpm=90, seed=7)
    assert not any("named no tempo" in note for note in body["notes"])
    stated = {
        item["text"]
        for bucket in body["requirements"].values()
        for item in bucket
    }
    assert "tempo 90 BPM" in stated


def test_a_preset_tempo_request_is_not_reported_as_an_override(
    client: TestClient,
) -> None:
    """Asking for exactly what the preset already does is not an override."""
    body = generate(client, style="boss", tempo_bpm=150, seed=7)
    assert body["tempo_bpm"] == pytest.approx(150.25, abs=0.01)
    assert not any("overriding" in note for note in body["notes"])


def test_an_explicit_style_is_honoured_and_reported_as_chosen(
    client: TestClient,
) -> None:
    """Was `test_style_control_overrides_inference_and_says_so`, which paired
    style="boss" with the prompt "a peaceful village" to prove the control
    beat the parser's inference. With prompts gone there is no inference to
    beat, so what remains is that the control is honoured and the report says
    where the style came from rather than implying the engine chose it."""
    body = generate(client, style="boss", seed=7)
    assert body["style"] == "boss"
    assert body["style_reason"] == "chosen directly"


def test_ui_style_labels_are_accepted(client: TestClient) -> None:
    """The dropdown shows "Boss Battle"; the composer calls it "boss"."""
    assert GenerateRequest(style="Boss Battle").style == "boss"
    assert GenerateRequest(style="boss_battle").style == "boss"
    assert generate(client, style="Boss Battle", seed=7)["style"] == "boss"


def test_duration_control_reaches_the_composer(client: TestClient) -> None:
    short = generate(client, style="boss", seconds=30, seed=7)
    long = generate(client, style="boss", seconds=120, seed=7)
    assert short["duration_sec"] < long["duration_sec"]
    assert short["requested_seconds"] == 30.0
    assert long["requested_seconds"] == 120.0


def test_an_empty_prompt_still_generates(client: TestClient) -> None:
    """Someone may only touch the controls."""
    body = generate(client, style="town", seconds=30, seed=7)
    assert body["style"] == "town"
    assert body["duration_sec"] > 0


# --- seeds ----------------------------------------------------------------


def test_an_omitted_seed_is_returned_and_reproduces(client: TestClient) -> None:
    first = generate(client, style="boss")
    assert isinstance(first["seed"], int)
    second = generate(client, style="boss", seed=first["seed"])
    digest = lambda url: hashlib.sha256(  # noqa: E731 - local, single use
        client.get(url).content
    ).hexdigest()
    assert digest(first["links"]["wav"]) == digest(second["links"]["wav"])


def test_a_new_request_gives_a_new_variation(client: TestClient) -> None:
    first = generate(client, style="boss")
    second = generate(client, style="boss")
    assert first["seed"] != second["seed"]
    assert first["song_id"] != second["song_id"]


def test_the_cli_and_the_api_agree_on_the_same_seed(
    client: TestClient, tmp_path: Path
) -> None:
    """One engine, one answer, whichever door you came in by.

    The two sides must be given the SAME request, which stopped being
    automatic when the `prompt` field was removed. This used to pass PROMPT
    to the library while the API got a prompt built from `style`, so the
    comparison quietly became "do two different requests happen to agree" --
    and it would have kept passing for as long as style selection converged,
    which is not what the test claims to check.

    Taking the string from `effective_prompt()` ties it to the real
    settings-to-plan translation, so the test breaks if that translation
    changes rather than silently comparing the wrong things.
    """
    from rse.song.generate import generate_song, write_artifacts

    body = generate(client, style="boss", seed=4242, seconds=60)
    result = generate_song(
        GenerateRequest(style="boss").effective_prompt(),
        seed=4242,
        target_seconds=60.0,
    )
    payload = write_artifacts(result, out_dir=tmp_path, name="cli")
    assert payload["duration_sec"] == body["duration_sec"]
    assert payload["composition"]["form"] == body["form"]
    served = client.get(body["links"]["wav"]).content
    assert hashlib.sha256(Path(payload["files"]["wav"]).read_bytes()).hexdigest() == (
        hashlib.sha256(served).hexdigest()
    )


# --- honesty --------------------------------------------------------------


def test_requirement_buckets_are_all_present(client: TestClient) -> None:
    body = generate(client, style="boss", seed=7)
    for bucket in ("satisfied", "approximated", "imposed", "unsupported"):
        assert bucket in body["requirements"]


# --- storage and caps -----------------------------------------------------


def test_songs_are_evicted_past_the_count_limit(tmp_path: Path) -> None:
    from rse.api.app import create_app

    client = TestClient(
        create_app(
            Settings(
                storage_dir=tmp_path,
                max_stored_songs=2,
                rate_per_minute=100000,
                rate_burst=1000,
            )
        )
    )
    ids = [
        generate(client, style="boss", seconds=10)["song_id"] for _ in range(3)
    ]
    # The oldest is gone; the newest two remain fetchable.
    assert client.get(f"/api/songs/{ids[0]}/song.wav").status_code == 404
    assert client.get(f"/api/songs/{ids[-1]}/song.wav").status_code == 200


def test_expired_songs_are_swept(tmp_path: Path) -> None:
    from rse.api.app import create_app

    client = TestClient(
        create_app(
            Settings(
                storage_dir=tmp_path,
                retention_seconds=0.0,
                rate_per_minute=100000,
                rate_burst=1000,
            )
        )
    )
    first = generate(client, style="boss", seconds=10)["song_id"]
    time.sleep(0.01)
    generate(client, style="boss", seconds=10)
    assert client.get(f"/api/songs/{first}/song.wav").status_code == 404


def test_rate_limiter_refills() -> None:
    limiter = RateLimiter(per_minute=60, burst=1)
    assert limiter.check("a", now=0.0) is None
    wait = limiter.check("a", now=0.0)
    assert wait is not None and wait == pytest.approx(1.0, abs=0.01)
    assert limiter.check("a", now=1.01) is None


def test_rate_limiter_is_per_caller() -> None:
    limiter = RateLimiter(per_minute=60, burst=1)
    assert limiter.check("a", now=0.0) is None
    assert limiter.check("b", now=0.0) is None
    assert limiter.check("a", now=0.0) is not None


def test_rate_limiter_table_does_not_grow_without_bound() -> None:
    limiter = RateLimiter(per_minute=60, burst=1)
    for index in range(100):
        limiter.check(f"caller-{index}", now=0.0)
    assert limiter.forget_before(10.0) == 100


# --- deployment defaults --------------------------------------------------


def test_the_bind_default_is_loopback() -> None:
    """What ships in the repository must not be reachable off the host."""
    assert DEFAULT_HOST == "127.0.0.1"
    assert Settings().host == "127.0.0.1"


def test_cors_is_not_wide_open_by_default() -> None:
    assert Settings().allowed_origins == ()


def test_host_override_is_explicit(monkeypatch) -> None:
    monkeypatch.setenv("RSE_API_HOST", "0.0.0.0")  # noqa: S104 - the test is that
    assert Settings.from_env().host == "0.0.0.0"  # noqa: S104 - it takes an override
    monkeypatch.delenv("RSE_API_HOST")
    assert Settings.from_env().host == "127.0.0.1"


def test_platform_port_is_honoured(monkeypatch) -> None:
    monkeypatch.setenv("PORT", "10000")
    assert Settings.from_env().port == 10000


def test_no_wildcard_bind_is_committed() -> None:
    """The sweep, as a test: nothing in the API package binds everything."""
    for path in Path("rse/api").glob("*.py"):
        source = path.read_text()
        for pattern in ('host="0.0.0.0"', "host='0.0.0.0'", 'bind="0.0.0.0"'):
            assert pattern not in source, f"{path}: {pattern}"


# --- the audio is real ----------------------------------------------------


def test_the_served_wav_is_a_valid_riff_file(client: TestClient, tmp_path: Path) -> None:
    body = generate(client, style="boss", seed=11)
    data = client.get(body["links"]["wav"]).content
    path = tmp_path / "served.wav"
    path.write_bytes(data)
    with wave.open(str(path)) as handle:
        assert handle.getframerate() == 44100
        assert handle.getnchannels() == 1
        seconds = handle.getnframes() / handle.getframerate()
    assert seconds == pytest.approx(body["duration_sec"], abs=0.05)


def test_the_served_midi_is_loadable(client: TestClient, tmp_path: Path) -> None:
    import mido

    body = generate(client, style="boss", seed=12)
    path = tmp_path / "served.mid"
    path.write_bytes(client.get(body["links"]["midi"]).content)
    midi = mido.MidiFile(path)
    onsets = sum(
        1
        for track in midi.tracks
        for message in track
        if message.type == "note_on" and message.velocity > 0
    )
    assert onsets > 100


def test_the_report_is_valid_json(client: TestClient) -> None:
    body = generate(client, style="boss", seed=13)
    raw = client.get(body["links"]["report"]).content
    assert json.loads(raw)["engine"] == "procedural"


def test_running_the_service_refuses_a_remote_bind(monkeypatch, capsys) -> None:
    """A warning is not a gate. The repo already refuses this in the Gradio
    UI; the API entry point must behave the same way."""
    from rse.api.app import main

    monkeypatch.setenv("RSE_API_HOST", "0.0.0.0")  # noqa: S104 - under test
    monkeypatch.delenv("RSE_API_ALLOW_REMOTE", raising=False)
    called: list[object] = []
    monkeypatch.setattr(
        "uvicorn.run", lambda *a, **k: called.append((a, k))
    )
    assert main() == 2
    assert called == [], "must not start a listener it refused"
    assert "refusing to bind" in capsys.readouterr().out


def test_an_explicit_opt_in_allows_a_remote_bind(monkeypatch) -> None:
    from rse.api.app import main

    monkeypatch.setenv("RSE_API_HOST", "0.0.0.0")  # noqa: S104 - under test
    monkeypatch.setenv("RSE_API_ALLOW_REMOTE", "1")
    called: list[object] = []
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: called.append(k))
    assert main() == 0
    assert called and called[0]["host"] == "0.0.0.0"  # noqa: S104 - under test


def test_loopback_needs_no_opt_in(monkeypatch) -> None:
    from rse.api.app import main

    monkeypatch.delenv("RSE_API_HOST", raising=False)
    monkeypatch.delenv("RSE_API_ALLOW_REMOTE", raising=False)
    called: list[object] = []
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: called.append(k))
    assert main() == 0
    assert called[0]["host"] == "127.0.0.1"


# --- the report must describe THIS engine ---------------------------------


# --- only connected controls are offered ----------------------------------


def test_mood_actually_changes_the_sound(client: TestClient) -> None:
    """Mood was an inert control: the interpreter read it into `key_mode`,
    which the composer never consults, so "a dark theme" and "a bright theme"
    were byte-identical once a style was chosen. It now selects the instrument
    profile, which is the dimension proven audible."""
    duties = {}
    for mood in ("dark", "mysterious", "bright", "peaceful"):
        body = generate(client, style="boss", mood=mood, seconds=30, seed=7)
        duties[mood] = tuple(sorted(body["sound"]["duty_by_channel"].items()))
        assert body["sound"]["reason"] == "mood"
    assert len(set(duties.values())) >= 3, duties


def test_an_unsupported_mood_is_refused(client: TestClient) -> None:
    """The rule for anything an interpreter may emit: if the engine does not
    support it, do not pretend it does."""
    response = client.post(
        "/api/generate", json={"style": "boss", "mood": "spicy", "seconds": 30}
    )
    assert response.status_code == 422


def test_an_unsupported_sound_profile_is_refused(client: TestClient) -> None:
    response = client.post(
        "/api/generate",
        json={"style": "boss", "sound_profile": "orchestral", "seconds": 30},
    )
    assert response.status_code == 422


def test_an_explicit_profile_outranks_mood(client: TestClient) -> None:
    body = generate(
        client, style="boss", mood="dark", sound_profile="eerie", seconds=30, seed=7
    )
    assert body["sound"]["profile"] == "eerie"
    assert body["sound"]["reason"] == "profile"


def test_capabilities_only_lists_profiles_the_engine_has(client: TestClient) -> None:
    from rse.procedural.sound import MOOD_PROFILES, SOUND_PROFILES

    caps = client.get("/api/capabilities").json()
    assert {p["name"] for p in caps["sound_profiles"]} == set(SOUND_PROFILES)
    assert set(caps["moods"]) == set(MOOD_PROFILES)
    # Every advertised mood maps to an advertised profile.
    advertised = {p["name"] for p in caps["sound_profiles"]}
    for profile in caps["sound_profiles"]:
        for mood in profile["moods"]:
            assert MOOD_PROFILES[mood] in advertised


def test_the_response_says_what_played_it(client: TestClient) -> None:
    body = generate(client, style="boss", seconds=30, seed=7)
    sound = body["sound"]
    for field in ("lead_timbre", "comp_timbre", "duty_by_channel",
                  "lead_duty_percent", "comp_duty_percent", "reason"):
        assert field in sound, field
    assert set(sound["duty_by_channel"]) == {"P1", "P2"}


def test_sound_design_is_on_by_default_through_the_api(client: TestClient) -> None:
    """Promoted after the listening test. A request that names nothing still
    gets shaped velocities and a chosen timbre pair."""
    body = generate(client, style="boss", seconds=30, seed=7)
    assert body["sound"]["lead_timbre"]
    assert body["sound"]["duty_by_channel"]
    # And the report on disk records it, so a downloaded song says how it was
    # performed and not only what was played.
    report = client.get(body["links"]["report"]).json()
    assert report["composition"]["sound"]["lead_timbre"] == (
        body["sound"]["lead_timbre"]
    )
    assert report["composition"]["duty_by_channel"] == (
        body["sound"]["duty_by_channel"]
    )


# --- SNES through the website -------------------------------------------


def test_snes_is_available_and_labelled_a_preview(client: TestClient) -> None:
    """It used to return 501. Now it generates, and the label has to travel
    with it: hardware accuracy is still not claimed."""
    consoles = {c["id"]: c for c in client.get("/api/capabilities").json()["consoles"]}
    assert consoles["snes"]["available"] is True
    assert consoles["snes"]["preview"] is True
    assert "NOT hardware-exact" in consoles["snes"]["detail"]


def test_snes_generates_genuine_stereo(client: TestClient) -> None:
    import wave as wave_module

    body = generate(client, console="snes", style="boss", seconds=60, seed=4242)
    assert body["snes"] is not None
    data = client.get(body["links"]["wav"]).content
    path = Path("/tmp") / f"snes_{body['song_id']}.wav"
    path.write_bytes(data)
    try:
        with wave_module.open(str(path)) as handle:
            assert handle.getnchannels() == 2
            frames = handle.getnframes()
            raw = np.frombuffer(
                handle.readframes(frames), dtype="<i2"
            ).astype(np.float64).reshape(-1, 2)
        assert not np.array_equal(raw[:, 0], raw[:, 1]), "not real stereo"
    finally:
        path.unlink(missing_ok=True)


def test_snes_respects_the_eight_voice_ceiling(client: TestClient) -> None:
    body = generate(client, console="snes", style="boss", seconds=60, seed=4242)
    snes = body["snes"]
    assert snes["peak_voices"] <= 8
    assert snes["allocation"]["dropped"] == 0
    assert snes["allocation"]["stolen"] == 0


def test_snes_reports_a_measured_aram_budget(client: TestClient) -> None:
    body = generate(client, console="snes", style="boss", seconds=60, seed=1)
    aram = body["snes"]["aram"]
    assert aram["aram_bytes"] == 65536
    assert aram["fits"] is True
    assert aram["breakdown"]["samples"] > 0
    # And it says which figures are assumed rather than measured.
    assert "driver_bytes" in aram["assumed"]
    assert body["snes"]["instrument_bank"]["brr_measured"] is True
    # "Measured" applies to part of the total, not all of it, and the
    # response has to let a caller tell which part.
    assert 0 < aram["measured_bytes"] < aram["total_bytes"]
    assert aram["evidence"]["samples"] == "measured"
    assert aram["evidence"]["driver"] == "assumed"
    # Fitting in 64 KiB is not a claim that a SNES would load it.
    assert body["snes"]["hardware_exact"] is False
    assert body["snes"]["hardware_budgeted"] is False
    assert aram["unmodelled"], "the unmodelled allocations must be named"


def test_echo_costs_audio_ram(client: TestClient) -> None:
    """The tradeoff the hardware imposes, visible through the API: a longer
    reverb is memory the instruments do not get."""
    budgets = {}
    for preset in ("dry", "room", "cavern"):
        body = generate(
            client, console="snes", style="boss", seconds=60, seed=5, echo=preset
        )
        budgets[preset] = body["snes"]["aram"]["total_bytes"]
    assert budgets["dry"] < budgets["room"] < budgets["cavern"]
    assert budgets["cavern"] - budgets["dry"] > 25_000


def test_the_echo_control_is_declared_inaudible_because_it_is(
    client: TestClient,
) -> None:
    """Release validation generated one song at every echo preset and got
    four byte-identical WAVs.

    The setting reserves the S-DSP's echo buffer in the memory account, and
    `VoiceSettings.echo` records a per-voice send, but no code reads either
    when rendering -- there is no echo processor. Advertising it as a reverb
    would be the "the metadata exists, so the capability exists" mistake.

    So this test asserts the control does NOT change the audio, which looks
    backwards and is the point: when an echo processor is written, this test
    fails, and whoever writes it has to come here and update the label in the
    request model, the capabilities payload, the page and the report
    caveats. A test that merely allowed silence would let the labels rot.
    """
    presets = client.get("/api/capabilities").json()["echo_presets"]
    assert presets
    for preset in presets:
        assert preset["affects_audio"] is False, preset["name"]

    audio = {}
    for name in ("dry", "room", "hall", "cavern"):
        body = generate(
            client, console="snes", style="boss", seconds=20, seed=9, echo=name
        )
        assert body["snes"]["echo"]["affects_audio"] is False
        wav = client.get(body["links"]["wav"])
        assert wav.status_code == 200
        audio[name] = hashlib.sha256(wav.content).hexdigest()

    assert len(set(audio.values())) == 1, (
        "an echo preset changed the audio -- if that is intentional, the "
        f"'budget only' labels are now wrong: {audio}"
    )


def test_capabilities_prices_every_echo_preset(client: TestClient) -> None:
    presets = client.get("/api/capabilities").json()["echo_presets"]
    assert presets
    for preset in presets:
        assert preset["aram_bytes"] >= 0
        if preset["delay_ms"] == 0:
            assert preset["aram_bytes"] == 0
        else:
            assert preset["aram_bytes"] > 0


def test_an_unknown_echo_preset_is_refused(client: TestClient) -> None:
    response = client.post(
        "/api/generate",
        json={"console": "snes", "style": "boss", "seconds": 30, "echo": "cathedral"},
    )
    assert response.status_code == 422


def test_reserving_voices_reduces_what_music_gets(client: TestClient) -> None:
    """Game-ready mode, and the loss is reported rather than silent."""
    full = generate(client, console="snes", style="boss", seconds=60, seed=9)
    held = generate(
        client, console="snes", style="boss", seconds=60, seed=9, reserved_voices=4
    )
    assert held["snes"]["allocation"]["voice_count"] == 4
    assert full["snes"]["allocation"]["voice_count"] == 8
    assert held["snes"]["allocation"]["dropped"] > 0
    assert held["snes"]["allocation"]["displacements"]


def test_reserving_too_many_voices_is_refused(client: TestClient) -> None:
    response = client.post(
        "/api/generate",
        json={"console": "snes", "seconds": 30, "reserved_voices": 9},
    )
    assert response.status_code == 422


def test_snes_is_reproducible_from_its_seed(client: TestClient) -> None:
    first = generate(client, console="snes", style="town", seconds=30, seed=77)
    second = generate(client, console="snes", style="town", seconds=30, seed=77)
    digest = lambda url: hashlib.sha256(  # noqa: E731 - local, single use
        client.get(url).content
    ).hexdigest()
    assert digest(first["links"]["wav"]) == digest(second["links"]["wav"])


def test_nes_is_untouched_by_the_snes_branch(client: TestClient) -> None:
    """NES stays mono, carries no `snes` field, and keeps its duty report."""
    import wave as wave_module

    body = generate(client, console="nes", style="boss", seconds=30, seed=7)
    assert body["snes"] is None
    assert body["sound"]["duty_by_channel"]
    data = client.get(body["links"]["wav"]).content
    path = Path("/tmp") / f"nes_{body['song_id']}.wav"
    path.write_bytes(data)
    try:
        with wave_module.open(str(path)) as handle:
            assert handle.getnchannels() == 1
    finally:
        path.unlink(missing_ok=True)


def test_the_cli_and_the_api_share_the_snes_path() -> None:
    """The duplication guard, for the SNES path too."""
    api = Path("rse/api/app.py").read_text()
    cli = Path("scripts/snes_boss.py").read_text()
    assert "from rse.song.snes_generate import" in api
    assert "from rse.song.snes_generate import" in cli
    # Neither reimplements the arrangement or the render.
    for source in (api, cli):
        assert "from rse.snes.arrange import arrange" not in source


def test_the_snes_preview_caveats_reach_the_response(client: TestClient) -> None:
    body = generate(client, console="snes", style="boss", seconds=30, seed=3)
    caveats = " ".join(body["snes"]["caveats"])
    assert "BRR-encoded" in caveats
    assert "NOT the S-DSP" in caveats
    assert "no emulator verification" in caveats.lower() or (
        "no SPC export" in caveats
    )
    assert body["snes"]["hardware_exact"] is False


# --- per-deployment console gating ----------------------------------------
#
# Render cost differs by 32x between the consoles: NES measures 0.022x
# realtime and SNES 0.70x. Generation is synchronous inside the request, so
# which consoles a host can serve is a property of that host's CPU, not of
# the code -- hence a deployment setting rather than a build flag. SNES stays
# fully working everywhere else, including in every other test in this file.


def _gated(tmp_path: Path, consoles: tuple[str, ...]) -> TestClient:
    from rse.api.app import create_app

    return TestClient(
        create_app(
            Settings(
                storage_dir=tmp_path / "songs",
                rate_per_minute=100000,
                rate_burst=1000,
                consoles=consoles,
            )
        )
    )


def test_an_empty_console_list_serves_everything(tmp_path: Path) -> None:
    """The default has to stay permissive: local development and the test
    suite must not need a variable set to reach the full engine."""
    assert Settings().consoles == ()
    assert Settings().serves("nes") and Settings().serves("snes")
    client = _gated(tmp_path, ())
    available = {
        c["id"] for c in client.get("/api/capabilities").json()["consoles"]
        if c["available"]
    }
    assert available == {"nes", "snes"}


def test_a_disabled_console_is_refused_before_any_work(tmp_path: Path) -> None:
    """503, not 422: the request is well-formed and would succeed on a
    differently configured host, so the shortfall is the deployment's rather
    than the caller's. And it must be refused rather than quietly served by
    the other engine -- the mistake this project already made once was
    letting a console label travel with audio from the wrong renderer.
    """
    client = _gated(tmp_path, ("nes",))
    response = client.post(
        "/api/generate", json={"console": "snes", "style": "boss", "seconds": 20}
    )
    assert response.status_code == 503
    # Flattened, not nested under "detail": the app's exception handler lifts
    # it so the page can read payload.message like every other error.
    body = response.json()
    assert body["error"] == "console_not_enabled"
    assert body["console"] == "snes"
    assert body["consoles_available"] == ["nes"]
    assert "not enabled" in body["message"]


def test_the_enabled_console_still_works_when_another_is_gated(
    tmp_path: Path,
) -> None:
    client = _gated(tmp_path, ("nes",))
    body = generate(client, console="nes", style="boss", seconds=20, seed=7)
    assert body["console"] == "nes"
    assert body["snes"] is None
    assert client.get(body["links"]["wav"]).status_code == 200


def test_a_withheld_console_is_disclosed_not_hidden(tmp_path: Path) -> None:
    """Hiding it would leave a user wondering where the 16-bit option went.
    It stays listed, marked unavailable, with the reason attached, which is
    what lets the page grey it out and explain itself."""
    client = _gated(tmp_path, ("nes",))
    consoles = {c["id"]: c for c in client.get("/api/capabilities").json()["consoles"]}
    assert set(consoles) == {"nes", "snes"}, "the console must not disappear"
    assert consoles["nes"]["available"] is True
    assert consoles["snes"]["available"] is False
    assert "not enabled on this deployment" in consoles["snes"]["detail"].lower()
    # And health agrees, so a monitor and the page cannot disagree.
    assert client.get("/api/health").json()["consoles_available"] == ["nes"]


def test_gating_is_case_and_whitespace_insensitive(monkeypatch) -> None:
    """It is typed into a dashboard field by a human."""
    monkeypatch.setenv("RSE_API_CONSOLES", " NES , SNES ")
    assert Settings.from_env().consoles == ("nes", "snes")
    monkeypatch.setenv("RSE_API_CONSOLES", "nes")
    assert Settings.from_env().consoles == ("nes",)
    # Unset must mean "everything", not "nothing".
    monkeypatch.delenv("RSE_API_CONSOLES")
    assert Settings.from_env().consoles == ()
    assert Settings.from_env().serves("snes")
