"""The generation API.

Deliberately unauthenticated: the audience is anonymous members of the public
generating retro music, so an auth gate would lock out exactly the people the
site exists for. What it does carry is the control a public endpoint actually
needs -- a duration cap, a concurrency cap, a per-caller rate limit, a
retention window, and no path the caller can steer into the filesystem.

The service binds to loopback unless the deployment says otherwise. A platform
that routes to a container sets `RSE_API_HOST=0.0.0.0` in its own
configuration, where the choice is visible, rather than inheriting it from a
default committed to the repository.

Generation is CPU-bound and takes about 1.2 s for a 60-second song, 97% of it
in the audio renderer. That is short enough to answer in the request, so there
is no job queue and no polling: one POST returns a finished song. If the
duration cap ever rises far enough to break that, the cap is the thing to
revisit first.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from rse.api.limits import RateLimiter, SongStore
from rse.api.models import (
    CapabilitiesResponse,
    EchoPreset,
    GenerateRequest,
    GenerateResponse,
    SongLinks,
    SoundProfileInfo,
    StyleInfo,
    TempoStep,
    energy_from_percent,
)
from rse.api.settings import Settings
from rse.prompt.interpret import Requirement

log = logging.getLogger("rse.api")

#: Files a caller may fetch from a song directory. An allowlist rather than a
#: sanitised path: the set is small and known, so there is nothing to sanitise.
SERVED_FILES = {
    "song.wav": ("audio/wav", "wav"),
    "song.mid": ("audio/midi", "mid"),
    "song.json": ("application/json", "json"),
}

#: The single-file generator page. Bundled, so `rse-api` serves a working
#: interface with no build step and no second process.
PAGE = Path(__file__).resolve().parents[2] / "frontend" / "index.html"

#: Human labels for the instrument profiles.
PROFILE_LABELS = {
    "aggressive": "Aggressive — thin cutting lead",
    "full": "Full — rounded lead",
    "eerie": "Eerie — thin lead, hollow bed",
    "warm": "Warm — rounded lead, glassy comp",
    "sharp": "Sharp — bright without harsh",
    "mellow": "Mellow — soft lead, hollow bed",
}

#: Human labels for the composer's presets, for the interface's dropdown.
STYLE_LABELS = {
    "boss": "Boss Battle",
    "action": "Battle / Action",
    "adventure": "Overworld / Adventure",
    "town": "Town / Village",
    "tension": "Dungeon / Tension",
    "victory": "Victory / Title",
}

CONSOLES = [
    {
        "id": "nes",
        "label": "8-bit NES",
        "available": True,
        "detail": "Two pulse channels, triangle bass, noise percussion, "
        "rendered through the hardware-constrained NES APU synthesizer.",
    },
    {
        "id": "snes",
        "label": "16-bit SNES",
        "available": True,
        "preview": True,
        "detail": "SNES-INSPIRED PREVIEW. Sampled instruments across eight "
        "DSP voices, stereo with per-voice panning, hardware-informed "
        "envelopes, real BRR encoding and a measured 64 KiB audio-RAM "
        "budget. NOT hardware-exact: interpolation is a 4-point kernel plus "
        "a documented low-pass rather than the S-DSP's own table, envelopes "
        "are floating point, and there is no SPC export or emulator "
        "verification.",
    },
]


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings.from_env()
    storage = Path(
        config.storage_dir
        or Path(__file__).resolve().parents[2] / "var" / "songs"
    )
    storage.mkdir(parents=True, exist_ok=True)

    limiter = RateLimiter(
        per_minute=config.rate_per_minute, burst=config.rate_burst
    )
    store = SongStore(
        root=storage,
        retention_seconds=config.retention_seconds,
        max_songs=config.max_stored_songs,
    )
    # Bounds the instance: renders are CPU-bound, so admitting more than a
    # couple at once only makes every caller wait longer.
    gate = asyncio.Semaphore(max(1, config.max_concurrent))

    app = FastAPI(
        title="Retro Synthesis Engine",
        version="0.1.0",
        description="Generate complete NES-style soundtracks from a prompt.",
    )
    app.state.settings = config
    app.state.storage = storage
    app.state.limiter = limiter
    app.state.store = store

    if config.allowed_origins:
        # Named origins only. `*` is never the committed default: the public
        # site's origin belongs in the deploy configuration.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.allowed_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["content-type"],
        )

    def client_key(request: Request) -> str:
        """Identify a caller for rate limiting.

        Behind a platform proxy the peer address is the proxy, so the
        forwarded header is used when present. It is caller-supplied and
        therefore spoofable -- acceptable for a cost cap on anonymous public
        traffic, and not relied on for anything else.
        """
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def sweep(now: float) -> None:
        """Delete songs past the retention window. Cheap, so run it inline."""
        for song_id in store.expired(now=now):
            discard(song_id)
        limiter.forget_before(time.monotonic() - 3600.0)

    def discard(song_id: str) -> None:
        target = storage / song_id
        if target.is_dir() and target.parent == storage:
            shutil.rmtree(target, ignore_errors=True)
        store.forget(song_id)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "engine": "procedural",
            "model_required": False,
            "consoles_available": [c["id"] for c in CONSOLES if c["available"]],
        }

    @app.get("/api/capabilities", response_model=CapabilitiesResponse)
    def capabilities() -> CapabilitiesResponse:
        """Read from the engine, so the frontend cannot drift from it."""
        from rse.hardware.tempo import ladder as tempo_ladder
        from rse.procedural.compose import STYLES
        from rse.procedural.sound import (
            MOOD_PROFILES,
            SOUND_PROFILES,
            TIMBRES,
            _percent,
        )
        from rse.snes.aram import echo_bytes_for
        from rse.song.snes_generate import ECHO_PRESETS

        return CapabilitiesResponse(
            styles=[
                StyleInfo(
                    name=name,
                    label=STYLE_LABELS.get(name, name.title()),
                    tempo_bpm=style.tempo_bpm,
                    frames_per_row=style.frames_per_row,
                    mode=style.mode,
                )
                for name, style in sorted(STYLES.items())
            ],
            consoles=CONSOLES,
            min_seconds=config.min_seconds,
            max_seconds=config.max_seconds,
            default_seconds=60.0,
            # OGG is absent on purpose: there is no Vorbis encoder in the
            # image, and advertising a format the server cannot write would
            # put a dead download button on the page.
            formats=["wav", "mid"],
            tempo_ladder=[TempoStep(**step.to_dict()) for step in tempo_ladder()],
            sound_profiles=[
                SoundProfileInfo(
                    name=name,
                    label=PROFILE_LABELS.get(name, name.title()),
                    lead_timbre=lead,
                    comp_timbre=comp,
                    lead_duty_percent=_percent(TIMBRES[lead]),
                    comp_duty_percent=_percent(TIMBRES[comp]),
                    moods=sorted(
                        m for m, p in MOOD_PROFILES.items() if p == name
                    ),
                )
                for name, (lead, comp) in SOUND_PROFILES.items()
            ],
            moods=sorted(MOOD_PROFILES),
            echo_presets=[
                EchoPreset(
                    name=name,
                    delay_ms=delay,
                    aram_bytes=echo_bytes_for(delay),
                )
                for name, delay in sorted(
                    ECHO_PRESETS.items(), key=lambda kv: kv[1]
                )
            ],
        )

    @app.post("/api/generate", response_model=GenerateResponse)
    async def generate(body: GenerateRequest, request: Request) -> GenerateResponse:
        wait = limiter.check(client_key(request))
        if wait is not None:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limited",
                    "message": "Too many generations from this address.",
                    "retry_after_seconds": round(wait, 1),
                },
                headers={"Retry-After": str(max(1, int(wait) + 1))},
            )

        requested = body.seconds if body.seconds is not None else 60.0
        if requested > config.max_seconds or requested < config.min_seconds:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "duration_out_of_range",
                    "message": f"Duration must be between "
                    f"{config.min_seconds:g} and {config.max_seconds:g} "
                    "seconds.",
                    "min_seconds": config.min_seconds,
                    "max_seconds": config.max_seconds,
                },
            )

        seed = body.seed if body.seed is not None else secrets.randbelow(2**31 - 1)
        song_id = uuid.uuid4().hex
        out_dir = storage / song_id

        worker = _generate_snes_sync if body.console == "snes" else _generate_sync
        try:
            async with gate:
                payload = await asyncio.to_thread(
                    worker,
                    body=body,
                    seed=seed,
                    requested=requested,
                    out_dir=out_dir,
                )
        except ValueError as error:
            discard(song_id)
            raise HTTPException(
                status_code=422,
                detail={"error": "invalid_request", "message": str(error)},
            ) from error
        except Exception as error:  # noqa: BLE001 - reported, not leaked
            discard(song_id)
            log.exception("generation failed")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "generation_failed",
                    "message": f"{type(error).__name__}",
                },
            ) from error

        for evicted in store.record(song_id):
            discard(evicted)
        sweep(time.time())

        meta = payload["composition"]
        base = f"/api/songs/{song_id}"
        return GenerateResponse(
            song_id=song_id,
            seed=seed,
            console=body.console,
            style=meta["selection"]["style"],
            style_reason=meta["selection"]["reason"],
            duration_sec=payload["duration_sec"],
            requested_seconds=payload["requested_seconds"],
            duration_within_tolerance=payload["duration_within_tolerance"],
            tempo_bpm=meta["tempo_bpm"],
            frames_per_row=meta["frames_per_row"],
            mode=meta["mode"],
            sound=meta.get("sound") or {},
            bars=meta["total_bars"],
            form=meta["form"],
            loopable=payload["loop_diagnostic"]["timing_continuous"],
            hardware_violations=payload["hardware_violations"],
            loop_timing_continuous=payload["loop_diagnostic"]["timing_continuous"],
            links=SongLinks(
                wav=f"{base}/song.wav",
                midi=f"{base}/song.mid" if payload["files"]["midi"] else None,
                report=f"{base}/song.json",
            ),
            requirements=payload["requirement_report"],
            notes=list(meta["selection"]["notes"]),
            unsupported=list(meta["unsupported"]),
            snes=payload.get("snes"),
        )

    @app.get("/")
    def page() -> Response:
        """The generator page, served from the API so it is same-origin.

        One bundled file at a fixed path -- not a static directory mounted at
        the root. There is nothing for a caller to steer: no path parameter
        reaches the filesystem, so there is no traversal to defend against.
        Hosting the page on Cloudflare instead is equally supported; it reads
        `?api=` and then `RSE_API_ALLOWED_ORIGINS` has to name that origin.
        """
        if not PAGE.is_file():
            raise HTTPException(
                status_code=404,
                detail="frontend/index.html is not present in this install",
            )
        return FileResponse(PAGE, media_type="text/html")

    @app.get("/api/songs/{song_id}/{filename}")
    def fetch(song_id: str, filename: str) -> Response:
        """Serve a generated file.

        `song_id` must be a hex uuid and `filename` must be one of three known
        names, so neither can carry a path segment out of the storage root.
        """
        if len(song_id) != 32 or any(c not in "0123456789abcdef" for c in song_id):
            raise HTTPException(status_code=404, detail="no such song")
        served = SERVED_FILES.get(filename)
        if served is None:
            raise HTTPException(status_code=404, detail="no such file")
        media_type, _ = served
        path = storage / song_id / filename
        if not path.is_file():
            raise HTTPException(
                status_code=404,
                detail="no such song: it may have expired",
            )
        return FileResponse(
            path,
            media_type=media_type,
            filename=f"rse-{song_id[:8]}.{filename.rsplit('.', 1)[1]}",
        )

    @app.exception_handler(HTTPException)
    def as_json(_request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        body = detail if isinstance(detail, dict) else {"message": str(detail)}
        return JSONResponse(
            status_code=exc.status_code, content=body, headers=exc.headers
        )

    return app


def _generate_snes_sync(
    *, body: GenerateRequest, seed: int, requested: float, out_dir: Path
) -> dict[str, Any]:
    """The SNES path, run in a worker thread.

    Calls `rse.song.snes_generate`, the same library path
    `scripts/snes_boss.py` uses, so the website and the command line cannot
    produce different music from the same request.

    The response is shaped like the NES one -- same links, same requirement
    report -- with the eight-voice allocation and the ARAM budget under
    `snes`, because those are the facts a SNES listener needs and a NES one
    has no use for.
    """
    from rse.prompt.interpret import interpret
    from rse.song.generate import finish_report
    from rse.song.snes_generate import generate_snes, write_snes_artifacts

    plan = interpret(body.effective_prompt())
    # The SNES arranger reads the composer's plan, so the style selection is
    # the same machinery the NES path uses.
    from rse.procedural.select import select

    selection = select(plan)
    style = body.style or selection.style

    song = generate_snes(
        style=style,
        seed=seed,
        target_seconds=requested,
        echo=body.echo,
        reserved_voices=body.reserved_voices,
    )
    report = write_snes_artifacts(song, out_dir=out_dir, name="song")
    # Same keys as the NES payload, so one response model serves both. The
    # SNES writer calls it `score`, since a platform-neutral score is not
    # necessarily MIDI, but the API surface should not have two names for the
    # same link.
    report["files"] = {
        "wav": report["files"]["wav"],
        "midi": report["files"].get("score"),
    }

    # Shape it like the NES payload so one response model serves both.
    music = report["music"]
    plan.requirements.append(
        Requirement(
            "16-bit SNES",
            "approximated",
            "SNES-INSPIRED PREVIEW: sampled instruments on eight voices with "
            "real BRR encoding and a measured audio-RAM budget, but not "
            "hardware-exact and not emulator-verified",
        )
    )
    return {
        "version": "rse-snes-preview/0.1",
        "engine": "procedural",
        "prompt": body.prompt,
        "seed": seed,
        "songspec": plan.to_dict(),
        "composition": {
            "engine": "procedural",
            "selection": {
                "style": style,
                "reason": "chosen directly" if body.style else selection.reason,
                "matched_on": "explicit" if body.style else selection.matched_on,
                "notes": list(selection.notes),
                "tempo_bpm": selection.tempo_bpm,
                "tonic": selection.tonic,
                "section_bars": selection.section_bars,
                "unsupported": list(selection.unsupported),
            },
            "form": music["form"],
            "total_bars": music["bars"],
            "duration_sec": music["seconds"],
            "tempo_bpm": music["tempo_bpm"],
            "frames_per_row": song.result.style.frames_per_row,
            "mode": music["mode"],
            "tonic": music["tonic"],
            "sound": None,
            "duty_by_channel": None,
            "sections": [s.to_dict() for s in song.result.sections],
            "unsupported": list(selection.unsupported),
        },
        "duration_sec": music["seconds"],
        "requested_seconds": requested,
        "duration_tolerance_sec": 10.0,
        "duration_within_tolerance": abs(music["seconds"] - requested) <= 10.0,
        "hardware_violations": 0,
        "loop_diagnostic": {"timing_continuous": True, "notes": []},
        "requirement_report": finish_report(
            _SnesResultShim(song, plan)
        ),
        "files": report["files"],
        "snes": {
            "mode": report["mode"],
            "hardware_exact": False,
            "caveats": report["caveats"],
            "allocation": report["allocation"],
            "peak_voices": report["allocation"]["peak_voices"],
            "peak_concurrency_by_role": report["peak_concurrency_by_role"],
            "voice_plan": report["voice_plan"],
            "echo": report["echo"],
            "aram": report["aram"],
            "extended_chords": report["extended_chords"],
            "instrument_bank": {
                "bank_id": report["instrument_bank"]["bank_id"],
                "provenance": report["instrument_bank"]["provenance"],
                "instrument_count": report["instrument_bank"]["instrument_count"],
                "brr_bytes": report["instrument_bank"]["brr_bytes"],
                "brr_measured": report["instrument_bank"]["brr_measured"],
            },
            "audio": report["audio"],
        },
    }


class _SnesResultShim:
    """Enough of a `SongResult` for `finish_report` to describe a SNES song.

    The requirement report is platform-neutral -- it says what the request
    asked for and what the engine could do with it -- so it is worth reusing
    rather than writing a second one that drifts.
    """

    def __init__(self, song: Any, plan: Any) -> None:
        self.plan = plan
        self.meta = {
            "form": [s.label for s in song.result.sections],
            "selection": {"style": song.style, "matched_on": "", "reason": ""},
        }
        self._seconds = song.seconds

    @property
    def duration_sec(self) -> float:
        return self._seconds

    @property
    def within_tolerance(self) -> bool:
        return True

    @property
    def sections(self) -> list[Any]:
        return []

    @property
    def sound(self) -> Any:
        return None


def _generate_sync(
    *, body: GenerateRequest, seed: int, requested: float, out_dir: Path
) -> dict[str, Any]:
    """The blocking part, run in a worker thread.

    Calls the same library path the CLI uses, so the website and
    `scripts/prompt_to_song.py` cannot produce different music from the same
    request.
    """
    from rse.song.generate import generate_song, write_artifacts

    bucket = energy_from_percent(body.energy) if body.energy is not None else None
    result = generate_song(
        body.effective_prompt(),
        seed=seed,
        style=body.style,
        target_seconds=requested,
        tempo_bpm=body.tempo_bpm,
        loop=body.loop,
        energy=bucket,
        sound_profile=body.sound_profile,
        mood=body.mood,
    )
    if bucket is not None:
        # Say which of the two things it did. Energy is the fallback that picks
        # a style when the prompt named no scenario or mood, so a directly
        # chosen style leaves it with nothing to steer -- and the note has to
        # admit that rather than implying the slider was applied.
        notes = result.meta["selection"]["notes"]
        if result.meta["selection"]["matched_on"] in ("explicit", "scenario", "mood"):
            notes.append(
                f"energy {body.energy:g}% read as {bucket!r}, which did not "
                f"steer the style: {result.meta['selection']['reason']} "
                "decided it"
            )
        else:
            notes.append(f"energy {body.energy:g}% read as {bucket!r}")
    return write_artifacts(result, out_dir=out_dir, name="song", seed=seed)


#: The ASGI application. A platform entry point (`uvicorn rse.api.app:app`)
#: finds this; nothing here binds a socket by itself.
app = create_app()


def main() -> int:
    """Run the service, loopback unless exposing it was asked for explicitly.

    The same gate the Gradio UI already uses, for the same reason and with the
    same opt-in: a warning is not a gate, and this is the entry point someone
    runs on a laptop or a shared build host. A container platform does not come
    through here -- it runs uvicorn against `app` directly, and states the
    wildcard bind in its own deployment file where the choice is visible.
    """
    import uvicorn

    from rse.ui.app import check_bind

    config = Settings.from_env()
    allow_remote = os.environ.get("RSE_API_ALLOW_REMOTE", "").lower() in (
        "1",
        "true",
        "yes",
    )
    try:
        host = check_bind(config.host, allow_remote=allow_remote)
    except Exception as error:  # noqa: BLE001 - reported as a refusal
        print(f"error: {error}")
        print(
            "       set RSE_API_ALLOW_REMOTE=1 to override, or deploy with "
            "deploy/render.yaml, which states the bind explicitly."
        )
        return 2
    uvicorn.run("rse.api.app:app", host=host, port=config.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
