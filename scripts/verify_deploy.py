#!/usr/bin/env python3
"""Check a deployed RSE: the API, the page, and the CORS link between them.

Run this after deploying, not before declaring it done. It exists because the
page cannot tell you what is wrong -- browsers deliberately make a blocked
cross-origin request indistinguishable from a dead server, so "it doesn't
work" from a browser is the same message for a sleeping service, a missing
environment variable and a typo in an origin. This asks the server directly.

    python scripts/verify_deploy.py \
        --api https://rse-api.onrender.com \
        --page https://rse.pages.dev

Exit status is 0 only if every check passes, so it is usable in a pipeline.

It makes real generation requests against the deployed service, which costs
CPU on whatever plan is running it. That is the point -- a health endpoint
answering 200 says the process is up, not that it can render a song.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

TIMEOUT = 120


class Check:
    """A named pass/fail with the evidence attached."""

    def __init__(self) -> None:
        self.rows: list[tuple[bool, str, str]] = []

    def record(self, ok: bool, name: str, detail: str) -> bool:
        self.rows.append((ok, name, detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}: {detail}")
        return ok

    @property
    def failures(self) -> int:
        return sum(1 for ok, _, _ in self.rows if not ok)


def fetch(url: str, *, method: str = "GET", data: dict | None = None,
          headers: dict[str, str] | None = None, timeout: int = TIMEOUT):
    """Return (status, headers, body-bytes). No network problem raises.

    Both failure classes are RESULTS here, not exceptions, because a checker
    that dies on the most common failure is useless. An HTTP error carries its
    own status; an unreachable host, bad DNS or timeout gets status 0 and the
    reason in the body. The first version of this caught only HTTPError and
    crashed with a traceback the moment it was pointed at a host that was
    down -- the exact case it exists to report.
    """
    body = json.dumps(data).encode() if data is not None else None
    request = urllib.request.Request(url, data=body, method=method)
    request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        reason = getattr(error, "reason", error)
        return 0, {}, f"unreachable: {reason}".encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", required=True, help="API origin, e.g. https://rse-api.onrender.com")
    parser.add_argument("--page", required=True, help="Page origin, e.g. https://rse.pages.dev")
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="skip the two generation requests (faster, but proves much less)",
    )
    args = parser.parse_args()
    api = args.api.rstrip("/")
    page = args.page.rstrip("/")
    check = Check()

    print(f"\nAPI  {api}\nPAGE {page}\n")

    # --- the API is up and is the right thing ---------------------------
    print("API")
    started = time.monotonic()
    status, _, body = fetch(f"{api}/api/health")
    elapsed = time.monotonic() - started
    ok = check.record(
        status == 200,
        "health",
        f"HTTP {status} in {elapsed:.1f}s"
        if status
        else body.decode("utf-8", "replace"),
    )
    if ok and elapsed > 10:
        # Not a failure. Render's free and starter plans sleep, and the first
        # request pays for the wake-up; the user should know that is what
        # they are seeing rather than a slow renderer.
        print(f"         note: {elapsed:.0f}s suggests a cold start, not slow rendering")
    if ok:
        health = json.loads(body)
        check.record(
            health.get("model_required") is False,
            "no model required",
            f"model_required={health.get('model_required')}",
        )
        consoles = health.get("consoles_available", [])
        check.record(
            {"nes", "snes"} <= set(consoles), "both consoles", f"{consoles}"
        )

    status, _, body = fetch(f"{api}/api/capabilities")
    if check.record(status == 200, "capabilities", f"HTTP {status}"):
        caps = json.loads(body)
        claims = {c["capability"]: c["claim"] for c in caps.get("snes_claims", [])}
        check.record(bool(claims), "SNES claims served", f"{len(claims)} rows")
        check.record(
            claims.get("SPC export") == "Unsupported",
            "SPC export disclosed unsupported",
            f"{claims.get('SPC export')!r}",
        )
        echo = caps.get("echo_presets", [])
        check.record(
            bool(echo) and all(not p["affects_audio"] for p in echo),
            "echo declared inaudible",
            f"{len(echo)} presets, none claiming audio",
        )

    # --- CORS: the one thing the page cannot diagnose for itself --------
    print("\nCORS")
    status, headers, _ = fetch(
        f"{api}/api/generate",
        method="OPTIONS",
        headers={
            "Origin": page,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
        timeout=30,
    )
    allowed = headers.get("access-control-allow-origin") or headers.get(
        "Access-Control-Allow-Origin"
    )
    check.record(
        allowed in (page, "*"),
        "preflight allows the page origin",
        f"HTTP {status}, access-control-allow-origin={allowed!r}"
        + ("  -> set RSE_API_ALLOWED_ORIGINS on the API" if not allowed else ""),
    )
    if allowed == "*":
        check.record(False, "origin is not a wildcard", "'*' is never the right answer here")

    # --- the page is reachable and points at this API -------------------
    print("\nPAGE")
    status, headers, body = fetch(page, timeout=60)
    if check.record(
        status == 200,
        "page loads",
        f"HTTP {status}" if status else body.decode("utf-8", "replace"),
    ):
        html = body.decode("utf-8", "replace")
        # The build must have substituted the placeholder. If it did not, the
        # page will call its own origin and every request will 404.
        check.record(
            "%VITE_API_BASE%" not in html,
            "API base was substituted at build time",
            "placeholder is gone" if "%VITE_API_BASE%" not in html
            else "placeholder still present -> set VITE_API_BASE and rebuild",
        )
        check.record(
            api in html, "page points at this API", f"{api} appears in the HTML"
        )
        check.record(
            headers.get("x-content-type-options") == "nosniff",
            "security headers applied",
            f"x-content-type-options={headers.get('x-content-type-options')!r}"
            " (from frontend/public/_headers)",
        )

    # --- it can actually render ------------------------------------------
    if not args.skip_generate:
        print("\nGENERATION")
        for console, expect_stereo in (("nes", False), ("snes", True)):
            request = {
                "console": console, "style": "boss", "seed": 20261027, "seconds": 20,
            }
            started = time.monotonic()
            status, _, body = fetch(f"{api}/api/generate", method="POST", data=request)
            elapsed = time.monotonic() - started
            if not check.record(
                status == 200, f"{console} generates", f"HTTP {status} in {elapsed:.1f}s"
            ):
                continue
            song = json.loads(body)
            snes = song.get("snes")
            check.record(
                (snes is not None) == expect_stereo,
                f"{console} renderer is the right one",
                "snes block present" if snes else "no snes block (correct for NES)",
            )
            if snes:
                check.record(
                    snes["hardware_exact"] is False
                    and snes["hardware_budgeted"] is False,
                    "SNES claims stay false",
                    f"hardware_exact={snes['hardware_exact']}, "
                    f"hardware_budgeted={snes['hardware_budgeted']}",
                )
            # The download has to work, not just be advertised.
            status, _, wav = fetch(api + song["links"]["wav"])
            check.record(
                status == 200 and wav[:4] == b"RIFF",
                f"{console} WAV downloads",
                f"HTTP {status}, {len(wav):,} bytes, "
                f"{'RIFF header ok' if wav[:4] == b'RIFF' else 'NOT a WAV'}",
            )

    print(f"\n{len(check.rows) - check.failures}/{len(check.rows)} checks passed")
    if check.failures:
        print("\nFailures above. The two that account for most bad deploys:")
        print("  - CORS: RSE_API_ALLOWED_ORIGINS on the API must name the page origin")
        print("  - API base: VITE_API_BASE must be set when the page is built")
        return 1
    print("Deployment verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
