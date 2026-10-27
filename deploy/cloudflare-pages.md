# Deploying the UI to Cloudflare Pages

The page is static and the generator is Python, so they deploy separately:
Pages serves one HTML file, Render runs the API (`deploy/render.yaml`). Two
origins, which means the only real work is telling each one about the other.

## Pages project settings

| Setting | Value |
|---|---|
| Framework preset | None |
| Build command | `npm run build` |
| Build output directory | `frontend/dist` |
| Root directory | *(repo root — leave blank)* |
| Node version | 20 or newer |

Build-time environment variables:

| Name | Value |
|---|---|
| `VITE_API_BASE` | `https://rse-api.onrender.com` — the API's origin, scheme included, no trailing path |
| `NODE_VERSION` | `20` |

`NODE_VERSION` is worth setting explicitly even though `package.json` declares
`engines.node >= 18`. Vite 5 needs Node 18 or newer, and Cloudflare Pages'
default version depends on when the project was created — older projects were
given Node 12, which fails in a way that looks nothing like a version
problem. Declaring it in both places costs nothing.

The live project is **`retro-synth`**, so the page is at
`https://retro-synth.pages.dev` and that is the origin the API must allow.

`vite.config.js` substitutes it into the page and **fails the build** if it is
not an absolute `http(s)` origin, because a relative value here can never be
right: the whole purpose of the variable is to cross an origin. If it is
unset, the build succeeds with a warning and the page falls back to calling
its own origin — correct locally, useless on Pages.

## The matching API setting

On Render, set:

```
RSE_API_ALLOWED_ORIGINS = https://retro-synth.pages.dev
```

Exactly the Pages origin, no trailing slash, comma-separated if there is more
than one, and never `*`. It is `sync: false` in `render.yaml` — deliberately
unset, because the API refuses to guess an origin rather than defaulting to
permissive CORS. Until it is set, every request from Pages fails.

**Preview deployments will not work against production.** Pages gives each
branch its own subdomain (`<hash>.retro-synth.pages.dev`), which will not be
in the allowlist. That is the correct behaviour, not a bug — don't go
debugging it. If previews need to work, point them at a separate preview API
rather than widening the production allowlist.

## Verifying the deploy

The page can't tell a blocked cross-origin request from a dead server —
browsers deliberately hide the difference — so check the API directly:

```bash
# Is the API up?
curl -s https://rse-api.onrender.com/api/health

# Does it allow the Pages origin? Look for access-control-allow-origin.
curl -si -X OPTIONS https://rse-api.onrender.com/api/generate \
  -H 'Origin: https://rse.pages.dev' \
  -H 'Access-Control-Request-Method: POST' | grep -i access-control
```

Then load the page and generate once. If it fails, the page now names the
origin it tried and mentions `RSE_API_ALLOWED_ORIGINS`, which is enough to
tell the two causes apart.

## Deploy order (it matters)

There is a circular dependency: the page needs the API's origin baked in at
build time, and the API needs the page's origin in its CORS allowlist. Break
it by predicting the Pages URL, which is `<project-name>.pages.dev` and known
as soon as you name the project.

1. **Name the Pages project.** Done: `retro-synth`, so the page lands at
   `https://retro-synth.pages.dev`.
2. **On Render, set `RSE_API_ALLOWED_ORIGINS`** to that URL and deploy the
   API. Note the URL Render actually assigns.
3. **On Pages, set `VITE_API_BASE`** to that URL and deploy the page.

   `render.yaml` names the service `rse-api`, which normally gives
   `https://rse-api.onrender.com` — but Render appends a suffix when the name
   is taken, so the predicted value needs confirming against the real one. If
   it differs, correct the variable and retry the Pages deployment; the API
   origin is baked in at build time, so changing it requires a rebuild rather
   than just a refresh.
4. **Verify**, with the script below rather than by clicking around.

## Verifying

```bash
python scripts/verify_deploy.py \
    --api https://rse-api.onrender.com \
    --page https://rse.pages.dev
```

Sixteen checks: the API answers and reports both consoles, the claims table
is served with SPC export disclosed unsupported and echo declared inaudible,
the CORS preflight admits the page origin (and is not a wildcard), the page
loads with its API base substituted and its security headers applied, and
both consoles actually generate and return a downloadable RIFF file. Exit
status is 0 only if all of them pass.

It makes real generation requests, which costs CPU on whatever plan is
running. That is deliberate — a health endpoint returning 200 proves the
process is up, not that it can render a song.

## The API: free plan, 8-bit only

`render.yaml` (at the repo root, which is the only place Render looks for a
blueprint) is configured for the **free** plan with **SNES disabled**. The
reason is a 32× gap in render cost, measured rather than assumed:

| console | render rate | 180 s of audio, one core |
|---|---|---|
| NES | 0.022× realtime | 3.9 s |
| SNES | 0.70× realtime | 126 s |

Generation is synchronous inside the HTTP request, so render time *is*
request time and Render closes a request at 100 s. Scaling by each plan's
share of a core:

| plan | CPU | NES at 180 s | SNES at 60 s |
|---|---|---|---|
| free | 0.1 | ~40 s — fine | ~420 s — **times out** |
| starter | 0.5 | ~8 s — fine | ~84 s — 16 s margin, tight |
| standard | 1.0 | ~4 s — fine | ~42 s — comfortable |

So free serves NES with room to spare and cannot serve SNES at all.
Confirmed end-to-end against a server configured this way: a 180-second NES
track generated in 4.2 s with a valid 15.8 MB WAV, and a SNES request was
refused with HTTP 503.

**SNES is off by configuration, not removed.** `RSE_API_CONSOLES=nes` gates
it; the engine, the tests and local development are untouched. The
capabilities endpoint still lists the 16-bit console, marked unavailable with
the reason attached, so the page greys it out and explains itself rather than
silently dropping the option.

To enable it later: move to starter or above, set `RSE_API_CONSOLES=nes,snes`,
and lower `RSE_API_MAX_SECONDS` to 45 (starter) or 60 (standard) in the same
change — those two settings are coupled, and raising one without lowering the
other is how the 16-bit option starts returning 502s.

**The free plan sleeps after inactivity.** The first visitor after a quiet
period waits roughly half a minute for the container before their request is
served. That is the main reason to move up a tier even for NES alone.

## Three settings that changed for this deploy

All three were sized when NES mono was the only output, and stereo SNES
invalidated the arithmetic. Fixed in `render.yaml`:

| Setting | Was | Now | Why |
|---|---|---|---|
| *(file location)* | `deploy/render.yaml` | `render.yaml` | Render only looks for a blueprint at the repo root. In `deploy/` it would never have been found |
| `plan` | `starter` | `free` | Paid, and committed silently. Free serves NES fine |
| `PYTHON_VERSION` | *(unset)* | `3.12` | `pyproject` declares `>=3.11,<3.13`, so a runner defaulting to 3.13 fails the install outright — a build failure with nothing wrong in the code |
| `RSE_API_CONSOLES` | *(new)* | `nes` | SNES cannot render inside the request timeout on any plan the project is on yet |
| `RSE_API_MAX_STORED_SONGS` | `200` | `40` | NES mono is 0.088 MB/s, so 40 songs at the 180 s cap is 0.63 GB of ephemeral disk. At 200 with stereo SNES enabled it would have been **6.35 GB** — enough to fill the instance and take the service down |
| `RSE_API_RETENTION_SECONDS` | `1800` | `900` | Long enough to find the tab and download; keeps the disk arithmetic small |

`RSE_API_MAX_SECONDS` stays at 180, because NES is cheap enough for it. It is
**coupled to `RSE_API_CONSOLES`** — re-do the arithmetic whenever either
changes, since their product is what lands on disk and what the request
timeout has to absorb.

## What to expect in production

- **Cold starts.** Render's starter plan sleeps when idle. The first
  generation after a quiet period waits roughly 30 s for the container on top
  of ~1.5 s of actual rendering, and the page has no loading state tuned for
  that. Worth knowing before concluding the deploy is broken.
- **Download links are not durable.** Songs live in memory with
  `RSE_API_RETENTION_SECONDS=1800`, and a restart drops all of them. The WAV
  is fetched from the API, not from Pages, so a sleeping service also means a
  slow download.
- **No authentication, and that is correct here.** This is an anonymous
  public generator; gating it would lock out the audience it exists for. The
  rate limiter (20/min, burst 5) is the only protection, and it is per-process
  and in-memory, so it resets on every restart and does not hold across
  multiple instances.

## Not verified on this machine

`npm install` could not run here — npm is pointed at an internal registry
that rejects our credentials — so `npm run build` has not been executed
against real Vite.

**This does not block the deploy.** Cloudflare builds on its own runners
against the public registry, and nothing in the repository pins a registry:
there is no committed `.npmrc` and no lock file, so the Pages build resolves
`vite ^5.4.0` cleanly. Verified by listing the committed npm files, which is
empty.

The absence of a lock file is a real if minor gap — Pages will resolve
whatever is latest in 5.x at build time, so two builds of the same commit can
differ. Generating one needs a working registry. Run `npm install` on a
machine with public npm access and commit the resulting
`package-lock.json`.

What *was* verified, by running the actual code rather than reasoning about
it: the page's three-way API-base resolution (`?api=` beats the baked value
beats same-origin, in all four combinations) and the config plugin's
substitution, trailing-slash trimming, unset warning and two invalid-value
rejections. What remains unverified is only that Vite calls
`transformIndexHtml` on the entry HTML and copies `frontend/public/_headers`
into the output — both documented behaviour, neither exercised here.

So: run `npm install && npm run build` once on a machine with working
registry access, and confirm `frontend/dist/index.html` contains the API
origin instead of `%VITE_API_BASE%` and that `frontend/dist/_headers` exists.
