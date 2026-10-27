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

Build-time environment variable:

| Name | Value |
|---|---|
| `VITE_API_BASE` | `https://rse-api.onrender.com` — the API's origin, scheme included, no trailing path |

`vite.config.js` substitutes it into the page and **fails the build** if it is
not an absolute `http(s)` origin, because a relative value here can never be
right: the whole purpose of the variable is to cross an origin. If it is
unset, the build succeeds with a warning and the page falls back to calling
its own origin — correct locally, useless on Pages.

## The matching API setting

On Render, set:

```
RSE_API_ALLOWED_ORIGINS = https://rse.pages.dev
```

Exactly the Pages origin, comma-separated if there is more than one, and
never `*`. It is `sync: false` in `render.yaml` — deliberately unset, because
the API refuses to guess an origin rather than defaulting to permissive CORS.
Until it is set, every request from Pages fails.

**Preview deployments will not work against production.** Pages gives each
branch its own subdomain (`abc123.rse.pages.dev`), which will not be in the
allowlist. That is the correct behaviour, not a bug — don't go debugging it.
If previews need to work, point them at a separate preview API rather than
widening the production allowlist.

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

1. **Name the Pages project.** That fixes `https://<name>.pages.dev`.
2. **On Render, set `RSE_API_ALLOWED_ORIGINS`** to that URL and deploy the
   API. Note its URL.
3. **On Pages, set `VITE_API_BASE`** to the API's URL and deploy the page.
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

## Three settings that changed for this deploy

All three were sized when NES mono was the only output, and stereo SNES
invalidated the arithmetic. Fixed in `render.yaml`:

| Setting | Was | Now | Why |
|---|---|---|---|
| `PYTHON_VERSION` | *(unset)* | `3.12` | `pyproject` declares `>=3.11,<3.13`, so a runner defaulting to 3.13 fails the install outright — a build failure with nothing wrong in the code |
| `RSE_API_MAX_SECONDS` | `180` | `60` | Generation is synchronous in the request, and the SNES sampler renders at **0.67–0.77× realtime** (measured: 115 s of audio → 82 s). Render closes a request at 100 s, so the longest SNES songs would 502 |
| `RSE_API_MAX_STORED_SONGS` | `200` | `40` | Stereo SNES is 0.176 MB/s, so 200 songs at the old cap was **6.35 GB** of ephemeral disk — enough to fill the instance. Now 0.42 GB |

**The render-rate limit is the one to re-check.** The 60-second cap assumes
the starter plan is within ~2× of a full core, which is an assumption and not
a measurement. Generate a 60-second SNES track on the real instance and time
it: comfortably inside 100 s means the cap can rise, a timeout means it
should fall. The structural fix is asynchronous generation with a job id,
which is a feature rather than a deployment setting.

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
