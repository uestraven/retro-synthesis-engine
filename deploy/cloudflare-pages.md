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

`npm install` could not run here — npm is pointed at a registry that rejects
our credentials — so `npm run build` has not been executed against real Vite.

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
