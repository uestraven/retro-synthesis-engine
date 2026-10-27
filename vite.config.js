// Dev server and static build for the generator page.
//
// The page needs no build step to work: `rse-api` serves `frontend/index.html`
// directly at http://127.0.0.1:8000, and that is the shortest path to a
// working generator. This config covers the other two cases -- `npm run dev`
// for hot reload while editing, and `npm run build` for hosting the page as a
// static file on Cloudflare Pages while the Python service runs elsewhere.
//
// In development `/api` is proxied to the Python service, so the page stays
// same-origin and needs no CORS configuration. Generation happens in Python;
// this server only ever serves the page.
import { defineConfig } from "vite";

/**
 * Substitute the API origin into the page at build time.
 *
 * Vite has its own `%VITE_FOO%` convention for HTML files, which would
 * probably do this. It is done explicitly instead, for one reason: it is
 * visible. The placeholder, the variable that fills it and the behaviour when
 * that variable is missing are all right here, rather than depending on
 * whether an HTML env hook applies inside an inline `<script>` -- and this
 * repository has twice shipped a defect that came from assuming a mechanism
 * behaved as remembered.
 *
 * Leaving the placeholder alone when the variable is unset is deliberate, not
 * laziness. `frontend/index.html` tests for a leading `%` and falls back to
 * same-origin, so an unset variable degrades to the local behaviour instead of
 * baking the string "undefined" into a deployed page.
 */
function apiBase() {
  const placeholder = "%VITE_API_BASE%";
  return {
    name: "rse-api-base",
    transformIndexHtml(html) {
      const base = (process.env.VITE_API_BASE || "").trim().replace(/\/$/, "");
      if (!base) {
        console.warn(
          "[rse] VITE_API_BASE is not set: the built page will call the " +
            "origin it is served from. That is correct only if the API " +
            "serves the page. For Cloudflare Pages, set VITE_API_BASE to " +
            "the API's origin (e.g. https://rse-api.onrender.com).",
        );
        return html;
      }
      if (!/^https?:\/\//.test(base)) {
        // Fail the build rather than emit a page that cannot reach anything.
        // A relative or scheme-less value here is always a mistake: the whole
        // point of this variable is to cross an origin.
        throw new Error(
          `VITE_API_BASE must be an absolute http(s) origin, got "${base}"`,
        );
      }
      console.log(`[rse] API base baked into the page: ${base}`);
      return html.split(placeholder).join(base);
    },
  };
}

export default defineConfig({
  root: "frontend",
  plugins: [apiBase()],
  server: {
    // Loopback. Vite's default is already localhost-only; stated here so it
    // cannot drift, and `--host` is never added to the dev script.
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: process.env.RSE_API_URL || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    // Relative to `root`, so this is `frontend/dist`.
    outDir: "dist",
    emptyOutDir: true,
  },
});
