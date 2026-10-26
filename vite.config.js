// Dev server for the generator page.
//
// The page itself needs no build step -- `rse-api` serves it directly at
// http://127.0.0.1:8000 and that is the shortest path to a working generator.
// This config exists for two other reasons: it gives `npm run dev` hot reload
// while editing the page, and it is the on-ramp for the React port, which
// drops in without changing how the API is reached.
//
// `/api` is proxied to the Python service, so the page stays same-origin in
// development and needs no CORS configuration. Generation happens in Python;
// this server only serves the page.
import { defineConfig } from "vite";

export default defineConfig({
  root: "frontend",
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
    outDir: "../frontend/dist",
    emptyOutDir: true,
  },
});
