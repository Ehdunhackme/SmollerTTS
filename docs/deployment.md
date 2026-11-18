# Static hosting

Build from the repository root:

```sh
npm ci
npm run setup
npm test
npm run build
```

Publish the contents of `dist/` under the host’s root. The demo will be at `/projects/smollertts/`. Alternatively, copy the contents of `dist/projects/smollertts/` into another directory; its asset URLs are relative.

Serve the directory over **HTTPS** in public, or HTTP on localhost during development. Do not open `index.html` directly through `file://`. Browser cryptographic integrity checks and workers require a suitable origin.

The host must serve `.mjs` as JavaScript and `.wasm` as `application/wasm`. Keep the `ort/`, `fonts/`, `icons/`, `licenses/`, HTML, ONNX and manifest files together. No API key, Python service or GPU endpoint is needed.

The current single-thread runtime does not require cross-origin isolation headers. Avoid applying a generic single-page-app fallback to missing model or runtime files: a missing ONNX file should return 404, not HTML.

Use revalidation for HTML and manifests so updates can be discovered. The build adds content-derived query keys to its scripts and styles; model downloads are keyed by SHA-256 and checked in the worker. Upload the full build together rather than mixing assets from different builds.

Before making a hosted URL public, check a fresh browser profile, generation, WAV download, reload, mobile layout and the Credits page. This repository contains a runnable demo; publishing it to a personal domain is a separate hosting step.
