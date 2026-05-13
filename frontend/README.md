# WMR Frontend (React + TypeScript + Semi)

## Goal

This directory hosts the new large-screen manual-annotation workspace frontend.

## Local development

```bash
cd frontend
npm install
npm run dev
```

## Build to runtime dist

```bash
./scripts/build_frontend.sh
```

Build output target:

- `src/gui/templates/dist/`

Electron loads this output through its preload-isolated renderer path.

## Electron desktop shell

The Electron shell is the cross-platform desktop runtime. It keeps the React UI
and exposes a narrow `window.wmr` preload API; the renderer no longer falls back
to the legacy desktop bridge.

```bash
cd frontend
npm run electron:dev
```

Use `npm run electron:dev` for day-to-day feature development. It launches the
development Electron runtime directly and does not depend on a packaged `.app`.

The Electron core supports the manual annotation workflow, preview, sidecar
files, settings, model status/download hooks, and LaMa ONNX processing.
ProPainter is intentionally not embedded in the Electron product because it is
too heavy for the target Mac workload. Real-ESRGAN and SeedVR2 entries remain
visible for the upscale workflow but are blocked until official/self-owned
native assets and runners are supplied.

Model download uses a manifest URL:

```bash
WMR_MODEL_MANIFEST_URL=https://example.com/model-manifest.json npm run electron:dev
```

The application expects the manifest to provide model assets using the unified
`asset_id`, `model_id`, `engine`, `runtime_kind`, `platform`, `arch`,
`license`, and `sha256` fields. LaMa requires `big-lama-onnx`; Real-ESRGAN and
SeedVR2 require same-model non-Python assets before they can process media.

Release packages are only for distribution checks and must pass the Python-free scan:

```bash
npm run electron:verify-release
```
