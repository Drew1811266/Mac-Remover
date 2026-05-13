# Functional Testing

This project treats the Electron app as the acceptance target. Browser/Vite checks
are useful during development, but final UI and workflow evidence should come from
the desktop shell.

## Commands

Run the standard regression gate:

```bash
cd frontend
npm run typecheck
npm run build
npm test
npm run electron:build
```

Generate deterministic media fixtures:

```bash
cd frontend
npm run functional:fixtures
```

Run Electron service-level functional smoke tests:

```bash
cd frontend
npm run functional:smoke
```

Run real Electron UI smoke tests with DevTools Protocol screenshots:

```bash
cd frontend
npm run functional:ui
```

Run the local functional gate:

```bash
cd frontend
npm run functional:test
```

The functional scripts write generated videos, JSON reports, and screenshots under
`frontend/.functional-test/`. That directory is local-only and ignored by Git.

## Fixture Coverage

`functional:fixtures` creates:

- `5s_720p_h264_aac.mp4`: H.264/AAC video with a visible top-right watermark block.
- `5s_720p_no_audio.mp4`: H.264 video without audio.
- `3s_webm.webm`: WebM video for import/metadata compatibility.
- `invalid.txt`: invalid media for negative-path validation.

The fixture manifest also includes single ROI, multi ROI, and edge ROI annotation
segments. Edge ROI intentionally extends past the bottom-right frame bounds so
sidecar normalization and coordinate clipping can be verified.

## Automated Coverage

`functional:smoke` validates:

- bundled FFmpeg/FFprobe resolution and execution;
- media info for MP4/WebM and invalid-file rejection;
- preview session open/read/close behavior and JPEG data URLs;
- sidecar save/load/delete, required fields, and ROI normalization;
- settings persistence for language, theme, output path, and model selection;
- model download status and structured errors without a manifest;
- Real-ESRGAN/SeedVR2 blocked-state behavior while native assets are unavailable;
- removed ProPainter requests are rejected without fallback;
- LaMa processing only when LaMa ONNX assets are installed.

Use `node scripts/functional-smoke.mjs --require-processing` when the environment
must prove full LaMa processing. Without that flag, missing LaMa ONNX is reported
as a skip rather than a failure.

`functional:ui` launches the compiled Electron app directly, connects via Chrome
DevTools Protocol, and validates:

- `window.wmr` preload API exists;
- the app shell renders meaningful content;
- top title and navigation render correctly;
- the left rail and top bar do not show hard divider borders;
- primary pages can be opened through the navigation rail;
- settings and manual panels open without blanking the app;
- screenshots are captured from the real Electron window.

## Manual Acceptance Pass

After automation passes, run a short manual pass in the real app:

1. Start with `./run.sh`.
2. Import `5s_720p_h264_aac.mp4`.
3. Confirm metadata, playback, pause, and frame stepping.
4. Draw a top-right ROI, set an in/out frame range, and save annotations.
5. Reopen the same video and confirm the sidecar restores the ROI.
6. Run LaMa processing when LaMa ONNX assets are installed; confirm output video,
   preserved audio, result preview, and open-output-dir behavior.
7. Confirm removed ProPainter requests are rejected, and Real-ESRGAN/SeedVR2
   show clear blocked/unavailable states when native same-model assets are absent.
8. Save settings, restart Electron, and confirm settings persist.
9. Check `1400x900` and `1200x700` windows for clipped text, hidden buttons,
   page-level scrolling, or layout overlap.

## Release Gate

Package builds should pass:

```bash
cd frontend
npm run electron:pack -- --mac
npm run electron:verify-release
```

Windows x64 packages must additionally pass startup, import, preview, settings
persistence, model status display, and installer launch smoke tests on a Windows
machine or CI runner. Full processing is required there only when native assets
for that platform are installed.
