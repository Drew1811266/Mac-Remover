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

The Python runtime (`src/gui/window.py`) prefers `dist/index.html` by default.
Set `WMR_UI_LEGACY=1` to force legacy Alpine UI.
