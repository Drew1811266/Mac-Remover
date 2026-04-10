# Embedded FFmpeg Runtime Assets

This project resolves `ffmpeg`/`ffprobe` in the following order:

1. `vendor/ffmpeg/<platform-arch>/ffmpeg|ffprobe`
2. System `PATH`

Current platform key format is `<system>-<arch>`, for example:

- `darwin-arm64`
- `darwin-x86_64`

## Bundled Files

- `darwin-arm64/ffmpeg`: copied from `imageio-ffmpeg` wheel in this development environment
- `darwin-arm64/ffprobe`: local shim script using embedded `ffmpeg` stream inspection
- `darwin-x86_64/ffmpeg`: embedded Intel binary (extracted from local offline archive)
- `darwin-x86_64/ffprobe`: local shim script using sibling embedded `ffmpeg` stream inspection (no PATH dependency)

## Intel Offline Embedding

`darwin-x86_64` runtime now ships with an embedded ffmpeg binary in-repo.
When running on Intel macOS, the app resolves and executes this embedded binary first.

## License / Compliance

See `vendor/ffmpeg/LICENSES/` for attribution and legal notes.
