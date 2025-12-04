# Changelog

## Unreleased
- Added radial image->sound impulse derivation with multiple phase modes (zero/random/image/min-phase/spiral) and stereo/mid-side colorspaces.
- Added auto impulse length for image->sound and expanded demo CLI options.
- Introduced bi-directional video processor (`biconv_video_from_files` + `scripts/video_biconv.py`) with parallel/serial chaining, audio length strategies, optional ffmpeg-based audio extraction, and muxing.
- Added per-frame progress bars via `tqdm`.
- Documented new CLIs in README and noted ffmpeg/tqdm requirements.
