# Changelog

## Unreleased
- Added block-based video bi-conv (`--block-size`) so groups of frames share an audio chunk and the audio is driven by the mean image of each block.
- Added `video-biconv` subcommand to the main `exconv` CLI and auto-detection of video inputs when an audio path is a video.
- Added radial image->sound impulse derivation with multiple phase modes (zero/random/image/min-phase/spiral) and stereo/mid-side colorspaces.
- Added auto impulse length for image->sound and expanded demo CLI options.
- Introduced bi-directional video processor (`biconv_video_from_files` + `scripts/video_biconv.py`) with parallel/serial chaining, audio length strategies, optional ffmpeg-based audio extraction, and muxing.
- Added per-frame progress bars via `tqdm`.
- Documented new CLIs in README and noted ffmpeg/tqdm requirements.
- Added DSP helpers for envelopes, crossfades, normalization, windows, and segment slicing; exposed via `exconv.dsp`.
- Added block crossover modes (`none`, `lin`, `equal`, `power`) and ADSR shaping for video bi-conv blocks.
- Centralized ffprobe metadata parsing in `exconv.video_meta`.
- Refactored image->sound normalization to use shared DSP normalization helpers.
- Added a small DSP segments test and expanded documentation for crossovers and DSP utilities.
