# CLI + scripts guide

Batch workflows have moved into the `exconv` CLI. The scripts under `scripts/`
are still available for backward compatibility or experiments, but the
recommended entry points are the CLI subcommands below. Paths assume the
default `samples/` layout.
For parameter meanings and cross-references, see `docs/api.md` (CLI section)
and `docs/design.md` (video block and cross-modal sections).

## Examples gallery
- Instagram highlight: [Example highlight on Instagram](https://www.instagram.com/stories/highlights/18441302296108489/)

## Batch helpers (CLI)

### Settings files (shared)
- Use `--settings <path>` to load option defaults from a JSON or CSV file.
- Use `--save-settings <path>` to write the current option values (non-positional) for the command.
- JSON can store multiple commands by name (e.g., `video-folderbatch`, `video-biconv`); CSV is flat `key,value`.
- CLI flags always override values loaded from settings.

### `exconv video-folderbatch` - project-wide video bi-conv
- Purpose: run `exconv video-biconv` over every video in a project folder.
- Inputs: `samples/input/video/<project>/*` (or `--root <root>`), optional
  `--recursive` for nested folders.
- Outputs: `samples/output/video/<project>/<name>_biconv.<ext>` by default.
- With `--variants all`, extra outputs: `<name>_..._video.<ext>` (processed
  video, original audio) and `<name>_..._audio.<ext>` (original video, processed
  audio). All three are rendered from a single processing pass; the extras are
  mux-only.
- Quick start:
  ```bash
  exconv video-folderbatch my_project \
    --jobs 2 --blas-threads 1 --suffix _biconv --serial-mode parallel
  ```
  Crossover example:
  ```bash
  exconv video-folderbatch my_project \
    --serial-mode parallel --block-size 24 \
    --crossover equal --crossover-frames 2
  ```
  Add `--dry-run` to see what would execute, or `--overwrite` to replace
  existing outputs.
- Options (grouped)

  | General / IO | Default | What it does |
  | --- | --- | --- |
  | `project` (positional) | required | Input project path under `input/video/`; also used for output unless `--out-project`. |
  | `--root` | `samples` | Base folder containing `input/` and `output/`. |
  | `--out-project` | None | Separate output project path under `output/video/`. |
  | `--recursive` | off | Recurse into subfolders of the project input dir. |
  | `--jobs` | `1` | Parallel worker processes (one per video). |
  | `--blas-threads` | None | Force BLAS threads; set to `1` when `--jobs>1` to avoid oversubscription. |
  | `--suffix` | `_biconv` | Added before the extension for outputs. |
  | `--overwrite` | off | Replace existing outputs. |
  | `--dry-run` | off | Print planned outputs only. |
  | `--audio` | None | Use one audio file for all videos; otherwise extract from each video. |
  | `--fps` | None | Override FPS when metadata is wrong/missing. |
  | `--fps-guard` | `auto` | Detect mismatched FPS metadata (`off`, `ask`, `auto`). |
  | `--variants` | `both` | `all` also writes `<name>_video.*` (proc video + orig audio) and `<name>_audio.*` (orig video + proc audio). |

  | Pipeline (chaining + blocks) | Default | What it does |
  | --- | --- | --- |
  | `--serial-mode` | `parallel` | Bi-conv chaining (`serial-image-first`, `serial-sound-first`). |
  | `--block-strategy` | `fixed` | `fixed` (frame-count), `beats`/`novelty`/`structure` (audio-driven). |
  | `--block-size` | None | Fixed frames per block (fixed strategy only). |
  | `--block-size-div` | `12` | Split into N blocks if `--block-size` not set (fixed strategy only). |
  | `--block-min-frames` | `1` | Minimum block length (audio-driven strategies). |
  | `--block-max-frames` | None | Maximum block length (audio-driven strategies). |
  | `--beats-per-block` | `1` | Group this many beats into a block (beats strategy). |
  | `--crossover` | `none` | Crossfade mode across block boundaries (`none`, `lin`, `equal`, `power`). |
  | `--crossover-frames` | `1` | Frames per block side used for blending; overlap is clamped to block lengths. |
  | `--block-adsr-attack-s` | `0.0` | Attack time in seconds. |
  | `--block-adsr-decay-s` | `0.0` | Decay time in seconds. |
  | `--block-adsr-sustain` | `1.0` | Sustain level [0..1]. |
  | `--block-adsr-release-s` | `0.0` | Release time in seconds. |
  | `--block-adsr-curve` | `linear` | Envelope curve (`linear`, `equal-energy`, `equal-power`). |

  Crossover modes:
  - `none`: hard boundary, no blending.
  - `lin`: linear amplitude fade (`w_prev = 1 - t`, `w_cur = t`).
  - `equal`: equal-energy fade (`w = sqrt(t)`).
  - `power`: equal-power fade (`w_prev = cos(t*pi/2)`, `w_cur = sin(t*pi/2)`).

  | Sound->image (s2i) | Default | What it does |
  | --- | --- | --- |
  | `--s2i-mode` | `mid-side` | Sound->image mode (`mono`, `stereo`, `mid-side`). |
  | `--s2i-colorspace` | `color` | Sound->image colorspace (`luma`, `color`). |
  | `--s2i-unsafe-color` | off | Disable chroma-safe filtering. |
  | `--s2i-chroma-strength` | `0.05` | Safe-color strength. |
  | `--s2i-chroma-clip` | `0.1` | Safe-color clip radius. |

  | Image->sound (i2s) | Default | What it does |
  | --- | --- | --- |
  | `--i2s-mode` | `radial` | Image->sound impulse (`flat`, `hist`, `radial`). |
  | `--i2s-colorspace` | `ycbcr-mid-side` | Image->sound colorspace (`luma`, `rgb-mean`, `rgb-stereo`, `ycbcr-mid-side`). |
  | `--i2s-pad-mode` | `same-center` | Convolution pad mode (`full`, `same-center`, `same-first`). |
  | `--i2s-impulse-len` | `frame` | Impulse length (`int`, `auto`, `frame`). |
  | `--i2s-radius-mode` | `linear` | Radial binning (`linear`, `log`) for `radial` mode. |
  | `--i2s-phase-mode` | `spiral` | Phase strategy (`zero`, `random`, `image`, `min-phase`, `spiral`). |
  | `--i2s-smoothing` | `hann` | Radial profile smoothing (`none`, `hann`). |
  | `--i2s-impulse-norm` | `energy` | Impulse normalization (`energy`, `peak`, `none`). |
  | `--i2s-out-norm` | `match_rms` | Output normalization (`match_rms`, `match_peak`, `none`). |
  | `--i2s-n-bins` | `256` | Histogram bins (`hist` mode). |

Note: `scripts/_video_folderbatch.py` still exists as a legacy wrapper; use
`exconv video-folderbatch` for the supported interface.

### `exconv folderbatch` - audio + sound2image batches
- Purpose: batch self/pair audio convolution and optional sound->image sculpting.
- Inputs: `<root>/input/audio/<project>/*` (required),
  `<root>/input/img/<project>/*` (optional, enables sound->image).
- Outputs: `<root>/output/audio/<project>/{self,pair}/` and
  `<root>/output/sound2image/<project>/`.
- Quick start:
  ```bash
  exconv folderbatch my_project \
    --root samples \
    --audio-mode same-center --audio-order 2 --audio-normalize rms \
    --s2i-mode mono --s2i-colorspace luma
  ```
- Audio flags: `--audio-mode {full,same-first,same-center}` (size policy),
  `--audio-order <int>` (self-conv order), `--audio-circular`,
  `--audio-normalize {rms,peak,none}`, `--audio-subtype PCM_16|PCM_24|FLOAT`.
- Sound->image flags: `--s2i-mode {mono,stereo,mid-side}`,
  `--s2i-colorspace {luma,color}`, `--s2i-no-normalize` to keep raw output.
- Animation flags (per audio file): `--s2i-animate` (enable), `--s2i-animate-format {gif,mp4,both}` (default `mp4`),
  `--s2i-animate-fps <float>`, `--s2i-animate-loop <int>` (GIF), `--s2i-animate-audio` / `--s2i-animate-no-audio`
  (mp4 mux, ffmpeg required; defaults to on for mp4/both).
  Animations are written to `<root>/output/sound2image/<project>/animations/`.

Note: `scripts/folderbatch.py` still exists as a legacy wrapper; use
`exconv folderbatch` for the supported interface.

## Other CLI helpers

### `exconv animate` - create GIF/MP4 from image sequence
- Purpose: turn a folder of images into a GIF or MP4.
- Example:
  ```bash
  exconv animate samples/output/sound2image/my_project/animations \
    samples/output/sound2image/my_project/preview.mp4 --fps 12
  ```
- Options: `--pattern "*.png"` to filter frames, `--duration` to set seconds per
  frame, `--audio` to mux audio into MP4 (ffmpeg required).

## Single-run helpers (scripts)

- `video_biconv.py`: legacy wrapper around `exconv video-biconv` for a single
  video/audio pair (the CLI command is the canonical entry point).
- `image2sound_demo.py`: derive an impulse from an image (`flat`, `hist`,
  `radial` modes) and convolve an audio file; writes WAV plus an impulse plot.
- `sound2image_demo.py`: minimal sound->image example; `exconv sound2image` is
  the supported CLI equivalent.
- `sweep_conv2d_options.py`: grid-sweep image auto/pair convolution settings;
  writes JPEGs and a metrics CSV under `samples/output/img_sweep/`.
- `animator.py`: legacy wrapper; use `exconv animate` instead.
- `test_video.py`: short scratchpad calling `sound2image_video_from_files`
  on bundled sample inputs.
