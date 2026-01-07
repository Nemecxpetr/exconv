# Scripts guide

Helper scripts live under `scripts/`. Run them from the repository root with
`python -m scripts.<name> ...` (or `python scripts/<name>.py ...`). Paths below
assume the default samples layout.

## Batch helpers

### `video_folderbatch.py` - project-wide video bi-conv
- Purpose: run `exconv video-biconv` over every video in a project folder.
- Inputs: `samples/input/video/<project>/*` (or `--root <root>`), optional
  `--recursive` for nested folders.
- Outputs: `samples/output/video/<project>/<name>_biconv.<ext>` by default.
- With `--variants all`, extra outputs: `<name>_..._video.<ext>` (processed video, original audio) and `<name>_..._audio.<ext>` (original video, processed audio).
  All three are rendered from a single processing pass; the extras are mux-only.
- Quick start:
  ```bash
  python -m scripts.video_folderbatch my_project \
    --jobs 2 --blas-threads 1 --suffix _biconv --serial-mode parallel
  ```
  Add `--dry-run` to see what would execute, or `--overwrite` to replace
  existing outputs.
- Options (grouped)

  | General / IO | Default | What it does |
  | --- | --- | --- |
  | `project` (positional) | — | Input project path under `input/video/`; also used for output unless `--out-project`. |
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
  | `--variants` | `both` | `all` also writes `<name>_video.*` (proc video + orig audio) and `<name>_audio.*` (orig video + proc audio). |

  | Pipeline (chaining + blocks) | Default | What it does |
  | --- | --- | --- |
  | `--serial-mode` | `parallel` | Bi-conv chaining (`serial-image-first`, `serial-sound-first`). |
  | `--block-size` | None | Fixed frames per block; overrides divisor when set. |
  | `--block-size-div` | `12` | Split into N blocks if `--block-size` not set. |

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
  | `--i2s-phase-mode` | `spiral` | Phase strategy (`zero`, `random`, `image`, `min-phase`, `spiral`). |
  | `--i2s-impulse-len` | `frame` | Impulse length (`int`, `auto`, `frame`). |

### `folderbatch.py` - audio + sound2image batches
- Purpose: batch self/pair audio convolution and optional sound->image sculpting.
- Inputs: `<root>/input/audio/<project>/*` (required),
  `<root>/input/img/<project>/*` (optional, enables sound->image).
- Outputs: `<root>/output/audio/<project>/{self,pair}/` and
  `<root>/output/sound2image/<project>/`.
- Quick start:
  ```bash
  python -m scripts.folderbatch my_project \
    --root samples \
    --audio-mode same-center --audio-order 2 --audio-normalize rms \
    --s2i-mode mono --s2i-colorspace luma
  ```
- Audio flags: `--audio-mode {full,same-first,same-center}` (size policy),
  `--audio-order <int>` (self-conv order), `--audio-circular`,
  `--audio-normalize {rms,peak,none}`, `--audio-subtype PCM_16|PCM_24|FLOAT`.
- Sound->image flags: `--s2i-mode {mono,stereo,mid-side}`,
  `--s2i-colorspace {luma,color}`, `--s2i-no-normalize` to keep raw output.

## Single-run helpers

- `video_biconv.py`: one-off CLI wrapper around `exconv video-biconv` for a
  single video/audio pair; exposes all per-video controls (serial mode,
  block size, audio length strategy, impulse settings).
- `image2sound_demo.py`: derive an impulse from an image (`flat`, `hist`,
  `radial` modes) and convolve an audio file; writes WAV plus an impulse plot.
- `sound2image_demo.py`: minimal sound->image example; choose mode and color
  mapping and save a sculpted image.
- `sweep_conv2d_options.py`: grid-sweep image auto/pair convolution settings;
  writes JPEGs and a metrics CSV under `samples/output/img_sweep/`.
- `animator.py`: turn a directory of images into a GIF (`python -m scripts.animator in_dir out.gif`).
- `test_video.py`: short scratchpad calling `sound2image_video_from_files`
  on bundled sample inputs.
