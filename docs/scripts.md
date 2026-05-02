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
- Use `--show-settings` to print the resolved option values as JSON and exit.
- Use `--save-settings <path> --update-settings` to update only the provided option values in-place and exit.
- JSON can store multiple commands by name (e.g., `video-folderbatch`, `video-biconv`); CSV is flat `key,value`.
- CLI flags always override values loaded from settings.

### `exconv video-biconv` - one video bi-conv render
- Purpose: run sound->image and image->sound processing over one video.
- Inputs: a video file plus optional external audio. If `--audio` is omitted,
  audio is extracted from the input video.
- Outputs: processed video and, optionally, processed audio.
- Quick start:
  ```bash
  exconv video-biconv \
    --video input.mp4 \
    --out-video out_biconv.mp4 \
    --out-audio out_biconv.wav \
    --serial-mode parallel \
    --audio-length-mode pad-zero
  ```

  | General / IO | Default | What it does |
  | --- | --- | --- |
  | `--video` | required | Input video file. |
  | `--audio` | None | External audio file; omitted means extract from input video. |
  | `--out-video` | required | Output video path. |
  | `--out-audio` | None | Optional processed audio output. |
  | `--preview-seconds` | None | Process only the first N seconds. |
  | `--fps` | None | Override FPS. |
  | `--fps-policy` | `auto` | FPS selection: `auto`, `metadata`, `avg_frame_rate`, `r_frame_rate`. |
  | `--mux` / `--no-mux` | `--mux` | Mux processed audio into output video. |
  | `--upscale` | `1.0` | Output scale factor. |
  | `--upscale-method` | `lanczos` | Upscale method; opencv-* requires a model. |
  | `--upscale-model` | None | Model path for opencv-* upscalers. |

  | Pipeline (chaining + blocks) | Default | What it does |
  | --- | --- | --- |
  | `--serial-mode` | `parallel` | `parallel`, `serial-image-first`, `serial-sound-first`. |
  | `--audio-length-mode` | `pad-zero` | `trim`, `pad-zero`, `pad-loop`, `pad-noise`, `center-zero`. |
  | `--block-size` | `1` | Fixed frames per block. |
  | `--block-size-div` | None | Split the video into N fixed blocks; overrides `--block-size`. |
  | `--block-strategy` | `fixed` | `fixed`, `beats`, `novelty`, `structure`. |
  | `--block-min-frames` | `1` | Minimum block length for audio-driven strategies. |
  | `--block-max-frames` | None | Maximum block length for audio-driven strategies. |
  | `--beats-per-block` | `1` | Group this many beats per block. |
  | `--crossover` | `none` | Crossfade mode: `none`, `lin`, `equal`, `power`. |
  | `--crossover-frames` | `1` | Frames per block side used for crossover blending. |
  | `--block-adsr-attack-s` | `0.0` | Per-block audio ADSR attack in seconds. |
  | `--block-adsr-decay-s` | `0.0` | Per-block audio ADSR decay in seconds. |
  | `--block-adsr-sustain` | `1.0` | Per-block sustain level in `[0,1]`. |
  | `--block-adsr-release-s` | `0.0` | Per-block audio ADSR release in seconds. |
  | `--block-adsr-curve` | `linear` | `linear`, `equal-energy`, `equal-power`. |

  | Sound->image (s2i) | Default | What it does |
  | --- | --- | --- |
  | `--s2i-mode` | `mono` | `mono`, `stereo`, `mid-side`. |
  | `--s2i-colorspace` | `luma` | `luma`, `color`. |
  | `--s2i-safe-color` / `--s2i-unsafe-color` | safe on | Enable or disable chroma-safe color normalization. |
  | `--s2i-chroma-strength` | `0.5` | Blend between original and filtered chroma. |
  | `--s2i-chroma-clip` | `0.25` | Max chroma deviation around neutral. |

  | Image->sound (i2s) | Default | What it does |
  | --- | --- | --- |
  | `--i2s-mode` | `radial` | `flat`, `hist`, `radial`. |
  | `--i2s-colorspace` | `luma` | `luma`, `rgb-mean`, `rgb-stereo`, `ycbcr-mid-side`. |
  | `--i2s-pad-mode` | `same-center` | `full`, `same-center`, `same-first`. |
  | `--i2s-impulse-len` | `auto` | Integer, `auto`, or `frame`. |
  | `--i2s-radius-mode` | `linear` | `linear`, `log`. |
  | `--i2s-phase-mode` | `zero` | `zero`, `random`, `image`, `min-phase`, `spiral`. |
  | `--i2s-smoothing` | `hann` | `none`, `hann`. |
  | `--i2s-impulse-norm` | `energy` | `energy`, `peak`, `none`. |
  | `--i2s-out-norm` | `match_rms` | `match_rms`, `match_peak`, `none`. |
  | `--i2s-n-bins` | `256` | Histogram bins for `hist` mode. |

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
  | `--preview-seconds` | None | Limit processing to the first N seconds (quick test run). |
  | `--variants` | `both` | `all` also writes `<name>_video.*` (proc video + orig audio) and `<name>_audio.*` (orig video + proc audio). |
  | `--upscale` | `1.0` | Output scale factor (1.0 disables). |
  | `--upscale-method` | `lanczos` | Upscale method; opencv-* requires a model file. |
  | `--upscale-model` | None | Model path for opencv-* upscalers. |

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
- Purpose: batch self/pair audio convolution and optional sound->image sculpting. The all-files N-fold output is available as an explicit opt-in.
- Inputs: `<root>/input/audio/<project>/*` (required),
  `<root>/input/img/<project>/*` (optional, enables sound->image).
- Outputs: `<root>/output/audio/<project>/{self,pair}/` and
  `<root>/output/sound2image/<project>/`. With `--audio-multi` or
  `--audio-multi-circular`, also writes `<root>/output/audio/<project>/multi/`.
- Quick start:
  ```bash
  exconv folderbatch my_project \
    --root samples \
    --audio-mode same-center --audio-order 2 --audio-normalize rms \
    --s2i-mode mono --s2i-colorspace luma
  ```

  | General / IO | Default | What it does |
  | --- | --- | --- |
  | `project` (positional) | required | Project name under `<root>/input/audio/` and optionally `<root>/input/img/`. |
  | `--root` | `samples` | Base folder containing `input/` and `output/`. |

  | Audio convolution | Default | What it does |
  | --- | --- | --- |
  | `--audio-mode` | `same-center` | Linear convolution size policy: `full`, `same-first`, `same-center`. |
  | `--audio-order` | `2` | Self-convolution order for files in the `self/` output. |
  | `--audio-circular` | off | Use circular convolution for self, pair and multi outputs. |
  | `--audio-multi` | off | Also write one all-files N-fold convolution into `multi/`. |
  | `--audio-multi-circular` | off | Enable the all-files N-fold output and use circular convolution for it. This is the practical choice for long projects. |
  | `--audio-normalize` | `rms` | Output normalization: `rms`, `peak`, `none`. |
  | `--audio-subtype` | `PCM_16` | libsndfile subtype, for example `PCM_16`, `PCM_24`, or `FLOAT`. |

  Multi-convolution note: linear N-fold convolution can be very large because
  full length is `sum(lengths) - (N - 1)`. It is disabled by default. If an
  explicitly requested linear multi run hits memory limits, the command retries
  the multi output as circular unless `--audio-multi-circular` was already set.

  | Sound->image | Default | What it does |
  | --- | --- | --- |
  | `--s2i-mode` | `mono` | Audio mapping: `mono`, `stereo`, `mid-side`. |
  | `--s2i-colorspace` | `luma` | Image filtering space: `luma`, `color`. |
  | `--s2i-no-normalize` | off | Disable final sound->image normalization/clipping. |
  | `--s2i-upscale` | `1.0` | Output scale factor; `1.0` disables. |
  | `--s2i-upscale-method` | `lanczos` | Upscale method; opencv-* requires a model. |
  | `--s2i-upscale-model` | None | Model path for opencv-* upscalers. |

  | Animation | Default | What it does |
  | --- | --- | --- |
  | `--s2i-animate` | off | Write per-audio animations from sound->image frames. |
  | `--s2i-animate-format` | `mp4` | `gif`, `mp4`, or `both`. |
  | `--s2i-animate-fps` | `10.0` | Animation frame rate. |
  | `--s2i-animate-loop` | `0` | GIF loop count; `0` means infinite. |
  | `--s2i-animate-audio` | auto | Mux source audio into MP4 animations. |
  | `--s2i-animate-no-audio` | off | Disable MP4 audio muxing. |

  Animations are written to `<root>/output/sound2image/<project>/animations/`.

Note: `scripts/folderbatch.py` still exists as a legacy wrapper; use
`exconv folderbatch` for the supported interface.

## Other CLI helpers

### `exconv audio-auto` - one-file audio self convolution
- Purpose: self-convolve a single audio file.
- Example:
  ```bash
  exconv audio-auto --in input.wav --out out_auto.wav --mode same-center --order 2
  ```

  | Option | Default | What it does |
  | --- | --- | --- |
  | `--in` | required | Input audio file. |
  | `--out` | required | Output audio file. |
  | `--mode` | `same-center` | Linear size policy: `full`, `same-first`, `same-center`. |
  | `--order` | `2` | Self-convolution order; `1` returns a copy. |
  | `--circular` | off | Use circular convolution, preserving input length. |
  | `--normalize` | `rms` | Output normalization: `rms`, `peak`, `none`. |
  | `--subtype` | `PCM_16` | libsndfile output subtype. |

### `exconv img-auto` - one-image auto/pair convolution
- Purpose: self-convolve an image, or convolve it with a generated Gaussian kernel.
- Example:
  ```bash
  exconv img-auto --in img.png --out out.png --mode same-center --colorspace channels
  ```

  | Option | Default | What it does |
  | --- | --- | --- |
  | `--in` | required | Input image. |
  | `--out` | required | Output image. |
  | `--mode` | `same-center` | Linear size policy: `full`, `same-first`, `same-center`. |
  | `--circular` | off | Use circular convolution. |
  | `--colorspace` | `channels` | `channels` for per-channel RGB, `luma` for luminance. |
  | `--normalize` | `rescale` | Image normalization: `clip`, `rescale`, `none`. |
  | `--kernel` | None | Optional Gaussian spec such as `gaussian:sigma=2.0,radius=7`. |
  | `--upscale` | `1.0` | Output scale factor; `1.0` disables. |
  | `--upscale-method` | `lanczos` | Upscale method; opencv-* requires a model. |
  | `--upscale-model` | None | Model path for opencv-* upscalers. |

### `exconv sound2image` - one image sculpted by one audio file
- Purpose: apply the spectrum of an audio file as a radial filter over an image.
- Example:
  ```bash
  exconv sound2image --img img.png --audio audio.wav --out sculpted.png --colorspace luma
  ```

  | Option | Default | What it does |
  | --- | --- | --- |
  | `--img` | required | Input image. |
  | `--audio` | required | Input audio file. |
  | `--out` | required | Output image. |
  | `--mode` | `mono` | Audio mapping: `mono`, `stereo`, `mid-side`. |
  | `--colorspace` | `luma` | Image filtering space: `luma`, `color`. |
  | `--no-normalize` | off | Disable final normalization/clipping. |
  | `--upscale` | `1.0` | Output scale factor; `1.0` disables. |
  | `--upscale-method` | `lanczos` | Upscale method; opencv-* requires a model. |
  | `--upscale-model` | None | Model path for opencv-* upscalers. |

### `exconv animate` - create GIF/MP4 from image sequence
- Purpose: turn a folder of images into a GIF or MP4.
- Example:
  ```bash
  exconv animate samples/output/sound2image/my_project/animations \
    samples/output/sound2image/my_project/preview.mp4 --fps 12
  ```

  | Option | Default | What it does |
  | --- | --- | --- |
  | `input_dir` (positional) | required | Directory containing input images. |
  | `output_path` (positional) | required | Output `.gif` or video path. |
  | `--format` | inferred | `gif` or `mp4`. |
  | `--pattern` | None | Optional glob filter such as `*.png`. |
  | `--recursive` | off | Search for frames recursively. |
  | `--fps` | `10.0` | Frames per second; ignored if `--duration` is set. |
  | `--duration` | None | Seconds per frame; overrides `--fps`. |
  | `--loop` | `0` | GIF loop count; `0` means infinite. |
  | `--audio` | None | Audio file to mux into MP4 output; ffmpeg required. |

## Single-run helpers (scripts)

- `video_biconv.py`: legacy wrapper around `exconv video-biconv` for a single
  video/audio pair (the CLI command is the canonical entry point).
- `split_audio_golden.py`: split one audio file into N golden-ratio parts and
  write each part to its own file (use before `exconv folderbatch`).
- `image2sound_demo.py`: derive an impulse from an image (`flat`, `hist`,
  `radial` modes) and convolve an audio file; writes WAV plus an impulse plot.
- `sound2image_demo.py`: minimal sound->image example; `exconv sound2image` is
  the supported CLI equivalent.
- `sweep_conv2d_options.py`: grid-sweep image auto/pair convolution settings;
  writes JPEGs and a metrics CSV under `samples/output/img_sweep/`.
- `animator.py`: legacy wrapper; use `exconv animate` instead.
- `test_video.py`: short scratchpad calling `sound2image_video_from_files`
  on bundled sample inputs.
