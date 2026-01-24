# Ideas To Test

Notes for future experiments and pipeline tweaks.

- Pre-upscale vs post-upscale: upscale the input frames before convolution and compare to the current post-processing upscale.
- Scale sensitivity: test x1.5, x2, x3, x4 to see when convolution artifacts become model-unfriendly.
- Method sensitivity: compare classic resampling (lanczos/bicubic) with ML backends (opencv-edsr/fsrcnn/espcn/lapsrn).
- Hybrid pipeline: mild denoise or blur before convolution to reduce ringing, then upscale.
- Downscale + re-upscale: downscale before convolution (anti-alias) then upscale; check if it stabilizes structure.
- Colorspace check: luma-only processing vs full color, then upscale; look for chroma artifacts.
- Video blocks: upscale per-frame vs upscale per-block mean image before reuse; check temporal consistency.
- instead of beat/novelty/... analysis from music - analize movement in image
    - do processing for image and sound in blocks derived from the beats of inequal length - for image use the beats of sound - for sound use the beats of video or however user choses

