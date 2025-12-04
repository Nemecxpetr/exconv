from exconv.xmodal.video import (
    sound2image_video_from_files,
)

frames_out, audio_out, fps_used, sr = sound2image_video_from_files(
    "samples/input/video/IMG_0257.MP4",   # pick any small video you have E:\programming\exconv\samples\input\video\IMG_0257.MP4
    "samples/input/audio/original.wav",
    mode="mono",
    colorspace="color",
    audio_out_mode="per-buffer-auto",
    out_video="samples/output/video/test_processed.mp4",
    out_audio="samples/output/audio/test_processed.wav",
)