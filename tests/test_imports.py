import exconv
from exconv import io, conv1d, conv2d
from exconv.core import fftnd, linear_freq_multiply, freq_grid_2d, rms_normalize

print("exconv version:", exconv.__version__)
print("core fftnd:", fftnd)
print("1D Audio class:", conv1d.Audio)
print("2D image auto conv:", conv2d.image_auto_convolve)
print("IO read_audio:", io.read_audio)

print("All imports OK ✅")
