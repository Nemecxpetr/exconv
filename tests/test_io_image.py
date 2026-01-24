from pathlib import Path

import numpy as np

from exconv.io.image import (
    read_image,
    write_image,
    as_float32,
    as_uint8,
    rgb_to_luma,
    luma_to_rgb,
    upscale_image,
)


def test_dtype_conversions():
    # uint8 → float32 → uint8 roundtrip
    arr_u8 = np.array([[0, 128, 255]], dtype=np.uint8)
    arr_f = as_float32(arr_u8)
    assert arr_f.dtype == np.float32
    assert arr_f.min() >= 0.0
    assert arr_f.max() <= 1.0 + 1e-6

    arr_u8_rt = as_uint8(arr_f)
    assert arr_u8_rt.dtype == np.uint8
    # allow 1-step quantization differences
    np.testing.assert_allclose(arr_u8_rt, arr_u8, atol=1)


def test_rgb_luma_conversions():
    # Construct a small RGB image
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    luma = rgb_to_luma(rgb)
    assert luma.shape == (2, 2)
    assert luma.dtype == float

    rgb_gray = luma_to_rgb(luma, dtype=np.float32)
    assert rgb_gray.shape == (2, 2, 3)
    assert rgb_gray.dtype == np.float32
    # all channels in grayscale must be equal
    np.testing.assert_allclose(rgb_gray[..., 0], rgb_gray[..., 1])
    np.testing.assert_allclose(rgb_gray[..., 0], rgb_gray[..., 2])


def test_image_roundtrip_small(tmp_path: Path):
    # Small synthetic gradient image
    h, w = 8, 8
    x = np.linspace(0.0, 1.0, h * w, dtype=np.float32).reshape(h, w)
    rgb = np.stack([x, x**2, x[::-1]], axis=-1)  # (H,W,3) float32

    out_path = tmp_path / "test.png"
    write_image(out_path, rgb)

    y = read_image(out_path, mode="RGB", dtype="uint8")
    assert y.shape == (h, w, 3)
    assert y.dtype == np.uint8

    # Basic sanity: values are monotonic in at least one channel
    # (because we wrote a gradient)
    assert y[..., 0].min() <= y[..., 0].max()


def test_upscale_nearest_grayscale():
    img = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    out = upscale_image(img, scale=2.0, method="nearest")
    expected = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)
    assert out.shape == (4, 4)
    np.testing.assert_array_equal(out, expected)


def test_upscale_bicubic_rgb_shape():
    img = np.zeros((4, 6, 3), dtype=np.uint8)
    out = upscale_image(img, scale=1.5, method="bicubic")
    assert out.shape == (6, 9, 3)
