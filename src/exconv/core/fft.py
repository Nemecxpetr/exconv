# src/exconv/core/fft.py
"""FFT utilities and frequency-domain convolution."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Optional, Union
import numpy as np

try:
    # Prefer SciPy's planner if available
    from scipy.fft import next_fast_len as _scipy_next_fast_len
except Exception:
    _scipy_next_fast_len = None  # type: ignore[attr-defined]

ArrayLike = np.ndarray
AxesLike = Optional[Union[int, Sequence[int]]]

__all__ = [
    "next_fast_len_ge",
    "pad_to_linear_shape",
    "fftnd",
    "ifftnd",
    "linear_freq_multiply",
    "make_hermitian_symmetric_unshifted",
]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def next_fast_len_ge(n: int) -> int:
    """
    Return a fast FFT length >= n.

    Tries scipy.fft.next_fast_len (if available), then numpy.fft.next_fast_len
    (if available). Falls back to next power of two.

    Parameters
    ----------
    n : int
        Minimum length.

    Returns
    -------
    int
        Fast length >= n.
    """
    if n <= 1:
        return 1
    # SciPy (most robust/modern)
    if _scipy_next_fast_len is not None:
        try:
            return int(_scipy_next_fast_len(n))
        except Exception:
            pass
    # NumPy (available in modern NumPy)
    try:
        return int(np.fft.next_fast_len(n))  # type: ignore[attr-defined]
    except Exception:
        # Fallback: next power of two
        m = 1
        while m < n:
            m <<= 1
        return m


def _normalize_axes(x: ArrayLike, axes: AxesLike) -> Tuple[int, ...]:
    if axes is None:
        return tuple(range(x.ndim))
    if isinstance(axes, int):
        axes = (axes,)
    # normalize negatives and remove duplicates preserving order
    norm = []
    for a in axes:
        a = int(a)
        if a < 0:
            a += x.ndim
        if a not in norm:
            norm.append(a)
    return tuple(norm)


def _full_linear_shape(sig_shape: Sequence[int], ker_shape: Sequence[int]) -> Tuple[int, ...]:
    return tuple(s + k - 1 for s, k in zip(sig_shape, ker_shape))


def _same_first_crop(same_ref: Sequence[int]) -> Tuple[slice, ...]:
    return tuple(slice(0, n) for n in same_ref)


def _same_center_crop(full_shape: Sequence[int], same_ref: Sequence[int]) -> Tuple[slice, ...]:
    slices: list[slice] = []
    for L, N in zip(full_shape, same_ref):
        if N >= L:
            # pad later (handled by caller) — but for slicing we give entire axis
            slices.append(slice(0, L))
        else:
            start = (L - N) // 2
            slices.append(slice(start, start + N))
    return tuple(slices)


def pad_to_linear_shape(x: ArrayLike, kernel_shape: Sequence[int], axes: AxesLike = None) -> Tuple[ArrayLike, Tuple[Tuple[int, int], ...]]:
    """
    Zero-pad `x` so that along `axes` it can hold a full linear convolution with
    a kernel of shape `kernel_shape`. Other axes are untouched.

    Parameters
    ----------
    x : ndarray
        Input array.
    kernel_shape : Sequence[int]
        Kernel sizes along `axes` (must match axes length).
    axes : int | sequence[int] | None
        Axes along which convolution happens. Default: all axes.

    Returns
    -------
    x_padded : ndarray
        Zero-padded array.
    pads : tuple[tuple[int,int], ...]
        Per-axis (pad_before, pad_after). Can be passed to `np.pad` on another
        array to ensure identical spatial size.
    """
    axes_t = _normalize_axes(x, axes)
    if len(kernel_shape) != len(axes_t):
        raise ValueError("kernel_shape length must equal number of conv axes")

    pads: list[Tuple[int, int]] = []
    target_shape = list(x.shape)

    for ax, klen in zip(axes_t, kernel_shape):
        out_len = x.shape[ax] + int(klen) - 1
        pads.append((0, out_len - x.shape[ax]))  # pad at the end deterministically
        target_shape[ax] = out_len

    # Build full pads for all axes
    full_pads: list[Tuple[int, int]] = []
    j = 0
    for d in range(x.ndim):
        if d in axes_t:
            full_pads.append(pads[j])
            j += 1
        else:
            full_pads.append((0, 0))

    x_padded = np.pad(x, pad_width=tuple(full_pads), mode="constant", constant_values=0)
    return x_padded, tuple(full_pads)


def _fft_shapes_for_linear(sig_shape: Sequence[int], ker_shape: Sequence[int]) -> Tuple[int, ...]:
    return tuple(next_fast_len_ge(s + k - 1) for s, k in zip(sig_shape, ker_shape))


# ---------------------------------------------------------------------------
# FFT wrappers
# ---------------------------------------------------------------------------

def fftnd(x: ArrayLike, axes: AxesLike = None, real_input: bool = False, n: Optional[Sequence[int]] = None) -> ArrayLike:
    """
    Multi-dimensional FFT wrapper that preserves dtype/complex handling.

    Parameters
    ----------
    x : ndarray
        Input.
    axes : int | sequence[int] | None
        Axes to transform. Default: all axes.
    real_input : bool
        If True, use real-to-complex FFT (rfftn). Otherwise use complex FFT (fftn).
    n : sequence[int] | None
        Optional FFT lengths per axis (in the same order as `axes`).

    Returns
    -------
    ndarray
        Frequency-domain array.
    """
    axes_t = _normalize_axes(x, axes)
    if real_input:
        return np.fft.rfftn(x, s=n, axes=axes_t)
    else:
        return np.fft.fftn(x, s=n, axes=axes_t)


def ifftnd(X: ArrayLike, axes: AxesLike = None, real_output: bool = False, n: Optional[Sequence[int]] = None) -> ArrayLike:
    """
    Multi-dimensional inverse FFT wrapper.

    Parameters
    ----------
    X : ndarray
        Frequency-domain input.
    axes : int | sequence[int] | None
        Axes to transform. Default: all axes.
    real_output : bool
        If True, use irfftn (expects last frequency axis to be rfft-sized).
        Otherwise use ifftn.
    n : sequence[int] | None
        Optional output lengths per axis (in the same order as `axes`).

    Returns
    -------
    ndarray
        Time/space-domain array. For `real_output=True` the dtype will be real.
    """
    axes_t = _normalize_axes(X, axes)
    if real_output:
        return np.fft.irfftn(X, s=n, axes=axes_t)
    else:
        return np.fft.ifftn(X, s=n, axes=axes_t)


# ---------------------------------------------------------------------------
# Frequency-domain linear/circular convolution
# ---------------------------------------------------------------------------

def _crop_mode(y_full: ArrayLike, mode: str, ref_sig_shape: Sequence[int], axes: Sequence[int]) -> ArrayLike:
    """
    Crop (or pad) full linear result to the requested mode.
    """
    if mode == "full":
        return y_full

    out = y_full
    # Prepare per-axis slicing
    linear_shape = list(y_full.shape)
    same_ref = list(ref_sig_shape)

    if mode == "same-first":
        slices = [slice(None)] * out.ndim
        for ax, N in zip(axes, same_ref):
            slices[ax] = slice(0, N)
        return out[tuple(slices)]

    if mode == "same-center":
        slices = [slice(None)] * out.ndim
        for ax, (L, N) in zip(axes, zip((linear_shape[a] for a in axes), same_ref)):
            start = max(0, (L - N) // 2)
            slices[ax] = slice(start, start + N)
        centered = out[tuple(slices)]
        # If N>L (rare), pad to center (deterministic zero pad)
        need_pad = []
        for ax, (L, N) in zip(axes, zip((centered.shape[a] for a in axes), same_ref)):
            if N > L:
                need_pad.append((ax, N - L))
        if need_pad:
            pads = [(0, 0)] * out.ndim
            for ax, d in need_pad:
                pads[ax] = ((d // 2), (d - d // 2))
            centered = np.pad(centered, pads, mode="constant")
        return centered

    raise ValueError(f"Unknown mode {mode!r}")


def linear_freq_multiply(
    x: ArrayLike,
    h: ArrayLike,
    axes: AxesLike = None,
    mode: str = "full",
    use_real_fft: bool = True,
) -> ArrayLike:
    """
    Compute convolution via frequency multiplication along `axes`.

    Pipeline: (optional zero-pad) → FFT → pointwise multiply → IFFT → crop.

    Parameters
    ----------
    x : ndarray
        Signal array.
    h : ndarray
        Kernel array (broadcastable to `x` outside `axes`).
    axes : int | sequence[int] | None
        Axes along which convolution is performed. Default: all axes.
    mode : {"full","same-first","same-center","circular"}
        Output size policy. For "circular" no padding is used and the length
        along each conv axis equals `x.shape[axis]`.
    use_real_fft : bool
        If True and both inputs are real-valued, use rfftn/irfftn.

    Returns
    -------
    ndarray
        Convolved output per `mode`.
    """
    axes_t = _normalize_axes(x, axes)
    if x.ndim != h.ndim:
        # Broadcast where possible by explicitly expanding singleton dims of h
        h = np.broadcast_to(h, x.shape)

    x_is_real = np.isrealobj(x)
    h_is_real = np.isrealobj(h)
    can_use_real = use_real_fft and x_is_real and h_is_real

    if mode == "circular":
        # Match signal length along axes; compute circular conv
        n = [x.shape[a] for a in axes_t]
        X = fftnd(x, axes=axes_t, real_input=False, n=n)
        H = fftnd(h, axes=axes_t, real_input=False, n=n)
        Y = X * H
        y = ifftnd(Y, axes=axes_t, real_output=False, n=n)
        # For real-input circular conv the output should be real (imag ~ 0)
        return y.real if x_is_real and h_is_real else y

    # Linear (full) size per axis
    sig_shape = tuple(x.shape[a] for a in axes_t)
    ker_shape = tuple(h.shape[a] for a in axes_t)
    fft_shape = _fft_shapes_for_linear(sig_shape, ker_shape)

    if can_use_real:
        X = fftnd(x, axes=axes_t, real_input=True, n=fft_shape)
        H = fftnd(h, axes=axes_t, real_input=True, n=fft_shape)
        Y = X * H
        # Real output length along axes is linear size (s+k-1)
        full_linear = tuple(s + k - 1 for s, k in zip(sig_shape, ker_shape))
        y_full = ifftnd(Y, axes=axes_t, real_output=True, n=full_linear)
    else:
        # Complex FFT path
        X = fftnd(x, axes=axes_t, real_input=False, n=fft_shape)
        H = fftnd(h, axes=axes_t, real_input=False, n=fft_shape)
        Y = X * H
        full_linear = tuple(s + k - 1 for s, k in zip(sig_shape, ker_shape))
        y_tmp = ifftnd(Y, axes=axes_t, real_output=False, n=fft_shape)
        # Trim to the true full linear shape
        slices = [slice(None)] * y_tmp.ndim
        for ax, L in zip(axes_t, full_linear):
            slices[ax] = slice(0, L)
        y_full = y_tmp[tuple(slices)]

    if mode == "full":
        return y_full.real if x_is_real and h_is_real else y_full

    if mode in ("same-first", "same-center"):
        cropped = _crop_mode(y_full, mode, ref_sig_shape=[x.shape[a] for a in axes_t], axes=axes_t)
        return cropped.real if x_is_real and h_is_real else cropped

    raise ValueError(f"Unknown mode {mode!r}")


# ---------------------------------------------------------------------------
# Hermitian symmetry helpers
# ---------------------------------------------------------------------------
def make_hermitian_symmetric_unshifted(F: ArrayLike, axes: AxesLike = None) -> ArrayLike:
    """
    Enforce Hermitian symmetry on an *unshifted* complex spectrum F so that
    np.fft.ifftn(F, axes=axes) is (numerically) real.

    For each axis in `axes`, we enforce:
        X[k] = conj(X[(-k) mod N]),   k = 0..N-1

    In index space 0..N-1 this is the mapping:
        0       -> 0         (DC)
        1..N-1  -> N-1..1    (i -> (N - i) % N)

    This is implemented by building a "mod-neg" indexer for each axis and
    averaging with its conjugate pair.

    Parameters
    ----------
    F : ndarray (complex)
        Unshifted spectrum (as returned by np.fft.fftn; do NOT fftshift).
    axes : int | sequence[int] | None
        Frequency axes. Default: all axes.

    Returns
    -------
    ndarray
        Spectrum with Hermitian symmetry enforced.
    """
    X = np.asarray(F)
    if not np.iscomplexobj(X):
        X = X.astype(np.complex128, copy=False)
    elif X.dtype.kind != "c":
        X = X.astype(np.complex128)

    axes_t = _normalize_axes(X, axes)
    if not axes_t:
        return X

    # Build mod-neg mapping along all axes at once:
    X_mod = X
    for ax in axes_t:
        N = X_mod.shape[ax]
        if N <= 0:
            raise ValueError("Empty axis for Hermitian symmetry.")
        idx = np.empty(N, dtype=int)
        idx[0] = 0
        if N > 1:
            idx[1:] = np.arange(N - 1, 0, -1)  # [N-1, N-2, ..., 1]
        X_mod = np.take(X_mod, idx, axis=ax)

    # Average with conjugate of the mod-neg version:
    Xh = 0.5 * (X + np.conj(X_mod))

    # Make all combinations of DC / Nyquist indices along axes strictly real
    special_lists = []
    for ax in axes_t:
        n = Xh.shape[ax]
        lst = [0]
        if n % 2 == 0 and n > 0:
            lst.append(n // 2)
        special_lists.append(lst)

    def _set_real(level: int, picks: list[int]):
        if level == len(axes_t):
            sel = [slice(None)] * Xh.ndim
            for a, v in zip(axes_t, picks):
                sel[a] = v
            sel_t = tuple(sel)
            Xh[sel_t] = np.real(Xh[sel_t])
            return
        for v in special_lists[level]:
            _set_real(level + 1, picks + [v])

    if axes_t:
        _set_real(0, [])

    return Xh

    """
    Enforce Hermitian symmetry on an *unshifted* complex spectrum F so that:
      (1) ifftn(F, axes) is (numerically) real, and
      (2) F == conj(np.flip(F, axis=axes)) holds (as in the tests).

    We average over the finite group orbit generated by:
      A0(X) = conj(flip(X, axis=axes))
      B0(X) = conj(modneg(X, axis=axes))  # modneg: i -> (-i) mod N per axis

    Averaging over the orbit yields a fixed point for both A0 and B0.
    Finally, DC and (if even) Nyquist bins are forced strictly real.

    Parameters
    ----------
    F : ndarray (complex)
        Unshifted spectrum (as returned by np.fft.fftn).
    axes : int | sequence[int] | None
        Frequency axes. Default: all axes.

    Returns
    -------
    ndarray
        Symmetrized spectrum.
    """
    X0 = np.asarray(F)
    if not np.iscomplexobj(X0):
        X0 = X0.astype(np.complex128, copy=False)
    elif X0.dtype.kind != "c":
        X0 = X0.astype(np.complex128)

    axes_t = _normalize_axes(X0, axes)
    if not axes_t:
        return X0

    # Build modneg indexer once for each axis: [0, N-1, N-2, ..., 1]
    def _apply_modneg(A: np.ndarray) -> np.ndarray:
        B = A
        for ax in axes_t:
            N = B.shape[ax]
            idx = np.empty(N, dtype=int)
            idx[0] = 0
            if N > 1:
                idx[1:] = np.arange(N - 1, 0, -1)
            B = np.take(B, idx, axis=ax)
        return B

    def A0(A: np.ndarray) -> np.ndarray:
        return np.conj(np.flip(A, axis=axes_t))

    def B0(A: np.ndarray) -> np.ndarray:
        return np.conj(_apply_modneg(A))

    # Generate the orbit under {A0, B0}. Two involutions can generate up to 8 elements.
    orbit = []
    def _add(E):
        orbit.append(E)

    X = X0
    _add(X)
    X_A = A0(X); _add(X_A)
    X_B = B0(X); _add(X_B)
    X_AB = A0(X_B); _add(X_AB)
    X_BA = B0(X_A); _add(X_BA)
    X_ABA = A0(X_BA); _add(X_ABA)
    X_BAB = B0(X_AB); _add(X_BAB)
    X_ABAB = A0(X_BAB); _add(X_ABAB)

    Xavg = sum(orbit) / len(orbit)

    # Force DC/Nyquist bins strictly real across all combinations of special indices.
    special_lists = []
    for ax in axes_t:
        n = Xavg.shape[ax]
        lst = [0]
        if n % 2 == 0 and n > 0:
            lst.append(n // 2)
        special_lists.append(lst)

    def _set_real(level: int, picks: list[int]):
        if level == len(axes_t):
            sel = [slice(None)] * Xavg.ndim
            for a, v in zip(axes_t, picks):
                sel[a] = v
            sel = tuple(sel)
            Xavg[sel] = np.real(Xavg[sel])
            return
        for v in special_lists[level]:
            _set_real(level + 1, picks + [v])

    _set_real(0, [])

    return Xavg
