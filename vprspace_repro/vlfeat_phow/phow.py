"""MATLAB-equivalent vl_phow (VLFeat 0.9.21) through ctypes: same smoothing (vl_imsmooth MEX), same dense SIFT
(vl_dsift MEX, 'fast', 'norm', window 1.5, uint8 descriptors), same contrast thresholding, same frame alignment.

    frames, descrs = phow(im)   # im: 2-D float32 array in [0, 1] (MATLAB single), rows x cols
    frames: (n, 4) float64 [x, y, contrast, binSize] (1-based, MATLAB convention); descrs: (n, 128) uint8

The shared library libphow.so (phow_shim.c) and VLFeat's libvl.so must sit in VPRS_VLFEAT_DIR (default: this dir).
"""
import ctypes, math, os

import numpy as np

def _find_lib():
    import glob
    for d in (os.environ.get("VPRS_VLFEAT_DIR"), os.path.dirname(os.path.abspath(__file__))):
        if d and os.path.exists(os.path.join(d, "libphow.so")):
            return os.path.join(d, "libphow.so")
    hits = sorted(glob.glob("/kaggle/input/**/libphow.so", recursive=True))  # Kaggle: build kernel attached as input
    assert hits, "libphow.so not found (set VPRS_VLFEAT_DIR)"
    return hits[0]


_lib = ctypes.CDLL(_find_lib())
_P = ctypes.c_void_p
_lib.mex_imsmooth.argtypes = [_P, _P, ctypes.c_int, ctypes.c_int, ctypes.c_double]
_lib.mex_imsmooth.restype = ctypes.c_int
_lib.mex_dsift.argtypes = [_P, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                           ctypes.c_int, ctypes.c_int, _P, _P]
_lib.mex_dsift.restype = ctypes.c_int


def imsmooth(im, sigma):
    a = np.asfortranarray(im, dtype=np.float32)
    out = np.empty(a.shape, dtype=np.float32, order="F")
    M, N = a.shape
    assert _lib.mex_imsmooth(out.ctypes.data, a.ctypes.data, M, N, float(sigma)) == 0
    return out


def dsift(im, bin_size, step, off, window_size=1.5, fast=True):
    a = np.asfortranarray(im, dtype=np.float32)
    M, N = a.shape
    cap = (M // step + 1) * (N // step + 1)
    frames = np.empty((cap, 3), dtype=np.float64)
    descr = np.empty((cap, 128), dtype=np.uint8)
    n = _lib.mex_dsift(a.ctypes.data, M, N, int(bin_size), int(step), float(off), float(window_size), int(fast), cap,
                       frames.ctypes.data, descr.ctypes.data)
    assert n >= 0, ("frame buffer too small", n, cap)
    return frames[:n], descr[:n]


def phow(im, sizes=(4, 6, 8, 10), step=2, magnif=6, window_size=1.5, contrast_threshold=0.005, fast=True):
    """vl_phow(im) for a gray single image; defaults are vl_phow.m's."""
    assert im.ndim == 2 and im.dtype == np.float32
    F, D = [], []
    for s in sizes:
        off = math.floor(1 + 3 / 2 * (max(sizes) - s))
        ims = imsmooth(im, s / magnif)
        f, d = dsift(ims, s, step, off, window_size, fast)
        d[f[:, 2] < contrast_threshold] = 0
        F.append(np.concatenate([f, np.full((len(f), 1), float(s))], 1))
        D.append(d)
    return np.concatenate(F), np.concatenate(D)
