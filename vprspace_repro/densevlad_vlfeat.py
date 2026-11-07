"""DenseVLAD (Torii et al., CVPR 2015) with the ORIGINAL VLFeat dense SIFT, replacing the kornia approximation of
densevlad_gpu.py. Every step follows the MATLAB pipeline of the DenseVLAD paper:

  * gray image as MATLAB's rgb2gray (uint8, rounded), im2single -> [0, 1]
  * optional downscaling ("640": long side to 640 px) with a port of MATLAB's imresize (bicubic, antialiasing)
  * vl_phow defaults, through VLFeat 0.9.21 itself (vlfeat_phow/phow.py + libphow.so): sizes 4/6/8/10, step 2,
    Gaussian pre-smoothing sigma = size/6, 'fast' flat window, window size 1.5, uint8 descriptors, and
    descriptors with contrast (norm) < 0.005 mapped to zero
  * RootSIFT (L1 normalisation, square root; all-zero descriptors stay zero)
  * K = 128 visual words: Torii's released centres if a .mat is given (--vocab-mat), else k-means on RootSIFT
    descriptors sampled from DATABASE images only
  * hard-assignment VLAD, intra-normalisation, global L2 -> 16384-D. Two treatments of the zeroed (low contrast)
    descriptors are written: "keep0" (they are aggregated like any descriptor, as vl_phow's output would be if fed
    unchanged to VLAD) and "drop0" (they are skipped)
  * if Torii's PCA projection is given (--pca-mat): PCA-whitening to 4096-D with HIS projection (never learned on
    the benchmark: a whitening learned from 3,989 database VLADs in 16,384-D is rank deficient)

python densevlad_vlfeat.py --root <vpr-space root> --desc-dir <out desc dir> --res 640 native [--vocab-mat f.mat]
"""
import argparse, json, math, os, sys, time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
from scipy import sparse
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = os.environ.get("VPRS_LOAD_TRUNCATED", "1") == "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vlfeat_phow"))
K, LONG_SIDE = 128, 640


def log(*a):
    print(time.strftime("%H:%M:%S"), "[densevlad-vlfeat]", *a, flush=True)


# ------------------------------------------------------------------ image I/O, MATLAB conventions
def rgb2gray_uint8(rgb):
    """MATLAB rgb2gray for uint8 input: weights of the NTSC luminance, result rounded to uint8."""
    x = rgb.astype(np.float64)
    g = 0.298936021293775 * x[..., 0] + 0.587043074451121 * x[..., 1] + 0.114020904255103 * x[..., 2]
    return np.clip(np.floor(g + 0.5), 0, 255).astype(np.uint8)


def _cubic(x):
    ax = np.abs(x)
    ax2, ax3 = ax * ax, ax * ax * ax
    return ((1.5 * ax3 - 2.5 * ax2 + 1) * (ax <= 1) +
            (-0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2) * ((ax > 1) & (ax <= 2)))


def _contributions(in_len, out_len, scale):
    """Weights/indices of MATLAB imresize along one dimension (bicubic, antialiasing when scale < 1)."""
    kernel_width = 4.0
    if scale < 1:
        h = lambda x: scale * _cubic(scale * x)
        kernel_width = kernel_width / scale
    else:
        h = _cubic
    x = np.arange(1, out_len + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(math.ceil(kernel_width)) + 2
    ind = left[:, None] + np.arange(P)[None, :]
    w = h(u[:, None] - ind)
    w = w / w.sum(1, keepdims=True)
    aux = np.concatenate([np.arange(1, in_len + 1), np.arange(in_len, 0, -1)])
    ind = aux[np.mod(ind.astype(np.int64) - 1, aux.size)] - 1
    keep = np.any(w != 0, axis=0)
    return w[:, keep], ind[:, keep]


def imresize_matlab(im, scale):
    """MATLAB imresize(im, scale) for a 2-D single image (default bicubic, antialiasing)."""
    out_shape = [int(math.ceil(scale * s)) for s in im.shape]
    out = im.astype(np.float64)
    for dim in (0, 1):  # equal scales: MATLAB processes dimension 1 (rows) first
        w, ind = _contributions(im.shape[dim], out_shape[dim], scale)
        if dim == 0:
            out = np.einsum("op,opc->oc", w, out[ind])
        else:
            out = np.einsum("op,rop->ro", w, out[:, ind])
    return out.astype(np.float32)


def load_image(path, res):
    im = Image.open(path)
    im.load()
    a = np.asarray(im) if im.mode == "L" else np.asarray(im.convert("RGB"))
    assert a.dtype == np.uint8, (path, im.mode, a.dtype)  # 8-bit release images only (rgb2gray/im2single below)
    g = rgb2gray_uint8(a) if a.ndim == 3 else a
    x = g.astype(np.float32) / 255.0  # im2single
    if res == "640" and max(x.shape) > LONG_SIDE:
        x = imresize_matlab(x, LONG_SIDE / max(x.shape))
    return np.ascontiguousarray(x, dtype=np.float32)


def rootsift(d_uint8):
    d = d_uint8.astype(np.float32)
    s = d.sum(1, keepdims=True)
    out = np.zeros_like(d)
    nz = s[:, 0] > 0
    out[nz] = np.sqrt(d[nz] / s[nz])
    return out, nz


# ------------------------------------------------------------------ workers
_C = None


def _init(centers):
    global _C
    _C = None if centers is None else (centers.astype(np.float32), (centers.astype(np.float32) ** 2).sum(1))


def _phow_scales(x):
    from phow import imsmooth, dsift
    sizes = (4, 6, 8, 10)
    for s in sizes:
        off = math.floor(1 + 3 / 2 * (max(sizes) - s))
        f, d = dsift(imsmooth(x, s / 6.0), s, 2, off, 1.5, True)
        d[f[:, 2] < 0.005] = 0
        yield d


def _sample(args):
    path, res, n, seed = args
    rng = np.random.default_rng(seed)
    D = []
    for d in _phow_scales(load_image(path, res)):
        r, nz = rootsift(d)
        D.append(r[nz])
    D = np.concatenate(D)
    return D[rng.choice(len(D), size=min(n, len(D)), replace=False)] if len(D) else np.zeros((0, 128), np.float32)


def _encode(args):
    path, res = args
    C, C2 = _C
    V = {z: np.zeros((K, 128), np.float64) for z in ("keep0", "drop0")}
    n_tot = n_zero = 0
    for d in _phow_scales(load_image(path, res)):
        R, nz = rootsift(d)
        for a0 in range(0, len(R), 200000):
            r, m = R[a0:a0 + 200000], nz[a0:a0 + 200000]
            a = np.argmax(2.0 * (r @ C.T) - C2[None, :], axis=1)  # nearest centre (squared L2)
            r64 = r.astype(np.float64)
            for z, sel in (("keep0", slice(None)), ("drop0", m)):
                aa, rr = a[sel], r64[sel]
                onehot = sparse.csr_matrix((np.ones(len(aa)), (aa, np.arange(len(aa)))), shape=(K, len(aa)))
                V[z] += onehot @ rr - np.bincount(aa, minlength=K)[:, None] * C.astype(np.float64)  # sum of residuals
            n_tot += len(r)
            n_zero += int((~m).sum())
    out = {}
    for z, v in V.items():
        nrm = np.linalg.norm(v, axis=1, keepdims=True)
        v = np.where(nrm > 0, v / np.maximum(nrm, 1e-300), 0.0)  # intra-normalisation
        v = v.reshape(-1)
        out[z] = (v / max(np.linalg.norm(v), 1e-300)).astype(np.float32)
    return out, n_tot, n_zero


def kmeans(X, k, iters=50, seed=0):
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.as_tensor(X, device=dev)
        g = torch.Generator(device="cpu").manual_seed(seed)
        c = x[torch.randperm(len(x), generator=g)[:k].to(dev)].clone()
        for _ in range(iters):
            a = torch.cat([torch.cdist(x[i:i + 262144], c).argmin(1) for i in range(0, len(x), 262144)])
            cnt = torch.bincount(a, minlength=k).float()
            s = torch.zeros_like(c).index_add_(0, a, x)
            empty = cnt == 0
            c = torch.where(empty[:, None], x[torch.randint(len(x), (k,), generator=g).to(dev)], s / cnt.clamp(min=1)[:, None])
        return c.cpu().numpy()
    except ImportError:
        rng = np.random.default_rng(seed)
        c = X[rng.choice(len(X), k, replace=False)].copy()
        for _ in range(iters):
            a = np.argmax(2 * X @ c.T - (c ** 2).sum(1)[None], 1)
            for j in range(k):
                m = a == j
                c[j] = X[m].mean(0) if m.any() else X[rng.integers(len(X))]
        return c


def load_mat_centers(path):
    """Torii's vocabulary file: return the 128 x 128 centre matrix (one row per centre)."""
    try:
        from scipy.io import loadmat
        m = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
    except NotImplementedError:  # v7.3 (HDF5)
        import h5py
        f = h5py.File(path, "r")
        m = {k: np.array(f[k]).T for k in f.keys()}
    cands = {k: np.asarray(v, dtype=np.float64) for k, v in m.items() if hasattr(v, "shape") and sorted(np.shape(v)) == [128, 128]}
    log("vocab mat variables:", {k: getattr(v, "shape", None) for k, v in m.items()})
    assert len(cands) == 1, cands.keys()
    return next(iter(cands.values())).T.astype(np.float32)  # MATLAB/VLFeat store centres as columns (D x K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--desc-dir", required=True)
    ap.add_argument("--res", nargs="+", default=["640", "native"], choices=["640", "native"])
    ap.add_argument("--vocab-mat")
    ap.add_argument("--samples-per-image", type=int, default=300)
    ap.add_argument("--trajectories", nargs="*")
    ap.add_argument("--procs", type=int, default=max(1, len(os.sched_getaffinity(0)) - 2))
    a = ap.parse_args()
    root, desc = Path(a.root), Path(a.desc_dir)
    sys.path.insert(0, str(HERE / "upstream"))
    import dataset
    pairs = [dataset.SETUPS[i] for i in dataset.ids()]
    db_trajs = sorted({db for db, _ in pairs})
    trajs = sorted({t for p in pairs for t in p})
    if a.trajectories:
        trajs = [t for t in trajs if t in a.trajectories]
    ctx = get_context("fork")
    for res in a.res:
        t0 = time.time()
        tag = f"densevlad_vlfeat_{res}"
        vocab_f = desc / "_densevlad_vlfeat" / f"vocab_{res}.npy"
        vocab_f.parent.mkdir(parents=True, exist_ok=True)
        if a.vocab_mat:
            C = load_mat_centers(a.vocab_mat)
            tag += "_torii"
        elif vocab_f.exists():
            C = np.load(vocab_f)
        else:
            paths = [str(p) for t in db_trajs for p in sorted((root / "images" / t).glob("*.png"))]
            with ctx.Pool(a.procs, initializer=_init, initargs=(None,)) as pool:
                S = list(pool.imap(_sample, [(p, res, a.samples_per_image, i) for i, p in enumerate(paths)], chunksize=4))
            X = np.concatenate(S).astype(np.float32)
            log(f"{res}: {len(paths)} database images, {len(X)} sampled non-zero RootSIFT ({time.time() - t0:.0f}s)")
            C = kmeans(X, K)
            np.save(vocab_f, C)
        log(f"{res}: vocabulary {C.shape} ready ({time.time() - t0:.0f}s)")
        stats = {}
        with ctx.Pool(a.procs, initializer=_init, initargs=(C,)) as pool:
            for t in trajs:
                files = sorted((root / "images" / t).glob("*.png"))
                outs = list(pool.imap(_encode, [(str(p), res) for p in files], chunksize=2))
                for z in ("keep0", "drop0"):
                    d = desc / f"{tag}_{z}"
                    d.mkdir(parents=True, exist_ok=True)
                    X = np.stack([o[0][z] for o in outs])
                    assert X.shape == (len(files), K * 128) and np.isfinite(X).all()
                    np.save(d / f"{t}.npy", X)
                n_tot, n_zero = sum(o[1] for o in outs), sum(o[2] for o in outs)
                stats[t] = {"images": len(files), "descriptors_per_image": n_tot / len(files), "zero_fraction": n_zero / max(n_tot, 1)}
                log(f"{res} {t}: {len(files)} imgs, {n_tot / len(files):.0f} descr/img, zero {n_zero / max(n_tot, 1):.3f} "
                    f"({time.time() - t0:.0f}s)")
        json.dump(stats, open(desc / "_densevlad_vlfeat" / f"stats_{res}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
