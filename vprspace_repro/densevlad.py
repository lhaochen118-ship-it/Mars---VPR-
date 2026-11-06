"""DenseVLAD re-implementation (Torii et al., "24/7 place recognition by view synthesis", CVPR 2015).

No official Python implementation / vocabulary exists, and the VPR-Space authors did not release theirs, so this is
a documented re-implementation following the original paper:
  * images converted to grayscale and downscaled so that the long side is 640 px (keeps aspect ratio),
  * SIFT descriptors densely sampled on a regular grid with a 2 px step at 4 scales (bin sizes 4, 6, 8, 10 px, i.e.
    keypoint diameter = 4 * bin size), RootSIFT normalisation,
  * vocabulary of 128 visual words (k-means) learned from database images only (never from queries),
  * VLAD (hard assignment), intra-normalisation, global L2 normalisation (16384-D),
  * optional PCA-whitening to 4096-D learned on the database VLADs of all setups pooled.
Run modes:
  extract : per trajectory dense RootSIFT -> sampled descriptors for the vocabulary + per-image descriptor files
  vocab   : k-means on the pooled database samples
  vlad    : VLAD per image for each trajectory  -> desc/densevlad_16384/<traj>.npy (+ PCA variant)
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

STEP, BIN_SIZES, LONG_SIDE, K = 2, (4, 6, 8, 10), 640, 128


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    s = LONG_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img


def dense_rootsift(img):
    sift = cv2.SIFT_create()
    h, w = img.shape
    out = []
    for b in BIN_SIZES:
        size = 4 * b  # SIFT descriptor covers 4x4 bins
        m = size // 2
        kps = [cv2.KeyPoint(float(x), float(y), float(size)) for y in range(m, h - m, STEP) for x in range(m, w - m, STEP)]
        if not kps:
            continue
        _, d = sift.compute(img, kps)
        if d is not None:
            out.append(d)
    d = np.concatenate(out).astype(np.float32)
    d /= np.maximum(d.sum(1, keepdims=True), 1e-12)  # L1
    return np.sqrt(d)  # RootSIFT (L2-normalised by construction)


def _sample_worker(args):
    path, n, seed = args
    d = dense_rootsift(load_gray(path))
    rng = np.random.default_rng(seed)
    return d[rng.choice(len(d), size=min(n, len(d)), replace=False)]


def vlad(desc, centers):
    # hard assignment by nearest center (squared L2), residual aggregation, intra-norm + global L2
    d2 = (desc ** 2).sum(1, keepdims=True) - 2 * desc @ centers.T + (centers ** 2).sum(1)[None]
    a = d2.argmin(1)
    v = np.zeros_like(centers)
    np.add.at(v, a, desc - centers[a])
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    v = v.ravel()
    return v / max(np.linalg.norm(v), 1e-12)


def _vlad_worker(args):
    path, centers = args
    return vlad(dense_rootsift(load_gray(path)), centers)


def kmeans(x, k, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)].copy()
    for it in range(iters):
        d2 = (x ** 2).sum(1, keepdims=True) - 2 * x @ c.T + (c ** 2).sum(1)[None]
        a = d2.argmin(1)
        for j in range(k):
            m = a == j
            c[j] = x[m].mean(0) if m.any() else x[rng.integers(len(x))]
    return c.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["vocab", "vlad", "pca", "all"], default="all")
    ap.add_argument("--trajectories", nargs="*")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--samples-per-image", type=int, default=300)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "upstream"))
    import dataset
    setups = {i: dataset.SETUPS[i] for i in dataset.ids()}
    db_trajs = sorted({db for db, _ in setups.values()})
    all_trajs = sorted({t for pair in setups.values() for t in pair})
    vocab_f = out / "densevlad_vocab_k128.npy"
    if a.stage in ("vocab", "all") and not vocab_f.exists():
        jobs = []
        for t in db_trajs:  # vocabulary from database images only
            for i, p in enumerate(sorted((root / "images" / t).glob("*.png"))):
                jobs.append((p, a.samples_per_image, hash((t, i)) % (2**31)))
        log(f"vocab: dense RootSIFT on {len(jobs)} database images with {a.workers} workers")
        with ProcessPoolExecutor(a.workers) as ex:
            samples = np.concatenate(list(ex.map(_sample_worker, jobs, chunksize=8)))
        log(f"vocab: k-means k={K} on {len(samples)} descriptors")
        np.save(vocab_f, kmeans(samples, K))
    centers = np.load(vocab_f)
    if a.stage in ("vlad", "all"):
        dd = out / "densevlad_16384"
        dd.mkdir(exist_ok=True)
        for t in (a.trajectories or all_trajs):
            f = dd / f"{t}.npy"
            if f.exists():
                continue
            paths = sorted((root / "images" / t).glob("*.png"))
            t0 = time.time()
            with ProcessPoolExecutor(a.workers) as ex:
                V = np.stack(list(ex.map(_vlad_worker, [(p, centers) for p in paths], chunksize=4)))
            np.save(f, V.astype(np.float32))
            log(f"vlad {t}: {len(paths)} imgs in {time.time() - t0:.0f}s")
    if a.stage in ("pca", "all"):
        dd, dp = out / "densevlad_16384", out / "densevlad_pca4096"
        if all((dd / f"{t}.npy").exists() for t in all_trajs):
            X = np.concatenate([np.load(dd / f"{t}.npy") for t in db_trajs]).astype(np.float64)
            mu = X.mean(0)
            U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
            k = min(4096, Vt.shape[0])
            P = Vt[:k] / (S[:k, None] / np.sqrt(len(X)) + 1e-12)  # whitening
            dp.mkdir(exist_ok=True)
            for t in all_trajs:
                Y = (np.load(dd / f"{t}.npy") - mu) @ P.T
                Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
                np.save(dp / f"{t}.npy", Y.astype(np.float32))
            log(f"pca: {k} dims from {len(X)} database VLADs")


if __name__ == "__main__":
    main()
