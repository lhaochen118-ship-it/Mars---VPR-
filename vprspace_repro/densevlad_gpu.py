"""DenseVLAD re-implementation on GPU (Torii et al., CVPR 2015), used because neither an official Python version
nor the VPR-Space authors' implementation/vocabulary is available. Follows the original pipeline (vl_phow + VLAD):
  * grayscale, long side downscaled to 640 px (area/box filter), aspect ratio kept
  * dense SIFT on a 2-px grid at 4 scales, bin sizes 4/6/8/10 px (descriptor width 16/24/32/40 px), per-scale
    Gaussian pre-smoothing sigma = sqrt((bin/6)^2 - 0.25) as in VLFeat's vl_phow (magnif = 6), RootSIFT
  * 128-word vocabulary (k-means) learned from database images only, hard-assignment VLAD, intra-normalisation,
    global L2 -> 16384-D; optional PCA-whitening to 4096-D learned from pooled database VLADs.
Dense SIFT uses kornia.feature.DenseSIFTDescriptor (preinstalled in the Kaggle image).

Stages (run by the session bootstrap): --stage vocab (before workers), --stage pca (after workers).
"""
import argparse, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile

# same handling as worker.py for the truncated MadMax-E0/0293.png of the release (decode available rows)
ImageFile.LOAD_TRUNCATED_IMAGES = os.environ.get("VPRS_LOAD_TRUNCATED", "1") == "1"

BINS, STEP, LONG_SIDE, K = (4, 6, 8, 10), 2, 640, 128
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[densevlad]", *a, flush=True)


def load_gray(path):
    img = Image.open(path).convert("L")
    w, h = img.size
    s = LONG_SIDE / max(w, h)
    if s < 1:
        img = img.resize((round(w * s), round(h * s)), Image.BOX)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None]  # (1, H, W)


class DenseRootSIFT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        import kornia
        self.kornia = kornia
        self.sift = torch.nn.ModuleList([kornia.feature.DenseSIFTDescriptor(num_ang_bins=8, num_spatial_bins=4,
                                                                            spatial_bin_size=b, rootsift=True, stride=STEP, padding=0)
                                         for b in BINS])

    @torch.no_grad()
    def forward(self, x):  # x: (B, 1, H, W) in [0, 1] -> list of (N_i, 128)
        out = []
        for b, sift in zip(BINS, self.sift):
            sigma = float(np.sqrt((b / 6.0) ** 2 - 0.25))
            k = int(2 * round(3 * sigma) + 1)
            xs = self.kornia.filters.gaussian_blur2d(x, (k, k), (sigma, sigma), border_type="replicate")
            d = sift(xs)  # (B, 128, h, w)
            out.append(d.flatten(2).transpose(1, 2))  # (B, h*w, 128)
        return torch.cat(out, dim=1)  # (B, N, 128)


class DenseVLAD(torch.nn.Module):
    def __init__(self, centers):
        super().__init__()
        self.dsift = DenseRootSIFT()
        self.register_buffer("c", torch.as_tensor(centers, dtype=torch.float32))

    @torch.no_grad()
    def forward(self, x):
        D = self.dsift(x)  # (B, N, 128)
        outs = []
        for d in D:
            a = torch.cdist(d, self.c).argmin(1)  # hard assignment
            v = torch.zeros_like(self.c).index_add_(0, a, d - self.c[a])
            v = F.normalize(v, dim=1)  # intra-normalisation
            outs.append(F.normalize(v.flatten(), dim=0))
        return torch.stack(outs)


def kmeans_gpu(x, k, iters=50, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g)[:k].to(x.device)].clone()
    for _ in range(iters):
        a = torch.cat([torch.cdist(x[i:i + 262144], c).argmin(1) for i in range(0, len(x), 262144)])
        cnt = torch.bincount(a, minlength=k).float()
        s = torch.zeros_like(c).index_add_(0, a, x)
        empty = cnt == 0
        c = torch.where(empty[:, None], x[torch.randint(len(x), (k,), generator=g).to(x.device)], s / cnt.clamp(min=1)[:, None])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--desc-dir", required=True)
    ap.add_argument("--stage", choices=["vocab", "pca"], required=True)
    ap.add_argument("--samples-per-image", type=int, default=300)
    a = ap.parse_args()
    root, desc = Path(a.root), Path(a.desc_dir)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "upstream"))
    import dataset
    pairs = [dataset.SETUPS[i] for i in dataset.ids()]
    db_trajs = sorted({db for db, _ in pairs})
    all_trajs = sorted({t for p in pairs for t in p})
    vocab_f = desc / "_densevlad" / "vocab_k128.npy"
    if a.stage == "vocab":
        vocab_f.parent.mkdir(parents=True, exist_ok=True)
        ds = DenseRootSIFT().to(DEV)
        rng = np.random.default_rng(0)
        samples, t0 = [], time.time()
        paths = [p for t in db_trajs for p in sorted((root / "images" / t).glob("*.png"))]
        for p in paths:  # vocabulary from DATABASE images only
            d = ds(load_gray(p)[None].to(DEV))[0]
            samples.append(d[torch.as_tensor(rng.choice(len(d), size=min(a.samples_per_image, len(d)), replace=False), device=DEV)])
        X = torch.cat(samples)
        log(f"vocab: {len(paths)} database images, {len(X)} sampled RootSIFT descriptors in {time.time() - t0:.0f}s")
        C = kmeans_gpu(X, K)
        np.save(vocab_f, C.cpu().numpy())
        log(f"vocab saved {vocab_f} ({time.time() - t0:.0f}s)")
    else:
        dd, dp = desc / "densevlad_16384", desc / "densevlad_pca4096"
        missing = [t for t in all_trajs if not (dd / f"{t}.npy").exists()]
        if missing:
            log("pca skipped, missing", missing)
            return
        X = torch.as_tensor(np.concatenate([np.load(dd / f"{t}.npy") for t in db_trajs]), dtype=torch.float64, device=DEV)
        mu = X.mean(0)
        U, S, Vh = torch.linalg.svd(X - mu, full_matrices=False)
        k = min(4096, Vh.shape[0])
        P = Vh[:k] / (S[:k, None] / np.sqrt(len(X)) + 1e-12)
        dp.mkdir(exist_ok=True)
        for t in all_trajs:
            Y = (torch.as_tensor(np.load(dd / f"{t}.npy"), dtype=torch.float64, device=DEV) - mu) @ P.T
            np.save(dp / f"{t}.npy", F.normalize(Y, dim=1).float().cpu().numpy())
        log(f"pca-whitening: {k} dims from {len(X)} database VLADs")


if __name__ == "__main__":
    main()
