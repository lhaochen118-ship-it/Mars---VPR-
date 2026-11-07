"""PCA-whitening of DenseVLAD descriptors (the last step of Torii et al.'s DenseVLAD), learned from the pooled
DATABASE descriptors of the benchmark, for several output dimensions k.

Torii et al. learn a 4096-D whitening on an independent, large urban set; that projection is not available
(247code.zip is gone). With only n = 3,989 database images, a whitening must keep k well below n: the centred
data have rank <= n - 1, and whitening the training set at full rank maps every database vector to the same norm
with pairwise cosine -1/(n-1), i.e. it erases the structure retrieval relies on (this is why the earlier
densevlad_pca4096 collapsed to AUPRC 0.04). We therefore sweep k.

    y = diag(1/sigma_1..k) V_k^T (x - mu),  then L2-normalised        (sigma_i: std of PCA component i on the DB)

python densevlad_pca.py --root <vpr-space-v0> --desc-dir <desc> --name densevlad_vlfeat_640_drop0 --k 256 512 1024
"""
import argparse, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "upstream"))
import dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--desc-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--k", type=int, nargs="+", required=True)
    a = ap.parse_args()
    desc = Path(a.desc_dir)
    pairs = [dataset.SETUPS[i] for i in dataset.ids()]
    db_trajs = sorted({db for db, _ in pairs})
    trajs = sorted({t for p in pairs for t in p})
    X = np.concatenate([np.load(desc / a.name / f"{t}.npy") for t in db_trajs]).astype(np.float64)
    n = len(X)
    mu = X.mean(0)
    Xc = X - mu
    lam, U = np.linalg.eigh(Xc @ Xc.T)            # Gram trick: n x n instead of D x D
    order = np.argsort(lam)[::-1]
    lam, U = np.clip(lam[order], 0, None), U[:, order]
    rank = int((lam > lam[0] * 1e-10).sum())
    print(f"{a.name}: n={n} database descriptors, D={X.shape[1]}, numerical rank {rank}")
    for k in a.k:
        k_eff = min(k, rank)
        V = Xc.T @ U[:, :k_eff] / np.sqrt(lam[:k_eff])   # D x k, orthonormal principal directions
        sigma = np.sqrt(lam[:k_eff] / (n - 1))            # std of each component over the database
        P = V / sigma                                       # whitening: unit variance per component
        out = desc / f"{a.name}_pcaw{k_eff}"
        out.mkdir(exist_ok=True)
        for t in trajs:
            Y = (np.load(desc / a.name / f"{t}.npy").astype(np.float64) - mu) @ P
            Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            np.save(out / f"{t}.npy", Y.astype(np.float32))
        print(f"  k={k_eff}: explained variance {lam[:k_eff].sum() / lam.sum():.3f}")


if __name__ == "__main__":
    main()
