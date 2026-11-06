"""Qualitative reproductions of the paper's figures from reproduced descriptors.

Fig. 1 (right): BoQ similarity histograms (positives = hard GT, negatives = outside hard and soft GT) for
                SPICE-HL3 noon -> night (camera not stated in the paper: both SPAD and FLIR are drawn).
Fig. 5 (RPE-GT curve only): MegaLoc precision-recall on MADMAX A0-A2 with the authors' createPR.

python figures.py --root <vpr-space-v0> --desc-dir <desc> --out <dir>
"""
import argparse, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "upstream"))
import dataset  # noqa: E402
from evaluate import similarity  # noqa: E402
from metrics import createPR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--desc-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out, desc = Path(a.out), Path(a.desc_dir)
    out.mkdir(parents=True, exist_ok=True)

    def S_of(method, sid):
        d = dataset.load(sid, a.root)
        db, q = np.load(desc / method / f"{d.db.name}.npy"), np.load(desc / method / f"{d.query.name}.npy")
        return similarity(db, q), d

    # Fig. 1 right
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, sid, title in zip(axes, ["SPICE-A-noon-off-spad@5.0us--SPICE-A-night-on-spad@5.0us", "SPICE-A-noon-off@2500.0us--SPICE-A-night-on@2500.0us"],
                              ["SPICE-HL³ SPAD  DB: noon  Q: night", "SPICE-HL³ FLIR  DB: noon  Q: night"]):
        S, d = S_of("boq", sid)
        pos = S[d.gt_hard.astype(bool)]
        neg = S[~(d.gt_hard.astype(bool) | d.gt_soft.astype(bool))]
        bins = np.linspace(min(S.min(), 0), S.max(), 40)
        ax.hist(neg, bins=bins, alpha=0.6, label="negative matchings", color="tab:blue")
        ax.hist(pos, bins=bins, alpha=0.8, label="positive matchings", color="tab:orange")
        ax.set_yscale("log"); ax.set_xlabel("cosine similarity"); ax.set_ylabel("count"); ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig1_right_boq_similarity_hist.png", dpi=150)
    # Fig. 5 (RPE GT only)
    S, d = S_of("megaloc", "MadMax-A0--MadMax-A2")
    P, R = createPR(S, d.gt_hard, d.gt_soft)
    fig, ax = plt.subplots(figsize=(4, 3.2))
    ax.plot(R, P, label="Rel. Pose Est. GT (VPR-GT-RPE-0)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("MADMAX A0 – A2; MegaLoc descriptor", fontsize=9); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig5_megaloc_pr_madmax_a0_a2.png", dpi=150)
    print("figures written to", out)


if __name__ == "__main__":
    main()
