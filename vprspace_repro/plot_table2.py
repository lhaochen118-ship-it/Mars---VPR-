"""Scatter of reproduced vs paper AUPRC (Table II), one panel per paper column, with the +-0.005 rounding band.

python plot_table2.py --results results/runs --out results/figures/table2_ours_vs_paper.png \
    --sets megaloc@a-exact=megaloc boq@a-exact=boq salad_322@b-salad=salad anyloc_global_322@merged=anyloc_global ...
"""
import argparse, glob, json, os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REF = json.load(open(HERE / "paper_reference.json"))
O = REF["setup_order"]
NAMES = {"megaloc": "MegaLoc", "boq": "BoQ", "salad": "SALAD", "anyloc_global": "AnyLoc (global)", "densevlad": "DenseVLAD"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="+", required=True, help="name@tag=paper_column[:label]")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sets = []
    for s in a.sets:
        nt, col = s.split("=")
        col, _, lab = col.partition(":")
        n, t = nt.split("@")
        r = json.load(open(glob.glob(os.path.join(a.results, "*", "results", f"{n}__{t}.json"))[0]))
        sets.append((col, lab or n, np.array([r["per_setup"][i]["AUPRC"] for i in O])))
    cols = list(dict.fromkeys(c for c, _, _ in sets))
    fig, axes = plt.subplots(1, len(cols), figsize=(3.1 * len(cols), 3.3), squeeze=False)
    for ax, col in zip(axes[0], cols):
        p = np.array(REF["table_II_auprc"][col])
        g = np.linspace(0, 0.85, 2)
        ax.fill_between(g, g - 0.005, g + 0.005, color="0.85", lw=0)
        ax.plot(g, g, color="0.5", lw=0.8)
        for k, (c, lab, x) in enumerate([s for s in sets if s[0] == col]):
            exact = np.abs(x - p) <= 0.005 + 1e-9
            ax.scatter(p, x, s=14, marker="os^D"[k % 4], label=f"{lab} ({exact.sum()}/16)", alpha=0.85)
        ax.set_title(NAMES.get(col, col), fontsize=10)
        ax.set_xlabel("paper AUPRC (2 dp)")
        ax.set_xlim(0, 0.85); ax.set_ylim(0, 0.85); ax.set_aspect("equal")
        ax.legend(fontsize=6.5, loc="upper left")
    axes[0][0].set_ylabel("reproduced AUPRC")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("written", a.out)


if __name__ == "__main__":
    main()
