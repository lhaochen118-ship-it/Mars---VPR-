"""Is the MegaLoc/SALAD gap on MADMAX G0-G2 explained by a different ground truth at evaluation time?

Re-derives G0-G2 GT matrices from the released RPE estimates for (a) every one-parameter change of Table V and
(b) a grid of hard/soft thresholds (soft >= hard), and reports which variants put BOTH MegaLoc and BoQ inside the
paper's rounding intervals (MegaLoc 0.57, BoQ 0.49). Uses the authors' similarity/auprc unchanged.

python g0g2_gt_sensitivity.py --root <vpr-space-v0 with misc/rpe_estimates> --desc-dir <dir with boq/, megaloc/> --out f.json
"""
import argparse, itertools, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "upstream"))
import dataset  # noqa: E402
from evaluate import similarity  # noqa: E402
from metrics import auprc  # noqa: E402
from gt_param_search import pair_errors  # noqa: E402

SID = "MadMax-G0--MadMax-G2"
PAPER = {"megaloc": 0.57, "boq": 0.49}
BASE = dict(htt=3, htr=20, hn=4, stt=4, str_=30, sn=2, dmax=35)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--desc-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, dd = Path(a.root), Path(a.desc_dir)
    e = pair_errors(root, SID)
    d = dataset.load(SID, str(root))

    def gt(tt, tr, n, dmax):
        ok = ((e["dR"] < tr) & (e["dt"] < tt)).all(axis=1).sum(axis=0)
        pos = (ok >= n) & (e["dist"] <= dmax)
        M = np.zeros(e["shape"], bool)
        M[e["I"][pos], e["J"][pos]] = True
        return M

    S = {m: similarity(np.load(dd / m / "MadMax-G0.npy"), np.load(dd / m / "MadMax-G2.npy")) for m in PAPER}

    def run(p):
        H = gt(p["htt"], p["htr"], p["hn"], p["dmax"])
        So = gt(p["stt"], p["str_"], p["sn"], p["dmax"]) | H
        r = {m: float(auprc(S[m], H, So)) for m in S}
        r["matches_both"] = all(abs(r[m] - PAPER[m]) <= 0.005 for m in PAPER)
        return r

    base_H, base_S = gt(3, 20, 4, 35), gt(4, 30, 2, 35)
    rep = {"released_gt_equals_table_V": bool((base_H == d.gt_hard.astype(bool)).all() and (base_S == d.gt_soft.astype(bool)).all()),
           "released": run(BASE), "one_parameter": [], "grid_matches": []}
    grid1 = dict(htt=[2, 2.5, 3.5, 4], htr=[15, 25, 30], hn=[2, 3, 5], stt=[3, 5, 6], str_=[20, 25, 40], sn=[1, 3], dmax=[25, 30, 40, 50])
    for k, vals in grid1.items():
        for v in vals:
            p = dict(BASE, **{k: v})
            rep["one_parameter"].append({"change": f"{k}={v}", **run(p)})
    n_tot = 0
    for htt, htr, hn in itertools.product([2, 2.5, 3, 3.5, 4, 5], [15, 20, 25, 30], [2, 3, 4, 5]):
        for stt, str_, sn in itertools.product([3, 4, 5, 6], [20, 30, 40], [1, 2, 3]):
            if stt < htt or str_ < htr or sn > hn:
                continue
            n_tot += 1
            r = run(dict(htt=htt, htr=htr, hn=hn, stt=stt, str_=str_, sn=sn, dmax=35))
            if r["matches_both"]:
                rep["grid_matches"].append({"hard": [htt, htr, hn], "soft": [stt, str_, sn], **r})
    rep["grid_size"] = n_tot
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("one_parameter", "grid_matches")}, indent=1))
    print("one-parameter variants matching both:", [x["change"] for x in rep["one_parameter"] if x["matches_both"]])
    print(f"grid variants matching both: {len(rep['grid_matches'])}/{n_tot}")


if __name__ == "__main__":
    main()
