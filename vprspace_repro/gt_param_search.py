"""Search GT hyper-parameters (tau_t, tau_R, n, d_max) that reproduce the paper's Table IV columns
("% queries w/ match", "% of pos. matches") from the released RPE estimates, to explain why the released GT
(which matches Table V exactly) disagrees with Table IV's "% of pos. matches".

python gt_param_search.py --root <vpr-space-v0> --out search.json
"""
import argparse, itertools, json
from pathlib import Path

import numpy as np

from rederive_gt import RPES, rot_err_deg, trans_err

REF = json.load(open(Path(__file__).resolve().parent / "paper_reference.json"))["table_IV_dataset_stats"]


def pair_errors(root, sid):
    db, q = sid.split("--")
    Pdb, Pq = np.load(root / "poses" / f"{db}.npy").astype(float), np.load(root / "poses" / f"{q}.npy").astype(float)
    r = np.load(root / "misc" / "rpe_estimates" / f"{sid}.npz")
    ev, dist = r["evaluated"].astype(bool), r["gt_dist_m"].astype(float)
    I, J = np.nonzero(ev)
    G0 = np.linalg.inv(Pq[J]) @ Pdb[I]
    G1 = np.linalg.inv(G0)
    dR = np.zeros((len(RPES), 2, len(I)), np.float32)
    dt = np.zeros_like(dR)
    for k, name in enumerate(RPES):
        E = r[name][:, I, J].astype(float)
        bad = ~np.isfinite(E).all(axis=(-1, -2))
        E = np.where(bad[..., None, None], np.eye(4), E)
        for o, G in ((0, G0), (1, G1)):
            dR[k, o] = np.where(bad[o], 1e9, rot_err_deg(E[o][:, :3, :3], G[:, :3, :3]))
            dt[k, o] = np.where(bad[o], 1e9, trans_err(E[o][:, :3, 3], G[:, :3, 3], True))
    return {"I": I, "J": J, "dist": dist[I, J], "dR": dR, "dt": dt, "shape": ev.shape}


def stats(e, tt, tr, n, dmax):
    ok = ((e["dR"] < tr) & (e["dt"] < tt)).all(axis=1).sum(axis=0)  # both orderings, count RPEs
    pos = (ok >= n) & (e["dist"] <= dmax)
    nq = e["shape"][1]
    qm = np.zeros(nq, bool)
    qm[e["J"][pos]] = True
    return 100 * qm.mean(), 100 * pos.sum() / (e["shape"][0] * e["shape"][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    groups = {"terrestrial": [s for s in REF if not s.startswith("_") and not s.startswith("SPICE")],
              "SPICE": [s for s in REF if s.startswith("SPICE")]}
    grids = {"terrestrial": dict(tt=[1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6], tr=[5, 10, 15, 20, 25, 30, 40], n=[1, 2, 3, 4, 5], dmax=[20, 25, 30, 35, 40, 1e9]),
             "SPICE": dict(tt=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5], tr=[5, 10, 15, 20, 25, 30], n=[1, 2, 3, 4, 5], dmax=[0.5, 1, 1.5, 2, 1e9])}
    report = {}
    for g, sids in groups.items():
        errs = {s: pair_errors(root, s) for s in sids}
        res = []
        for tt, tr, n, dmax in itertools.product(*grids[g].values()):
            qerr, perr, exact_q, exact_p = 0.0, 0.0, 0, 0
            for s in sids:
                qm, pp = stats(errs[s], tt, tr, n, dmax)
                rq, rp = REF[s][2], REF[s][3]
                qerr += abs(qm - rq); perr += abs(pp - rp)
                exact_q += abs(round(qm, 1) - rq) < 1e-6 or abs(qm - rq) <= 0.05
                exact_p += abs(round(pp, 2) - rp) < 1e-6 or abs(pp - rp) <= 0.005
            res.append({"tau_t": tt, "tau_R": tr, "n": n, "d_max": dmax, "exact_pct_q": int(exact_q), "exact_pct_pos": int(exact_p),
                        "sum_abs_err_q": round(qerr, 3), "sum_abs_err_pos": round(perr, 3), "n_setups": len(sids)})
        res.sort(key=lambda x: (-(x["exact_pct_q"] + x["exact_pct_pos"]), x["sum_abs_err_pos"] + x["sum_abs_err_q"]))
        report[g] = res[:15]
        print(f"== {g}: best combos (exact %q / exact %pos of {len(sids)})")
        for x in res[:8]:
            print("  ", x)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
