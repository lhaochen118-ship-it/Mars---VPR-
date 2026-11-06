"""Re-derive VPR-GT-RPE-0 from the released per-RPE relative pose estimates (misc/rpe_estimates), the released
camera poses and the thresholds of the paper's Table V, and compare with the released gt_hard / gt_soft.

The RPE inference itself is not reproducible (code unreleased), but this checks the deterministic voting step:
  a pair (db i, query j) is positive iff  dist_ij <= d_max  and  >= n RPEs satisfy, for BOTH input orderings,
  dR < tau_R  and  dt < tau_t,  with dt = min_s || s * t_est - t_gt ||  (5-DoF relaxation).
Conventions not stated in the paper (direction of the relative transform, sign constraint on s) are searched.

python rederive_gt.py --root <vpr-space-v0> [--setups ...] --out gt_rederive.json
"""
import argparse, itertools, json
from pathlib import Path

import numpy as np

RPES = ["dust3r", "must3r", "mapanything", "madpose", "matchdepth"]
TABLE_V = {"default": {"soft": (35.0, 4.0, 30.0, 2), "hard": (35.0, 3.0, 20.0, 4)},   # d_max, tau_t, tau_R, n
           "SPICE": {"soft": (1.0, 0.4, 20.0, 1), "hard": (1.0, 0.2, 10.0, 1)}}


def rot_err_deg(Ra, Rb):
    c = (np.einsum("...ij,...ij->...", Ra, Rb) - 1.0) / 2.0  # trace(Ra^T Rb) = sum(Ra * Rb)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def trans_err(t_est, t_gt, nonneg):
    n = np.linalg.norm(t_est, axis=-1, keepdims=True)
    u = np.divide(t_est, n, out=np.zeros_like(t_est), where=n > 1e-12)
    s = np.einsum("...i,...i->...", t_gt, u)
    if nonneg:
        s = np.maximum(s, 0.0)
    return np.linalg.norm(t_gt - s[..., None] * u, axis=-1)


def errors(E, G):
    """E, G: (..., 4, 4) estimated / ground-truth relative transforms -> (dR deg, dt m)."""
    return rot_err_deg(E[..., :3, :3], G[..., :3, :3]), E[..., :3, 3], G[..., :3, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--setups", nargs="*")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    setups = a.setups or sorted(p.stem for p in (root / "misc" / "rpe_estimates").glob("*.npz"))
    report = {}
    for sid in setups:
        db, q = sid.split("--")
        Pdb, Pq = np.load(root / "poses" / f"{db}.npy").astype(np.float64), np.load(root / "poses" / f"{q}.npy").astype(np.float64)
        with np.load(root / "vpr_gt" / "VPR-GT-RPE-0" / f"{sid}.npz") as g:
            hard, soft = g["gt_hard"].astype(bool), g["gt_soft"].astype(bool)
        r = np.load(root / "misc" / "rpe_estimates" / f"{sid}.npz")
        ev, dist = r["evaluated"].astype(bool), r["gt_dist_m"].astype(np.float64)
        I, J = np.nonzero(ev)
        inv = np.linalg.inv
        # candidate GT relative transforms for ordering 0 (I1 = db_i, I2 = q_j); ordering 1 uses the inverse
        G_a = inv(Pdb[I]) @ Pq[J]          # pose of q-camera in db-camera frame
        G_b = inv(G_a)                     # pose of db-camera in q-camera frame
        params = TABLE_V["SPICE" if sid.startswith("SPICE") else "default"]
        per_conv = {}
        for gt_conv, est_inv, nonneg in itertools.product(("a", "b"), (False, True), (False, True)):
            G0 = G_a if gt_conv == "a" else G_b
            G1 = inv(G0)
            counts = {k: np.zeros(len(I), dtype=np.int32) for k in params}
            for name in RPES:
                E = r[name][:, I, J].astype(np.float64)  # (2, M, 4, 4)
                bad = ~np.isfinite(E).all(axis=(-1, -2))
                E = np.where(bad[..., None, None], np.eye(4), E)
                if est_inv:
                    E = inv(E)
                ok = {}
                for o, G in ((0, G0), (1, G1)):
                    dR, te, tg = errors(E[o], G)
                    dt = trans_err(te, tg, nonneg)
                    for k, (dmax, tt, tr, n) in params.items():
                        ok.setdefault(k, np.ones(len(I), bool))
                        ok[k] &= (dR < tr) & (dt < tt) & ~bad[o]
                for k in params:
                    counts[k] += ok[k]
            res = {}
            for k, (dmax, tt, tr, n) in params.items():
                pred = np.zeros_like(hard)
                pred[I, J] = (counts[k] >= n) & (dist[I, J] <= dmax)
                ref = hard if k == "hard" else soft
                res[k] = {"mismatch": int((pred != ref).sum()), "pred_pos": int(pred.sum()), "ref_pos": int(ref.sum()),
                          "fp": int((pred & ~ref).sum()), "fn": int((~pred & ref).sum())}
            per_conv[f"gt={gt_conv},est_inv={est_inv},s>=0={nonneg}"] = res
        best = min(per_conv, key=lambda c: per_conv[c]["hard"]["mismatch"] + per_conv[c]["soft"]["mismatch"])
        report[sid] = {"n_pairs": int(hard.size), "n_evaluated": int(ev.sum()), "best_convention": best,
                       "best": per_conv[best], "all": per_conv}
        b = per_conv[best]
        print(f"{sid:60s} best[{best}] hard mismatch {b['hard']['mismatch']:6d} (ref {b['hard']['ref_pos']}, pred {b['hard']['pred_pos']}) "
              f"soft mismatch {b['soft']['mismatch']:6d} (ref {b['soft']['ref_pos']}, pred {b['soft']['pred_pos']})", flush=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
