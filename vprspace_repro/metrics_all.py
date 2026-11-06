"""Compute the paper's metrics for every descriptor set found in --desc-dir, using the authors' code unchanged
(evaluate.similarity + evaluate.evaluate, dataset.load for GT). Writes one JSON per descriptor set.

python metrics_all.py --root <vpr-space root> --desc-dir <dir with <name>/<traj>.npy> --out <results dir> [--names ...]
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "upstream"))
import dataset  # noqa: E402  (authors' file)
from evaluate import evaluate, similarity  # noqa: E402  (authors' file)

REF = json.load(open(HERE / "paper_reference.json"))
METRICS = ["AUPRC", "R@100p", "R@1", "R@5", "R@10"]


def aggregates(per):
    out = {}
    ids = list(per)
    for m in METRICS:
        v = np.array([per[i][m] for i in ids], dtype=np.float64)
        out[m] = {"mean": float(v.mean()), "std_ddof0": float(v.std(ddof=0)), "std_ddof1": float(v.std(ddof=1)),
                  "min": float(v.min()), "argmin": ids[int(v.argmin())]}
    worst = min(ids, key=lambda i: per[i]["AUPRC"])  # alternative reading of "worst case": hardest setup by AUPRC
    out["worst_setup_by_auprc"] = {"setup": worst, **{m: per[worst][m] for m in METRICS}}
    return out


def fig6(S, gt_hard, query):
    rows = {}
    for base, qi in (("0-based", query), ("1-based", query - 1)):
        if not 0 <= qi < S.shape[1]:
            continue
        col = S[:, qi]
        pos = gt_hard[:, qi].astype(bool)
        rows[base] = {"top1_db": int(col.argmax()), "top1_sim": float(col.max()),
                      "top1_in_hard_gt": bool(pos[col.argmax()]),
                      "best_correct_sim": float(col[pos].max()) if pos.any() else None,
                      "best_correct_db": int(np.flatnonzero(pos)[col[pos].argmax()]) if pos.any() else None}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--desc-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    desc_dir, out = Path(a.desc_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = a.names or sorted(p.name for p in desc_dir.iterdir() if p.is_dir() and not p.name.startswith('_'))
    setups = {i: dataset.load(i, a.root) for i in dataset.ids()}
    for name in names:
        per, f6, missing = {}, {}, []
        for i, d in setups.items():
            fdb, fq = desc_dir / name / f"{d.db.name}.npy", desc_dir / name / f"{d.query.name}.npy"
            if not (fdb.exists() and fq.exists()):
                missing.append(i)
                continue
            db, q = np.load(fdb), np.load(fq)
            assert db.shape[0] == len(d.db.images) and q.shape[0] == len(d.query.images), (name, i, db.shape, q.shape)
            assert d.gt_hard.shape == (db.shape[0], q.shape[0]), (name, i, d.gt_hard.shape)
            S = similarity(db, q)
            per[i] = {**evaluate(S, d.gt_soft, d.gt_hard), "n_db": int(db.shape[0]), "n_q": int(q.shape[0]), "dim": int(db.shape[1])}
            if i in REF["fig6_boq_similarities"]:
                f6[i] = fig6(S, d.gt_hard, REF["fig6_boq_similarities"][i]["query"])
        res = {"name": name, "tag": a.tag, "complete": not missing, "missing": missing, "per_setup": per,
               "aggregates": aggregates(per) if len(per) > 1 else None, "fig6": f6, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(res, open(out / f"{name}{('__' + a.tag) if a.tag else ''}.json", "w"), indent=1)
        msg = "missing " + str(len(missing)) if missing else "AUPRC mean %.4f" % res["aggregates"]["AUPRC"]["mean"]
        print(f"{name}: {len(per)}/16 setups, {msg}", flush=True)


if __name__ == "__main__":
    main()
