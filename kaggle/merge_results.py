"""Merge per-setup metrics of one descriptor set computed in several runs (e.g. a session that failed on one trajectory
plus a patch session) into a single 16-setup result, recomputing the aggregates with metrics_all.aggregates.

Valid because every metric of a setup depends only on that setup's DB/query descriptors (the authors' similarity
standardizes with the setup's own DB statistics) and every descriptor here is computed image by image.
A setup present in more than one input must agree to 1e-9, otherwise the merge stops.

python3 kaggle/merge_results.py --name anyloc_global_322 --inputs a.json b.json --out results/runs/merged-anyloc/results
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "vprspace_repro"))
from metrics_all import REF, aggregates  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="merged")
    a = ap.parse_args()
    per, f6, sources = {}, {}, {}
    for f in a.inputs:
        r = json.load(open(f))
        assert r["name"] == a.name, (f, r["name"])
        for i, v in r["per_setup"].items():
            if i in per:
                diff = max(abs(per[i][m] - v[m]) for m in ("AUPRC", "R@100p", "R@1", "R@5", "R@10"))
                assert diff < 1e-9 and per[i]["n_db"] == v["n_db"] and per[i]["n_q"] == v["n_q"], (a.name, i, diff)
                continue
            per[i], sources[i] = v, os.path.basename(f)
        f6.update(r.get("fig6", {}))
    order = REF["setup_order"]
    assert set(per) <= set(order), set(per) - set(order)
    per = {i: per[i] for i in order if i in per}
    missing = [i for i in order if i not in per]
    dims = {v["dim"] for v in per.values()}
    assert len(dims) == 1, dims
    res = {"name": a.name, "tag": a.tag, "complete": not missing, "missing": missing, "per_setup": per,
           "aggregates": aggregates(per) if len(per) > 1 else None, "fig6": f6, "merged_from": sources,
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f"{a.name}__{a.tag}.json"), "w"), indent=1)
    msg = "missing " + ", ".join(missing) if missing else "AUPRC mean %.4f" % res["aggregates"]["AUPRC"]["mean"]
    print(f"{a.name}: {len(per)}/16 setups, {msg}")


if __name__ == "__main__":
    main()
