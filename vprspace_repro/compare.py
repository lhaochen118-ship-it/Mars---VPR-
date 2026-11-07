"""Compare reproduced metrics with the paper (paper_reference.json) and write a Markdown report.

python compare.py --results <dir with *.json from metrics_all> --out report.md [--json summary.json]
"""
import argparse, glob, json, os
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = json.load(open(HERE / "paper_reference.json"))
ORDER = REF["setup_order"]
LABEL = REF["paper_labels"]
METRICS = ["R@1", "R@5", "R@10", "R@100p"]


def paper_column(name):
    """Map a descriptor-set name to the paper column it should be compared with (or None)."""
    if name.startswith("megaloc"):
        return "megaloc"
    if name.startswith("boq"):
        return "boq"
    if name.startswith("salad"):
        return "salad"
    if name.startswith("anyloc_global"):
        return "anyloc_global"
    if name.startswith("densevlad"):
        return "densevlad"
    return None


def match2(x, p):  # paper values are rounded to 2 decimals
    return abs(round(x, 2) - p) < 1e-9 or abs(x - p) <= 0.005 + 1e-9


def load(results_dir):
    rows = {}
    for f in sorted(glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)):
        if os.path.basename(f).startswith("_"):
            continue
        r = json.load(open(f))
        if "per_setup" not in r:
            continue
        key = f"{r['name']}" + (f" [{r['tag']}]" if r.get("tag") else "")
        rows[key] = r
    # partial runs (split / failed-trajectory sessions) are dropped once a complete result of the same set exists
    complete = {r["name"] for r in rows.values() if r["complete"]}
    return {k: r for k, r in rows.items() if r["complete"] or r["name"] not in complete}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = load(a.results)
    summary, lines = {}, []
    lines.append("# VPR-Space reproduction: comparison with the paper\n")
    lines.append(f"{len(rows)} descriptor sets with results.\n")
    # ---- Table II (AUPRC per setup)
    for key, r in sorted(rows.items()):
        col = paper_column(r["name"])
        per = r["per_setup"]
        s = {"complete": r["complete"], "n_setups": len(per)}
        if col and r["complete"]:
            ref = REF["table_II_auprc"][col]
            diffs = [per[sid]["AUPRC"] - ref[i] for i, sid in enumerate(ORDER)]
            exact = [match2(per[sid]["AUPRC"], ref[i]) for i, sid in enumerate(ORDER)]
            agg = r["aggregates"]
            s.update(paper_column=col, auprc_exact=sum(exact), auprc_max_abs_diff=max(abs(d) for d in diffs),
                     auprc_mean_abs_diff=sum(abs(d) for d in diffs) / len(diffs),
                     mean=agg["AUPRC"]["mean"], paper_mean=REF["table_II_auprc"]["avg_mean"][col])
            t3 = REF["table_III_recall"][col]
            t3m = {}
            for m in METRICS:
                ours = agg[m]
                t3m[m] = {"worst_min": ours["min"], "worst_hardest_setup": agg["worst_setup_by_auprc"][m],
                          "mean": ours["mean"], "std0": ours["std_ddof0"], "std1": ours["std_ddof1"],
                          "paper_worst": t3["worst"][m], "paper_mean": t3["mean"][m], "paper_std": t3["std"][m]}
            s["table_III"] = t3m
        summary[key] = s
    best = defaultdict(list)
    for key, s in summary.items():
        if "paper_column" in s:
            best[s["paper_column"]].append((s["auprc_mean_abs_diff"], key))
    lines.append("## Table II (AUPRC): agreement with the paper\n")
    lines.append("| descriptor set | paper column | setups exact (2 dp) | mean abs diff | max abs diff | mean AUPRC (ours / paper) |")
    lines.append("|---|---|---|---|---|---|")
    for col in ["megaloc", "boq", "salad", "anyloc_global", "densevlad"]:
        for d, key in sorted(best.get(col, [])):
            s = summary[key]
            lines.append(f"| `{key}` | {col} | {s['auprc_exact']}/16 | {s['auprc_mean_abs_diff']:.4f} | {s['auprc_max_abs_diff']:.4f} | {s['mean']:.3f} / {s['paper_mean']:.2f} |")
    lines.append("")
    # per-setup detail for the exact group and for the closest variant of each approximate method
    for col in ["megaloc", "boq", "salad", "anyloc_global", "densevlad"]:
        if not best.get(col):
            continue
        d, key = sorted(best[col])[0]
        r = rows[key]
        lines.append(f"### Per setup: `{key}` vs paper column {col}\n")
        lines.append("| setup | ours | paper | diff | exact |")
        lines.append("|---|---|---|---|---|")
        for i, sid in enumerate(ORDER):
            x, p = r["per_setup"][sid]["AUPRC"], REF["table_II_auprc"][col][i]
            lines.append(f"| {LABEL[sid]} | {x:.4f} | {p:.2f} | {x - p:+.4f} | {'✅' if match2(x, p) else '❌'} |")
        s = summary[key]["table_III"]
        lines.append("\n| Table III | ours worst (min) | ours worst (hardest setup) | paper worst | ours mean ± std (ddof0/ddof1) | paper mean ± std |")
        lines.append("|---|---|---|---|---|---|")
        for m in METRICS:
            t = s[m]
            lines.append(f"| {m} | {t['worst_min']:.3f} | {t['worst_hardest_setup']:.3f} | {t['paper_worst']:.3f} | "
                         f"{t['mean']:.3f} ± {t['std0']:.3f}/{t['std1']:.3f} | {t['paper_mean']:.2f} ± {t['paper_std']:.2f} |")
        lines.append("")
    # ---- Fig. 6 (BoQ similarities)
    f6rows = [(k, r) for k, r in rows.items() if r["name"].startswith("boq") and r.get("fig6")]
    if f6rows:
        lines.append("## Fig. 6 (BoQ similarities, 3 decimals)\n")
        lines.append("| run | setup | query index | best correct (ours / paper) | false positive = top-1 (ours / paper) |")
        lines.append("|---|---|---|---|---|")
        for k, r in f6rows:
            for sid, v in r["fig6"].items():
                ref = REF["fig6_boq_similarities"][sid]
                for base, x in v.items():
                    bc = x["best_correct_sim"]
                    bcs = f"{bc:.3f}" if bc is not None else "– (no hard positive)"
                    lines.append(f"| `{k}` | {LABEL[sid]} | {ref['query']} ({base}) | {bcs} / {ref['correct']:.3f} | {x['top1_sim']:.3f} / {ref['fp']:.3f} |")
        lines.append("")
    # ---- everything else (no paper numbers)
    lines.append("## All descriptor sets (mean over 16 setups)\n")
    lines.append("| descriptor set | AUPRC | R@1 | R@5 | R@10 | R@100p | worst AUPRC |")
    lines.append("|---|---|---|---|---|---|---|")
    for key, r in sorted(rows.items(), key=lambda kv: -(kv[1]["aggregates"] or {}).get("AUPRC", {}).get("mean", -1)):
        if not r["complete"]:
            lines.append(f"| `{key}` | incomplete ({len(r['per_setup'])}/16) | | | | | |")
            continue
        g = r["aggregates"]
        lines.append(f"| `{key}` | {g['AUPRC']['mean']:.3f} | {g['R@1']['mean']:.3f} | {g['R@5']['mean']:.3f} | {g['R@10']['mean']:.3f} | {g['R@100p']['mean']:.3f} | {g['AUPRC']['min']:.3f} |")
    open(a.out, "w").write("\n".join(lines) + "\n")
    if a.json:
        json.dump(summary, open(a.json, "w"), indent=1)
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
