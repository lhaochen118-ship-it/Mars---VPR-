"""Markdown tables for the results report (Table II per setup, Table III, best variant per unpublished family),
generated from results/runs so no number is copied by hand.

python report_tables.py --results results/runs > tables.md
"""
import argparse, glob, json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = json.load(open(HERE / "paper_reference.json"))
O, L = REF["setup_order"], REF["paper_labels"]
MAIN = [("MegaLoc", "megaloc", "a-exact", "megaloc"), ("BoQ", "boq", "a-exact", "boq"),
        ("SALAD (322)", "salad_322", "b-salad", "salad"), ("AnyLoc-global (322)", "anyloc_global_322", "merged", "anyloc_global")]


def f(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def load(results, name, tag):
    return json.load(open(glob.glob(os.path.join(results, "*", "results", f"{name}__{tag}.json"))[0]))


def ok(x, p):
    return abs(round(x, 2) - p) < 1e-9 or abs(x - p) <= 0.005 + 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="label=name@tag[=paper_column] columns to add to Table II")
    a = ap.parse_args()
    cols = list(MAIN)
    for e in a.extra:
        lab, rest = e.split("=", 1)
        nt, _, col = rest.partition("=")
        n, t = nt.split("@")
        cols.append((lab, n, t, col or None))
    R = {c[0]: load(a.results, c[1], c[2]) for c in cols}
    print("| Cặp DB/Q | " + " | ".join(c[0] for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for k, s in enumerate(O):
        cells = []
        for lab, n, t, col in cols:
            x = R[lab]["per_setup"][s]["AUPRC"]
            if col:
                p = REF["table_II_auprc"][col][k]
                cells.append(f"{f(x)} / {f(p, 2)}" + ("" if ok(x, p) else " ✗"))
            else:
                cells.append(f(x))
        print(f"| {L[s]} | " + " | ".join(cells) + " |")
    for row, key, fn in (("Worst case", "worst_case", lambda g: g["AUPRC"]["min"]), ("Avg. mean", "avg_mean", lambda g: g["AUPRC"]["mean"]),
                         ("Avg. std (ddof=1)", "avg_std", lambda g: g["AUPRC"]["std_ddof1"])):
        cells = []
        for lab, n, t, col in cols:
            x = fn(R[lab]["aggregates"])
            cells.append(f"{f(x)} / {f(REF['table_II_auprc'][key][col], 2)}" if col else f(x))
        print(f"| **{row}** | " + " | ".join(cells) + " |")
    print()
    print("| Phương pháp | | R@1 | R@5 | R@10 | R@100p |")
    print("|---|---|---|---|---|---|")
    for lab, n, t, col in cols:
        if not col:
            continue
        g, t3 = R[lab]["aggregates"], REF["table_III_recall"][col]
        ms = ["R@1", "R@5", "R@10", "R@100p"]
        print(f"| {lab} | worst | " + " | ".join(f"{f(g[m]['min'])} / {f(t3['worst'][m], 3 if t3['worst'][m] not in (0,) and t3['worst'][m] < 0.01 else 2)}"
                                             + ("" if ok(g[m]['min'], t3['worst'][m]) else " ✗") for m in ms) + " |")
        print("| | avg ± std | " + " | ".join(f"{f(g[m]['mean'], 2)}±{f(g[m]['std_ddof1'], 2)} / {f(t3['mean'][m], 2)}±{f(t3['std'][m], 2)}"
                                              + ("" if ok(g[m]['mean'], t3['mean'][m]) and ok(g[m]['std_ddof1'], t3['std'][m]) else " ✗")
                                              for m in ms) + " |")
        print("| | (argmin) | " + " | ".join(L[g[m]["argmin"]] for m in ms) + " |")


if __name__ == "__main__":
    main()
