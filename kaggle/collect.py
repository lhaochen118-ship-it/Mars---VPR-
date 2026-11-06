"""Download the small result files (results/*.json, logs/*) of finished wave sessions.

python3 kaggle/collect.py kaggle/wave2_plan.json <out_dir> [--desc-pattern REGEX]
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from launch import USERNAMES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("out")
    ap.add_argument("--only", nargs="*", help="session names")
    ap.add_argument("--desc-pattern", help="also download descriptor files matching this regex")
    a = ap.parse_args()
    for s in json.load(open(a.plan))["sessions"]:
        if a.only and s["session"] not in a.only:
            continue
        ref = f"{USERNAMES[s['label']]}/vprspace-run-{s['session']}"
        st = subprocess.run(["kg", s["label"], "kernels", "status", ref], capture_output=True, text=True).stdout.strip()
        dest = os.path.join(a.out, s["session"])
        os.makedirs(dest, exist_ok=True)
        pat = r"(results/.*|logs/.*)" if not a.desc_pattern else f"(results/.*|logs/.*|{a.desc_pattern})"
        r = subprocess.run(["kg", s["label"], "kernels", "output", ref, "-p", dest, "--file-pattern", pat, "-o", "-q"],
                           capture_output=True, text=True)
        n = sum(len(fs) for _, _, fs in os.walk(dest))
        print(f"{s['session']:22s} {st.split('status')[-1].strip()[:40]:40s} files={n} {(r.stderr or '').strip()[-120:]}", flush=True)


if __name__ == "__main__":
    main()
