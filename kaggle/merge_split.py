"""Merge descriptors of a configuration split across sessions and compute metrics locally.

python3 kaggle/merge_split.py --sessions duckzuy:b-anyloc-demo-p1 duckzuy:b-anyloc-demo-p2 \
    --pattern 'desc/anyloc_.*_demo1024/.*' --root <vpr-space-v0 with GT/poses> --out results/runs/merged
"""
import argparse, glob, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from launch import USERNAMES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="merged")
    a = ap.parse_args()
    tmp = os.path.join(a.out, "_dl")
    merged = os.path.join(a.out, "desc")
    os.makedirs(merged, exist_ok=True)
    for spec in a.sessions:
        label, session = spec.split(":")
        dest = os.path.join(tmp, session)
        os.makedirs(dest, exist_ok=True)
        ref = f"{USERNAMES[label]}/vprspace-run-{session}"
        r = subprocess.run(["kg", label, "kernels", "output", ref, "-p", dest, "--file-pattern", a.pattern, "-o", "-q"],
                           capture_output=True, text=True)
        files = glob.glob(os.path.join(dest, "**", "*.npy"), recursive=True)
        print(f"{ref}: {len(files)} descriptor files {r.stderr.strip()[-200:]}", flush=True)
        for f in files:
            rel = os.path.relpath(f, os.path.join(dest, "desc")) if "/desc/" in f else os.path.relpath(f, dest)
            os.makedirs(os.path.dirname(os.path.join(merged, rel)), exist_ok=True)
            shutil.move(f, os.path.join(merged, rel))
    subprocess.run([sys.executable, os.path.join(os.path.dirname(HERE), "vprspace_repro", "metrics_all.py"), "--root", a.root,
                    "--desc-dir", merged, "--out", os.path.join(a.out, "results"), "--tag", a.tag], check=True)


if __name__ == "__main__":
    main()
