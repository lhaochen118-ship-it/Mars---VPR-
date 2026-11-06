"""Push Kaggle script kernels for several accounts (tokens come from /root/.kaggle/accounts.json via `kg`).

Usage (python3 kaggle/launch.py ...):
  push   <label> <slug> <code.py> [--gpu L4|T4] [--internet] [--kernel-src owner/slug ...] [--sub KEY=VALUE ...]
"""
import argparse, json, os, shutil, subprocess, sys, tempfile

USERNAMES = {"quynhtran35": "quynhtran35", "yuan1103": "yuan1103", "phatle1578": "phatle1578",
             "nguyenmanh0404": "nguyenmanh0404", "namhunhh": "namhunhh", "duckzuy": "duckzuykobip",
             "dinhthinh": "dinhthinh", "hieu": "hieuho669", "mxuanquy": "mxuanquy", "yeshenhi": "yeshenhi"}
COMPETITION = "gemma-4-developer-agent"  # competition id 149921: grants the L4 machine shape


def metadata(label, slug, code_file, gpu=None, internet=False, kernel_sources=(), dataset_sources=()):
    md = {"id": f"{USERNAMES[label]}/{slug}", "title": slug, "code_file": code_file, "language": "python",
          "kernel_type": "script", "is_private": True, "enable_gpu": bool(gpu), "enable_tpu": False,
          "enable_internet": bool(internet), "dataset_sources": list(dataset_sources),
          "competition_sources": [], "kernel_sources": list(kernel_sources), "model_sources": []}
    if gpu == "L4":  # operations procedure, section 4.2
        md.update(machine_shape="NvidiaL4", competition_sources=[COMPETITION], enable_internet=False)
    elif gpu == "T4":
        md.update(machine_shape="NvidiaTeslaT4", enable_internet=False)
    return md


def push(label, slug, code_path, gpu=None, internet=False, kernel_sources=(), subs=()):
    code = open(code_path).read()
    for kv in subs:
        k, v = kv.split("=", 1)
        code = code.replace(k, v)
    d = tempfile.mkdtemp(prefix=f"k_{label}_{slug}_")
    name = os.path.basename(code_path)
    open(os.path.join(d, name), "w").write(code)
    json.dump(metadata(label, slug, name, gpu, internet, kernel_sources), open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    r = subprocess.run(["kg", label, "kernels", "push", "-p", d], capture_output=True, text=True)
    shutil.rmtree(d, ignore_errors=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    print(f"[{label}] {slug}: {out[-1] if out else r.returncode}", flush=True)
    return r.returncode == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push"])
    ap.add_argument("label")
    ap.add_argument("slug")
    ap.add_argument("code")
    ap.add_argument("--gpu", choices=["L4", "T4"])
    ap.add_argument("--internet", action="store_true")
    ap.add_argument("--kernel-src", nargs="*", default=[])
    ap.add_argument("--sub", nargs="*", default=[])
    a = ap.parse_args()
    ok = push(a.label, a.slug, a.code, a.gpu, a.internet, a.kernel_src, a.sub)
    sys.exit(0 if ok else 1)
