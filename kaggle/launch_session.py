"""Build and push a VPR-Space GPU session (bootstrap + embedded vprspace_repro code + spec).

python3 kaggle/launch_session.py <label> <session-name> --gpu L4|T4 --configs c1 c2 ... [--env pinned|kaggle]
        [--deps core extra anyloc] [--trajectories t1 t2 ...] [--dry-run out.py]
"""
import argparse, base64, io, json, os, sys, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from launch import USERNAMES, metadata  # noqa: E402

CODE_FILES = ["configs.py", "worker.py", "metrics_all.py", "paper_reference.json", "densevlad_gpu.py",
              "densevlad_vlfeat.py", "vlfeat_phow/phow.py", "vlfeat_phow/phow_shim.c",
              "upstream/dataset.py", "upstream/evaluate.py", "upstream/methods.py", "upstream/metrics.py"]


def code_b64():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in CODE_FILES:
            tf.add(os.path.join(REPO, "vprspace_repro", rel), arcname=rel)
    return base64.b64encode(buf.getvalue()).decode()


def build_script(spec):
    tpl = open(os.path.join(HERE, "kernels", "gpu_run", "bootstrap.py")).read()
    return tpl.replace("__SPEC__", json.dumps(spec)).replace("__CODE_B64__", code_b64())


def push_session(label, session, gpu, configs, env="pinned", deps=("core",), trajectories=None, dry_run=None):
    import shutil, subprocess, tempfile
    sys.path.insert(0, os.path.join(REPO, "vprspace_repro"))
    from configs import CONFIGS
    bad = [c for c in configs if c not in CONFIGS]
    assert not bad, f"unknown configs {bad}"
    pixel_cost = any(isinstance(CONFIGS[c].get("size"), str) and not CONFIGS[c]["size"].startswith("pilbilinear") for c in configs)
    spec = {"session": session, "env": env, "configs": configs, "trajectories": trajectories, "pixel_cost": pixel_cost}
    if any(CONFIGS[c]["kind"] == "densevlad" for c in configs):
        spec["pre"] = [["{code}/densevlad_gpu.py", "--root", "{root}", "--desc-dir", "{desc}", "--stage", "vocab"]]
        spec["post"] = [["{code}/densevlad_gpu.py", "--root", "{root}", "--desc-dir", "{desc}", "--stage", "pca"]]
    script = build_script(spec)
    if dry_run:
        open(dry_run, "w").write(script)
        return True
    user = USERNAMES[label]
    sources = [f"{user}/vprspace-data-a", f"{user}/vprspace-data-b"] + [f"{user}/vprspace-deps-{d}" for d in deps]
    slug = f"vprspace-run-{session}"
    d = tempfile.mkdtemp(prefix=f"s_{label}_")
    open(os.path.join(d, "run.py"), "w").write(script)
    json.dump(metadata(label, slug, "run.py", gpu=gpu, kernel_sources=sources), open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    r = subprocess.run(["kg", label, "kernels", "push", "-p", d], capture_output=True, text=True)
    shutil.rmtree(d, ignore_errors=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    print(f"[{label}] {slug}: {out[-1] if out else r.returncode}", flush=True)
    return r.returncode == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("session")
    ap.add_argument("--gpu", choices=["L4", "T4"], default="L4")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--env", default="pinned", choices=["pinned", "kaggle"])
    ap.add_argument("--deps", nargs="+", default=["core"])
    ap.add_argument("--trajectories", nargs="*")
    ap.add_argument("--dry-run")
    a = ap.parse_args()
    sys.exit(0 if push_session(a.label, a.session, a.gpu, a.configs, a.env, a.deps, a.trajectories, a.dry_run) else 1)
