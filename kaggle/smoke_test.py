"""Local end-to-end smoke test of the GPU-session bootstrap with REAL images and REAL deps artifacts.

Builds /kaggle/input exactly like a Kaggle session (nested mounts), runs the bootstrap script produced by
launch_session (same code that is pushed), then checks every expected descriptor set: files exist for the requested
trajectories, row counts equal the number of images, finite values, and no FAILED lines in the worker logs.

python3 kaggle/smoke_test.py --mini <dir containing vpr-space-v0> --deps <dir with core/hf/extra/anyloc> --test <name>
"""
import argparse, glob, json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "vprspace_repro"))
from launch_session import build_script  # noqa: E402
from configs import CONFIGS, outputs_of  # noqa: E402

TESTS = {
    "groupA": dict(deps=["core", "hf"], configs=["boq", "megaloc", "boq_bs256", "megaloc_bs128", "megaloc_322", "megaloc_224", "boq_224", "boq_518"]),
    "salad": dict(deps=["core"], configs=["salad_322", "salad_322_official", "salad_224", "salad_518"]),
    "salad_native": dict(deps=["core"], configs=["salad_native14"], trajs=["SPICE-A-night-on@2500.0us", "S3LI-Etna-crater_inout"]),
    "cosplace_mixvpr": dict(deps=["core", "extra"], configs=["cosplace_r50_2048_native", "cosplace_r50_2048_512", "cosplace_r101_2048_native",
                                                          "cosplace_r152_2048_native", "cosplace_r50_512_native", "cosplace_vgg16_512_native",
                                                          "cosplace_r50_2048_322", "cosplace_r50_2048_518", "mixvpr_4096", "mixvpr_512", "mixvpr_128"]),
    "netvlad_dinov3": dict(deps=["core", "extra", "hf"], configs=["netvlad_4096_native", "netvlad_32768_native", "netvlad_4096_max640",
                                                               "netvlad_4096_322", "netvlad_4096_518", "netvlad_32768_518",
                                                               "dinov3_s16_224", "dinov3_b16_224", "dinov3_b16_512", "dinov3_l16_224", "dinov3_l16_512"]),
    "netvlad_small": dict(deps=["core", "extra"], configs=["netvlad_4096_native", "netvlad_32768_native", "netvlad_4096_max640",
                                                        "netvlad_4096_322", "netvlad_4096_518", "netvlad_32768_518"],
                          trajs=["SPICE-A-night-on@2500.0us", "S3LI-Etna-crater_inout"]),  # native VGG16 on 2064x1544 exceeds local RAM
    "dinov3_320": dict(deps=["core", "hf"], configs=["dinov3_s16_320", "dinov3_b16_320", "dinov3_l16_320"]),
    "anyloc_fixed": dict(deps=["core", "anyloc"], configs=["anyloc_224", "anyloc_322", "anyloc_518"]),
    "anyloc_demo": dict(deps=["core", "anyloc"], configs=["anyloc_demo1024"], trajs=["SPICE-A-night-on@2500.0us", "S3LI-Etna-crater_inout"]),
    "densevlad": dict(deps=["core"], configs=["densevlad_16384"]),
}


def sh(cmd):
    subprocess.run(cmd, shell=True, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mini", required=True)
    ap.add_argument("--deps", required=True)
    ap.add_argument("--test", required=True, choices=list(TESTS))
    ap.add_argument("--work", default="/tmp/vprs_smoke")
    a = ap.parse_args()
    t = TESTS[a.test]
    # fresh session state (only our own scratch dirs)
    for d in ("/tmp/vprdata", "/tmp/torch_home", "/tmp/hf_home", "/tmp/vprcode", "/tmp/vprenv", a.work):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(a.work)
    inp = "/kaggle/input"
    for d in glob.glob(f"{inp}/*"):
        if os.path.islink(d) or os.path.isfile(d):
            os.remove(d)
        else:
            shutil.rmtree(d)
    # data: tar the mini dataset and split into exactly 11 parts, mounted one level deeper (like Kaggle)
    os.makedirs(f"{inp}/datasets_mount/u/vprspace-data-a", exist_ok=True)
    os.makedirs(f"{inp}/datasets_mount/u/vprspace-data-b", exist_ok=True)
    sh(f"cd {a.mini} && tar cf {a.work}/mini.tar vpr-space-v0 && split -n 11 -d -a 4 {a.work}/mini.tar {a.work}/part.")
    for i in range(11):
        dest = "vprspace-data-a" if i < 6 else "vprspace-data-b"
        shutil.move(f"{a.work}/part.{i:04d}", f"{inp}/datasets_mount/u/{dest}/vpr-space.tar.{i:04d}")
    for dep in t["deps"]:
        os.makedirs(f"{inp}/datasets_mount/u", exist_ok=True)
        os.symlink(os.path.join(a.deps, dep), f"{inp}/datasets_mount/u/vprspace-deps-{dep}")
    pixel_cost = any(isinstance(CONFIGS[c].get("size"), str) and not CONFIGS[c]["size"].startswith("pilbilinear") for c in t["configs"])
    spec = {"session": f"smoke-{a.test}", "env": "kaggle", "configs": t["configs"], "trajectories": t.get("trajs"), "pixel_cost": pixel_cost}
    if any(CONFIGS[c]["kind"] == "densevlad" for c in t["configs"]):
        spec["pre"] = [["{code}/densevlad_gpu.py", "--root", "{root}", "--desc-dir", "{desc}", "--stage", "vocab"]]
        spec["post"] = [["{code}/densevlad_gpu.py", "--root", "{root}", "--desc-dir", "{desc}", "--stage", "pca"]]
    open(f"{a.work}/run.py", "w").write(build_script(spec))
    env = dict(os.environ, VPRS_W=f"{a.work}/working")
    r = subprocess.run([sys.executable, "-I", f"{a.work}/run.py"], env=env, capture_output=True, text=True, cwd=a.work)
    open(f"{a.work}/bootstrap.log", "w").write(r.stdout + r.stderr)
    # ---- checks
    import numpy as np
    root = glob.glob("/tmp/vprdata/**/vpr_gt", recursive=True)
    root = os.path.dirname(root[0]) if root else None
    trajs = t.get("trajs") or sorted(os.listdir(f"{a.mini}/vpr-space-v0/images"))
    logs = "".join(open(f).read() for f in glob.glob(f"{a.work}/working/logs/worker*.log"))
    problems = []
    if r.returncode != 0:
        problems.append(f"bootstrap exit {r.returncode}: {(r.stdout + r.stderr)[-1500:]}")
    for line in logs.splitlines():
        if "FAILED" in line or "Traceback" in line:
            problems.append(line[:300])
    for c in t["configs"]:
        for name in outputs_of(c):
            for tr in trajs:
                f = f"{a.work}/working/desc/{name}/{tr}.npy"
                n_img = len(glob.glob(f"{a.mini}/vpr-space-v0/images/{tr}/*.png"))
                if not os.path.exists(f):
                    problems.append(f"missing {name}/{tr}")
                    continue
                x = np.load(f)
                if x.shape[0] != n_img or not np.isfinite(x).all() or x.ndim != 2:
                    problems.append(f"bad {name}/{tr}: shape {x.shape}, finite {np.isfinite(x).all()}")
                else:
                    print(f"  ok {name:28s} {tr:28s} {x.shape}")
    print(f"SMOKE {a.test}: {'PASS' if not problems else 'FAIL'}")
    for p in problems[:30]:
        print("   ", p)
    if "hf plain" in logs or "hf fallback" in logs:
        print("   (note) HF plain/fallback resolution used:", sorted({l.split('] ')[-1][:80] for l in logs.splitlines() if 'hf plain' in l})[:3])
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
